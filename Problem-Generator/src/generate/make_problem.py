"""
Problem + short clue (context.md section 3.2, step 3): turns a statement into the problem the test models get,
the answer we expect, the reason it is right, and a short clue: the same idea happening in everyday life.

Usage (from the repo root):
    python src/generate/make_problem.py --auto CHEM-1      # one statement: problem + clue, no questions asked
    python src/generate/make_problem.py --auto             # every statement that still needs them
    python src/generate/make_problem.py --auto --again CHEM-1   # a fresh problem + clue for one that has them
    python src/generate/make_problem.py CHEM-1             # you choose the problem (see "By hand" below)
    python src/generate/make_problem.py                    # paste a statement instead

How a problem is made:
    1. The writer (gpt-oss) writes the key step, the facts to hide, the problem, the answer and the reason.
    2. The checker (gpt-oss, temperature 0) reads the problem next to the team's statement and principle: does
       it give the idea away, rule out the real answer, or say something the statement doesn't? If so, the
       writer is told why and tries again, up to ATTEMPTS times.

How a clue is made, once a problem is kept:
    1. The writer splits the principle into its parts (who does what), maps each part to an everyday event and
       writes the clue (2-3 sentences).
    2. The checker reads it: is a part missing or done by the wrong kind of actor, is it told as a lesson, or does
       it look too much like the problem on the surface? Same retries.

Automatic (--auto), for each statement in data/sources/statements.csv (or only the codes given):
    - skipped if the memory check says DROP, if its verdict isn't Keep (unless you name its code), or if it
      already has a kept problem and clue
    - the first problem that passes the checker is kept; if none passes in ATTEMPTS tries, the statement is
      listed as needing a person (tried again only with --retry; or run it by hand and use E)
    - then the first clue that passes is kept; if none passes, it is listed (tried again only with --retry)
    - writes data/candidates/problems_review_<date_time>.csv: every statement with its problem, answer, clue and
      status, to read through in one sheet
    - stops at Groq's daily limit; run the same command later and it carries on

By hand (a code without --auto): you see each problem version and choose.
    Enter = another version, E = write or edit one yourself (start from a version; its answer, reason and key
    step carry over, and the checker still reads it), a number = keep that version (its clue is then written
    automatically), Q = quit without keeping one

Everything goes to data/candidates/<code>/problem_<date_time>/: statement.json, versions/ (every problem try),
kept.json (the kept problem), clues/ (every clue try) and clue.json (the kept clue). Nothing is overwritten;
the newest folder with a kept problem counts.
"""

import argparse
import csv
import datetime as dt
import json
import sys
import textwrap

from llm import ROOT, DailyLimit, load_model_config, logged_call, make_client, parse_json_reply
from statements import StatementError, get_item, load_statements, read_statement

PROMPT_DIR = ROOT / "src" / "generate" / "prompts"
PROMPTS = {"write": PROMPT_DIR / "problem_system.txt", "check": PROMPT_DIR / "problem_check_system.txt",
           "clue": PROMPT_DIR / "clue_system.txt", "clue_check": PROMPT_DIR / "clue_check_system.txt"}
OUT_DIR = ROOT / "data" / "candidates"
TEXT_FIELDS = ("key_step", "problem", "answer", "reason")   # plus "hide": the facts left out of the problem
PROBLEM_CHECKS = {"gives_away": "GIVES THE IDEA AWAY", "blocks_answer": "RULES OUT THE ANSWER",
                  "wrong_facts": "SAYS WHAT THE STATEMENT DOESN'T", "too_technical": "TOO TECHNICAL",
                  "confusing": "THE QUESTION DOESN'T FIT"}
CLUE_CHECKS = {"missing_part": "MISSES OR MISPLACES A PART OF THE IDEA", "looks_like_trap": "READS AS THE TRAP",
               "states_lesson": "TOLD AS A LESSON",
               "too_close": "TOO CLOSE TO THE PROBLEM"}   # whether it works is for the difficulty filter
ATTEMPTS = 3   # writer tries before giving up (the last try is shown with the checker's warning)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:  # "x" = fail rather than overwrite
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def show_path(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def last_memory_check(code):
    """RESULT line of the newest memory check for this code, or None."""
    summaries = sorted((OUT_DIR / code).glob("memory_*/summary.json"))   # folder names sort by date
    if not summaries:
        return None
    return load_json(summaries[-1])["result"]


def bullets(items):
    return "\n".join(f"- {x}" for x in items)


def wrapped(label, text):
    print(textwrap.fill(text, width=110, initial_indent=f"{label:<10}", subsequent_indent=" " * 10))


# ---------- model calls ----------

def call(client, cfg, step, user):
    """One call for `step` (a key of PROMPTS). Returns (log record, parsed reply or None).
    Raises DailyLimit, or the error if the API keeps failing (see llm.logged_call)."""
    record, raw = logged_call(client, cfg, PROMPTS[step], user)
    return record, parse_json_reply(raw)


def clean_check(reply, checks):
    if not isinstance(reply, dict) or not all(isinstance(reply.get(k), bool) for k in checks):
        return None
    check = {k: reply[k] for k in checks}
    if isinstance(reply.get("reader_sees"), str):   # clue checker: what a reader takes the scene to mean
        check["reader_sees"] = reply["reader_sees"].strip()
    return {**check, "why": str(reply.get("why") or "").strip()}


def passed(check):
    """True if the checker found nothing wrong (none of its true/false marks is true)."""
    return check is not None and not any(v is True for v in check.values())


def fault_of(result, checks):
    if result is None:
        return "no usable reply from the writer"
    if result["check"] is None:
        return "the checker's reply was unreadable"
    return " and ".join(label for k, label in checks.items() if result["check"][k]) + f": {result['check']['why']}"


def verdict_text(result, checks):
    if passed(result["check"]):
        return f"OK (try {result['tries']} of {ATTEMPTS})" if result.get("tries") else "OK"
    if result["check"] is None:
        return "NOT CHECKED: the checker call failed or its reply was unreadable; check it yourself"
    after = f" after {result['tries']} tries" if result.get("tries") else ""
    return f"WARNING{after}: {fault_of(result, checks)}"


def run_tries(record, path, write, check):
    """Up to ATTEMPTS tries: write(rejected) -> (log, draft or None), then check(draft) -> (log, check or None).
    A rejected draft goes back to the writer with the checker's reason. Every try goes into record["tries"] and the
    record is saved to `path`, also when Groq's daily limit stops it (DailyLimit is then raised after saving).
    Returns the first draft that passed, else the last readable one, with its "check" and "tries"; or None."""
    result, rejected, limit = None, None, None
    try:
        for _ in range(ATTEMPTS):
            log, draft = write(rejected)
            attempt = {"write_call": log, "draft": draft}
            record["tries"].append(attempt)
            print(".", end="", flush=True)
            if draft is None:
                continue
            log, verdict = check(draft)
            attempt.update(check_call=log, check=verdict)
            result = {**draft, "check": verdict, "tries": len(record["tries"])}
            if passed(verdict):
                break
            rejected = (draft, verdict["why"]) if verdict and verdict["why"] else None
    except DailyLimit as e:
        limit = e
    except Exception as e:   # the API kept failing: keep what we have
        print(f"\n  The API call failed: {str(e)[:300]}")
    record.update(result=result, finished=limit is None)
    if record["tries"]:
        save_json(path, record)
    if limit:
        raise limit
    return result


# ---------- problem ----------

def problem_message(item, earlier, rejected):
    trap = item["trap"] or "(none given: rule out the most obvious textbook answer yourself)"
    user = f"STATEMENT: {item['statement']}\nPRINCIPLE: {item['principle']}\nTRAP: {trap}"
    if earlier:
        user += ("\n\nYou already wrote these problems. Write a clearly different one: new wording and a "
                 "different way into the situation. The key step stays the same.\n" + bullets(earlier))
    if rejected:
        draft, why = rejected
        user += (f"\n\nA checker rejected your last try.\nPROBLEM: {draft['problem']}\nWHY: {why}\n"
                 "Write it again fixing every fault named. Anything that gives the idea away or is not in the "
                 "statement must go completely; rewording it is not enough. Specialist words must become "
                 "everyday ones.")
    return user


def clean_draft(reply):
    """The writer's reply with stripped text, or None if a field is missing or empty."""
    if not isinstance(reply, dict) or not isinstance(reply.get("hide"), list):
        return None
    if not all(isinstance(reply.get(k), str) and reply[k].strip() for k in TEXT_FIELDS):
        return None
    draft = {k: reply[k].strip() for k in TEXT_FIELDS}
    draft["hide"] = [str(h).strip() for h in reply["hide"] if str(h).strip()]
    return draft


def check_problem(clients, cfgs, item, problem, answer):
    """(log record, check or None). The checker judges against the team's statement and principle, not the
    writer's own key step."""
    log, reply = call(clients["check"], cfgs["check"], "check",
                      f"STATEMENT: {item['statement']}\nPRINCIPLE: {item['principle']}\n"
                      f"ANSWER: {answer}\n\nPROBLEM: {problem}")
    return log, clean_check(reply, PROBLEM_CHECKS)


def write_version(clients, cfgs, item, earlier, folder, number, quiet=False):
    """One problem version (writer + checker, up to ATTEMPTS tries), saved as versions/v<number>.json.
    Returns it, or None. Raises DailyLimit after saving. quiet=True prints only progress dots."""
    if not quiet:
        print(f"\n  Writing version {number} ", end="", flush=True)
    earlier_problems = [e["problem"] for e in earlier]
    record = {"version": number, "created": now(), "earlier_problems_shown": earlier_problems, "tries": []}

    def write(rejected):
        log, reply = call(clients["write"], cfgs["write"], "write", problem_message(item, earlier_problems, rejected))
        return log, clean_draft(reply)

    version = run_tries(record, folder / "versions" / f"v{number}.json", write,
                        lambda draft: check_problem(clients, cfgs, item, draft["problem"], draft["answer"]))
    if not quiet:
        print()
        if version is None:
            print("  No usable reply this time. Press Enter for another version.")
        else:
            show_version(number, version)
    return version


def hand_version(clients, cfgs, item, good, folder, number):
    """A problem you write or edit yourself, starting from an earlier version: its answer, reason and key step
    carry over (you can change the answer). The checker reads it like any other. Saved as versions/v<number>.json.
    Returns the version, or None if you cancel."""
    latest = max(good)
    while True:
        choice = text_input(f"\nStart from which version? (Enter = {latest}): ") or str(latest)
        if choice.isdigit() and int(choice) in good:
            break
        print("  Type the number of a version shown above.")
    base = good[int(choice)]
    print(f"\nIts problem, to copy and edit:\n{base['problem']}\n")
    print("Type or paste your problem, then press Enter on an empty line (an empty problem cancels):")
    lines = []
    while line := text_input(""):
        lines.append(line)
    problem = " ".join(lines)
    if not problem:
        print("  Cancelled.")
        return None
    answer = text_input(f"\nAnswer (Enter keeps this one: {base['answer']})\n> ") or base["answer"]

    print("  Checking...")
    try:
        log, check = check_problem(clients, cfgs, item, problem, answer)
    except Exception as e:   # the API kept failing (or the daily limit): save it unchecked
        print(f"  The checker call failed: {str(e)[:300]}")
        log, check = None, None
    version = {"key_step": base["key_step"], "hide": base["hide"], "problem": problem, "answer": answer,
               "reason": base["reason"], "check": check, "based_on": int(choice)}
    save_json(folder / "versions" / f"v{number}.json", {
        "version": number, "created": now(), "by_hand": True, "based_on": int(choice), "check_call": log,
        "result": version})
    show_version(number, version)
    return version


def show_version(number, version):
    by_hand = f" (yours, from version {version['based_on']})" if version.get("based_on") else ""
    print(f"\n--- Version {number}{by_hand} ---")
    for label, key in (("Problem:", "problem"), ("Answer:", "answer"), ("Reason:", "reason"),
                       ("Key step:", "key_step")):
        wrapped(label, version[key])
    wrapped("Check:", verdict_text(version, PROBLEM_CHECKS))


# ---------- clue ----------

def clean_clue(reply):
    """The writer's clue reply with stripped text, or None if a field is missing or empty."""
    if not isinstance(reply, dict) or not isinstance(reply.get("clue"), str) or not reply["clue"].strip():
        return None
    if not isinstance(reply.get("parts"), list) or not isinstance(reply.get("mapping"), list):
        return None
    mapping = [{k: str(m.get(k, "")).strip() for k in ("part", "problem", "clue")}
               for m in reply["mapping"] if isinstance(m, dict)]
    return {"parts": [str(p).strip() for p in reply["parts"] if str(p).strip()], "mapping": mapping,
            "clue": reply["clue"].strip()}


def clue_status(folder):
    """'kept', 'none passed' (a finished run of tries kept nothing) or 'not done', for the problem in `folder`."""
    if (folder / "clue.json").exists():
        return "kept"
    if any(load_json(p).get("finished") for p in (folder / "clues").glob("c*.json")):
        return "none passed"
    return "not done"


def write_clue(clients, cfgs, item, problem, folder, quiet=False):
    """The clue for the kept `problem` (writer + clue checker, up to ATTEMPTS tries), saved as clues/c<n>.json; a
    clue that passes also goes to clue.json. Returns the clue, or None. Raises DailyLimit after saving."""
    number = len(list((folder / "clues").glob("c*.json"))) + 1
    trap = item.get("trap") or "(none given: the most obvious textbook answer)"
    base = (f"STATEMENT: {item['statement']}\nPRINCIPLE: {item['principle']}\nTRAP: {trap}\n"
            f"PROBLEM: {problem['problem']}\nANSWER: {problem['answer']}")
    record = {"run": number, "created": now(), "for_problem": problem["problem"], "tries": []}

    def write(rejected):
        user = base
        if rejected:
            draft, why = rejected
            user += (f"\n\nA checker rejected your last try.\nCLUE: {draft['clue']}\nWHY: {why}\n"
                     "Write it again fixing that. If the scene itself can't be made right, pick a different "
                     "everyday scene.")
        log, reply = call(clients["write"], cfgs["write"], "clue", user)
        return log, clean_clue(reply)

    def check(draft):
        log, reply = call(clients["check"], cfgs["check"], "clue_check",
                          f"PRINCIPLE: {item['principle']}\nANSWER: {problem['answer']}\nTRAP: {trap}\n\n"
                          f"PROBLEM: {problem['problem']}\n\nCLUE: {draft['clue']}")
        verdict = clean_check(reply, CLUE_CHECKS)
        if verdict and not passed(verdict) and verdict.get("reader_sees"):   # goes back to the writer with `why`
            verdict["why"] += f" (A reader takes the scene to mean: {verdict['reader_sees']})"
        return log, verdict

    if not quiet:
        print("\n  Writing the clue ", end="", flush=True)
    clue = run_tries(record, folder / "clues" / f"c{number}.json", write, check)
    if clue and passed(clue["check"]):
        save_json(folder / "clue.json", {"kept_at": now(), "kept_by": "auto (first try that passed the checker)",
                                         **clue})
    if not quiet:
        print()
        if clue is None:
            print("  No usable clue this time.")
        else:
            print("\n--- Clue ---")
            wrapped("Clue:", clue["clue"])
            wrapped("Parts:", "; ".join(f"{m['part']} = {m['clue']}" for m in clue["mapping"]))
            wrapped("Check:", verdict_text(clue, CLUE_CHECKS))
            if not passed(clue["check"]):
                print(f"  Not kept. Try again later with: python src/generate/make_problem.py --auto --retry "
                      f"{item['code']}")
    return clue


# ---------- by hand ----------

def text_input(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def ask_choice(good):
    """Enter, Q, E (only once there is a version to start from) or the number of a version."""
    while True:
        try:
            choice = input("\n[Enter] another version   [number] keep that version   "
                           "[E] write or edit one yourself   [Q] quit: ").strip().upper()
        except EOFError:
            return "Q"
        if choice in ("", "Q") or (choice.isdigit() and int(choice) in good) or (choice == "E" and good):
            return choice
        if choice == "E":
            print("  There is no version to start from yet: press Enter to write one first.")
        else:
            print("  Press Enter, or type E, Q, or the number of a version shown above.")


def new_folder(code):
    return OUT_DIR / code / f"problem_{dt.datetime.now():%Y%m%d_%H%M%S}"


def run_interactive(clients, cfgs, code):
    """One statement: you see each problem version and choose; the clue is then written automatically."""
    item = get_item(code)
    if item is None:
        return
    memory = last_memory_check(item["code"])
    if memory:
        print(f"  Memory check: {memory}")
    else:
        print(f"  Note: no memory check found for {item['code']}. Run it first (context.md section 3.2, step 4).")
    if not item["trap"]:
        print("  Note: no trap given, so the writer picks the obvious answer to rule out.")

    folder = new_folder(item["code"])
    save_json(folder / "statement.json", item)
    print(f"Writer and checker: {cfgs['write']['model']} via {cfgs['write']['provider']}. "
          f"Saving to {show_path(folder)}")

    good = {}      # version number -> version
    choice = ""    # start by writing a first version
    while True:
        number = len(list((folder / "versions").glob("v*.json"))) + 1
        try:
            if choice == "E":
                version = hand_version(clients, cfgs, item, good, folder, number)
            else:
                version = write_version(clients, cfgs, item, list(good.values()), folder, number)
        except DailyLimit as e:
            print(f"\n  {e}")
            version = None
        if version:
            good[number] = version
        choice = ask_choice(good)
        if choice == "Q":
            print(f"Nothing kept. Every version is saved in {show_path(folder / 'versions')}")
            return
        if choice.isdigit():
            break

    kept = int(choice)
    note = text_input("Why this one? (optional, Enter to skip): ")
    kept_by = "person (own wording)" if good[kept].get("based_on") else "person"
    save_json(folder / "kept.json", {"version": kept, "kept_at": now(), "kept_by": kept_by, "note": note,
                                     **good[kept]})
    print(f"Kept version {kept}: {show_path(folder / 'kept.json')}")
    try:
        write_clue(clients, cfgs, item, good[kept], folder)
    except DailyLimit as e:
        print(f"\n  {e} Add the clue later with: python src/generate/make_problem.py --auto {item['code']}")


# ---------- automatic ----------

def status(code):
    """('kept', kept.json) if the statement has a kept problem (the newest wins), else ('needs a person',
    no_pass.json) if an automatic run found no problem that passed, else ('not done', None)."""
    for state, name in (("kept", "kept.json"), ("needs a person", "no_pass.json")):
        files = sorted((OUT_DIR / code).glob(f"problem_*/{name}"))   # folder names sort by date
        if files:
            return state, files[-1]
    return "not done", None


def run_auto(clients, cfgs, codes, retry, again):
    """Every statement that needs a problem or a clue (or only `codes`): write, check, keep the first that passes.
    again=True: a fresh problem and clue for the named codes, even if they have them (the old ones stay saved)."""
    statements = load_statements()
    named = [c.upper() for c in codes]
    unknown = [c for c in named if c not in statements]
    if unknown:
        sys.exit(f"Not in data/sources/statements.csv: {', '.join(unknown)}")

    todo = []
    for code, row in statements.items():
        if named and code not in named:
            continue
        state, file = status(code)
        clue_state = clue_status(file.parent) if state == "kept" else None
        if (last_memory_check(code) or "").startswith("DROP"):
            reason = "the memory check says DROP"
        elif row.get("verdict", "").lower() != "keep" and code not in named:
            reason = f"verdict is '{row.get('verdict') or 'blank'}', not Keep"
        elif again:
            todo.append(code)
            continue
        elif clue_state == "kept":
            reason = "already has a problem and a clue"
        elif clue_state == "none passed" and not retry:
            reason = "no clue passed last time (add --retry)"
        elif state == "needs a person" and not retry:
            reason = "no problem passed last time (add --retry, or run it by hand and use E)"
        else:
            todo.append(code)
            continue
        print(f"  skip {code}: {reason}")
    print(f"\n{len(todo)} to do. Writer and checker: {cfgs['write']['model']} via {cfgs['write']['provider']}, "
          f"up to {ATTEMPTS} tries each.\n")

    done = {"problem kept": 0, "no problem": [], "clue kept": 0, "no clue": [], "unusable": 0}
    try:
        for i, code in enumerate(todo, 1):
            state, file = status(code)
            if state == "kept" and not again:   # the clue must match the problem's own statement record
                folder, problem = file.parent, load_json(file)
                item = load_json(folder / "statement.json")
                print(f"[{i}/{len(todo)}] {item['title']} | problem: kept earlier", end="", flush=True)
            else:
                try:
                    item = read_statement(code, statements, quiet=True)
                except StatementError as e:
                    print(f"[{i}/{len(todo)}] skipped: {e}")
                    done["unusable"] += 1
                    continue
                print(f"[{i}/{len(todo)}] {item['title']} | problem ", end="", flush=True)
                folder = new_folder(code)
                save_json(folder / "statement.json", {**item, "mode": "auto"})
                problem = write_version(clients, cfgs, item, [], folder, 1, quiet=True)
                if not (problem and passed(problem["check"])):
                    save_json(folder / "no_pass.json", {"at": now(), "why": fault_of(problem, PROBLEM_CHECKS),
                                                        "last_version": problem})
                    print(f" NEEDS A PERSON: {fault_of(problem, PROBLEM_CHECKS)}")
                    done["no problem"].append(code)
                    continue
                save_json(folder / "kept.json", {"version": 1, "kept_at": now(), "note": "",
                                                 "kept_by": "auto (first try that passed the checker)", **problem})
                print(f" kept (try {problem['tries']})", end="", flush=True)
                done["problem kept"] += 1

            print(" | clue ", end="", flush=True)
            clue = write_clue(clients, cfgs, item, problem, folder, quiet=True)
            clue_kept = clue and passed(clue["check"])
            if clue_kept:
                print(f" kept (try {clue['tries']})")
                done["clue kept"] += 1
            else:
                print(f" NONE PASSED: {fault_of(clue, CLUE_CHECKS)}")
                done["no clue"].append(code)
            wrapped("Problem:", problem["problem"])
            wrapped("Answer:", problem["answer"])
            if clue_kept:
                wrapped("Clue:", clue["clue"])
            print()
    except DailyLimit as e:
        print(f"\n\nSTOPPED: {e} Run the same command later; it carries on where it stopped.")
    except KeyboardInterrupt:
        print("\n\nSTOPPED by you. Run the same command again to carry on.")

    review = write_review(statements)
    print(f"\nThis run: {done['problem kept']} problems kept, {len(done['no problem'])} need a person; "
          f"{done['clue kept']} clues kept, {len(done['no clue'])} with none passing"
          + (f"; {done['unusable']} unusable rows in the CSV" if done["unusable"] else "") + ".")
    print(f"Review sheet (every statement): {show_path(review)}")
    for code in done["no problem"]:
        print(f"  {code}: no problem passed. See the versions, or write it with E: "
              f"python src/generate/make_problem.py {code}")
    for code in done["no clue"]:
        print(f"  {code}: no clue passed. Try again: python src/generate/make_problem.py --auto --retry {code}")


REVIEW_FIELDS = ["code", "title", "verdict", "memory_check", "status", "kept_by", "problem", "answer", "reason",
                 "key_step", "checker_note", "clue_status", "clue", "clue_parts", "clue_note", "folder"]


def write_review(statements):
    """One sheet with every statement's current problem, clue and status, to read through (e.g. in Google
    Sheets). A new file each time, so notes made in an older one are never overwritten."""
    path = OUT_DIR / f"problems_review_{dt.datetime.now():%Y%m%d_%H%M%S}.csv"
    rows = []
    for code, row in statements.items():
        state, file = status(code)
        memory = (last_memory_check(code) or "not run").split(":")[0]
        if state == "not done" and memory == "DROP":
            state = "dropped by the memory check"
        record = load_json(file) if file else {}
        version = record if state == "kept" else (record.get("last_version") or {})
        clue_state, clue, clue_note = "", {}, ""
        if state == "kept":
            folder = file.parent
            clue_state = clue_status(folder)
            if clue_state == "kept":
                clue = load_json(folder / "clue.json")
            elif clue_state == "none passed":
                runs = sorted((folder / "clues").glob("c*.json"), key=lambda p: int(p.stem[1:]))
                clue_note = fault_of(load_json(runs[-1]).get("result"), CLUE_CHECKS)
        rows.append({
            "code": code, "title": row["title"], "verdict": row.get("verdict", ""), "memory_check": memory,
            "status": state, "kept_by": record.get("kept_by", ""),
            **{k: version.get(k, "") for k in ("problem", "answer", "reason", "key_step")},
            "checker_note": record.get("why", "") if state == "needs a person" else "",
            "clue_status": clue_state, "clue": clue.get("clue", ""),
            "clue_parts": "; ".join(f"{m['part']} = {m['clue']}" for m in clue.get("mapping", [])),
            "clue_note": clue_note,
            "folder": show_path(file.parent) if file else ""})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8-sig", newline="") as f:   # -sig so Excel shows the symbols right
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------- main ----------

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Write the problem, expected answer, reason and short clue "
                                                 "for statements.")
    parser.add_argument("codes", nargs="*", metavar="CODE",
                        help="statement code(s) in data/sources/statements.csv, e.g. CHEM-1. Without --auto: one "
                             "code, or none to paste a statement")
    parser.add_argument("--auto", action="store_true",
                        help="no questions: write, check and keep a problem and clue for every statement that "
                             "needs them (or only the codes given)")
    parser.add_argument("--retry", action="store_true",
                        help="with --auto: also retry statements where nothing passed last time")
    parser.add_argument("--again", action="store_true",
                        help="with --auto and codes: a fresh problem and clue even if they already have them "
                             "(the old ones stay saved; the newest counts)")
    args = parser.parse_args()
    if not args.auto and len(args.codes) > 1:
        parser.error("give one code, or add --auto for several")
    if (args.retry or args.again) and not args.auto:
        parser.error("--retry and --again go with --auto")
    if args.again and not args.codes:
        parser.error("--again needs the codes to redo, e.g. --auto --again CHEM-1")

    cfgs = {"write": load_model_config("problem"), "check": load_model_config("judge")}
    clients = {role: make_client(cfg) for role, cfg in cfgs.items()}
    if args.auto:
        run_auto(clients, cfgs, args.codes, args.retry, args.again)
    else:
        run_interactive(clients, cfgs, args.codes[0] if args.codes else None)


if __name__ == "__main__":
    main()
