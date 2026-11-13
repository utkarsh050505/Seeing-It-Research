r"""
Difficulty filter (context.md section 3.2, step 5): is a problem hard alone, but solvable with its clue?

Run it on the lab PC from the repo root, with our own Ollama running in another window (as for the memory check):
    $env:OLLAMA_HOST="127.0.0.1:11500"; $env:OLLAMA_MODELS="E:\LLMxPsychology-models"; ollama serve
Then:
    python src/generate/difficulty_filter.py CHEM-1                  # the newest kept problem + clue for CHEM-1
    python src/generate/difficulty_filter.py CHEM-1 --groups weak    # only these strength groups
    python src/generate/difficulty_filter.py CHEM-1 --new            # a fresh run, even if the last one is unfinished
    python src/generate/difficulty_filter.py CHEM-1 --tries 5        # fewer tries than the config (saves Groq tokens)
    python src/generate/difficulty_filter.py CHEM-1 --tries 10       # after a 5-try run: adds tries 6-10 to it
    python src/generate/difficulty_filter.py CHEM-1 --rejudge        # mark the last run's answers again (new judge)
A run that stops (AnyDesk, restart, Groq's daily limit) carries on when you run the same command again.
Adding tries to a run keeps its first tries; try t always uses seed + t, so 5 + 5 gives the same answers as 10.
--rejudge makes a new filter folder that marks the last run's saved answers again with the current judge prompt
(Ollama isn't needed); the old marks stay where they are.

How it works:
1. Every local test model in the chosen groups answers the problem ALONE and WITH THE CLUE ("this short story may
   help you; use it"), `tries` times each (config: difficulty_filter), each in a fresh chat with its own seed,
   thinking off where it can be switched off. A group answers its own problem version if one was kept
   (kept_<group>.json next to kept.json), else the shared one.
2. The judge (gpt-oss-120b, temperature 0) marks every answer: does its main solution have every part of the key
   idea (the parts saved with the clue: what happens, to what, done by whom)? It sees the problem, the principle,
   its parts, the expected answer and the reply, but not whether the clue was given.
3. Per strength group, with the tries of its models pooled: PASS if solved alone in at most alone_max of the tries
   and, for the groups in clue_groups, with the clue in at least clue_min. The other groups' clue result is only
   reported (decided 12 Nov).

Everything goes to data/candidates/<code>/problem_<...>/filter_<date_time>/, next to the problem and clue it tested:
run.json (exact texts and settings), every answer (answers/), every mark (marks/) and the summary. Nothing saved is
redone or overwritten.
"""

import argparse
import datetime as dt
import json
import sys
import time

import yaml

from llm import CONFIG_PATH, ROOT, DailyLimit, load_model_config, logged_call, make_client, parse_json_reply
from local_llm import START_CMD, ask, load_test_config, server_info, unload
from make_problem import last_memory_check, load_json, now, save_json, show_path, status

JUDGE_PROMPT = ROOT / "src" / "generate" / "prompts" / "solve_judge_system.txt"
GROUPS = ("strong", "middle", "weak")
KINDS = ("alone", "clue")

# Fixed wording around the problem and clue, so every statement is asked the same way.
ALONE_QUESTION = "Here is a problem:\n\n{problem}\n\nWhat is your best solution? Answer in one short paragraph."
CLUE_QUESTION = ("Here is a problem:\n\n{problem}\n\nThis short story may help you solve it:\n\n{clue}\n\n"
                 "Use the story to find your best solution. Answer in one short paragraph.")
JUDGE_ATTEMPTS = 3   # tries for a readable mark (each with a new seed), on top of the API retries in llm.logged_call

# For the estimate printed before a run. From the 11 Nov memory check on the lab PC: OLMo Think took ~57 s per
# answer, the other models 2-8 s; its marks took ~1,100 Groq tokens each, and filter answers are a little longer.
SECONDS_PER_ANSWER = {"always": 60}
SECONDS_PER_ANSWER_OTHER = 6
TOKENS_PER_MARK = 1300


def file_name(model, kind, t):
    return f"{model['id'].replace(':', '_').replace('/', '_')}__{kind}__{t}.json"


def pct(part, whole):
    return f"{part}/{whole} ({part / whole:.0%})" if whole else "-"


# ---------- which problem, which run ----------

def problem_folder(code):
    """The newest folder with a kept problem and a kept clue for this statement."""
    state, kept = status(code)
    if state != "kept":
        sys.exit(f"{code} has no kept problem yet. Make one first: python src/generate/make_problem.py --auto {code}")
    folder = kept.parent
    if not (folder / "clue.json").exists():
        sys.exit(f"{code}'s newest problem has no kept clue yet: python src/generate/make_problem.py --auto --retry {code}")
    return folder


def problem_for(folder, group):
    """(file name, kept problem) that a group answers: its own version if one was kept, else the shared one."""
    own = folder / f"kept_{group}.json"
    path = own if own.exists() else folder / "kept.json"
    return path.name, load_json(path)


def last_run(folder, groups):
    """The newest filter run for this problem if it covers the same groups, else None."""
    runs = sorted(folder.glob("filter_*"))   # folder names sort by date
    if not runs or not (runs[-1] / "run.json").exists():
        return None
    return runs[-1] if load_json(runs[-1] / "run.json")["groups"] == groups else None


def load_run(run_folder):
    """run.json, with its tries raised by any later top-up (more_tries_*.json; run.json itself is never rewritten)."""
    run = load_json(run_folder / "run.json")
    for top_up in sorted(run_folder.glob("more_tries_*.json")):
        run["tries"] = max(run["tries"], load_json(top_up)["tries"])
    return run


def is_complete(run_folder):
    path = run_folder / "summary.json"
    return path.exists() and bool(load_json(path).get("complete"))


def answer_path(run_folder, run, name):
    """Where an answer is: this run's answers/ folder, or for a re-marking run, the run whose answers it marks."""
    own = run_folder / "answers" / name
    if own.exists() or not run.get("answers_from"):
        return own
    return ROOT / run["answers_from"] / name


def answer_files(run_folder, run):
    """{file name: path} of every saved answer this run marks."""
    files = {}
    for folder in ([ROOT / run["answers_from"]] if run.get("answers_from") else []) + [run_folder / "answers"]:
        files.update({p.name: p for p in folder.glob("*.json")})
    return files


def run_parts(run):
    """The parts of the key idea the judge checks (saved with the clue; older runs read them from clue.json)."""
    parts = run.get("parts") or load_json(ROOT / run["problem_folder"] / "clue.json").get("parts")
    if not parts:
        sys.exit("The clue has no saved parts of the key idea, so the judge can't mark against them. Make a new "
                 f"clue: python src/generate/make_problem.py --auto --again {run['code']}")
    return parts


def confirm(args):
    if args.yes:
        return True
    try:
        go = input("Start? [Y/n]: ").strip().upper()
    except EOFError:
        go = "N"
    if go not in ("", "Y"):
        print("Nothing started.")
        return False
    return True


def groq_note(marks):
    tokens = marks * TOKENS_PER_MARK
    keys = len(make_client(load_model_config("judge")))
    allowance = f"{200_000 * keys:,} a day with {keys} key{'s' if keys > 1 else ''}"   # free plan: 200,000 per key
    words = (f"fits in the free plan's {allowance} if little was used today" if tokens <= 200_000 * keys else
             f"more than the free plan's {allowance}, so judging will stop and carry on later")
    return f"about {tokens:,} Groq tokens; {words}"


# ---------- 1. answers ----------

def missing_answers(run_folder, run, model):
    return [(kind, t) for kind in KINDS for t in range(1, run["tries"] + 1)
            if not answer_path(run_folder, run, file_name(model, kind, t)).exists()]


def ask_models(run_folder, run, models, host, version, digests):
    tries = run["tries"]
    if not any(missing_answers(run_folder, run, m) for m in models):
        return
    print(f"\nAsking {len(models)} models: the problem alone and with the clue, {tries} tries each, thinking off.")
    for i, model in enumerate(models, 1):
        problem = run["problems"][model["group"]]["problem"]
        questions = {"alone": ALONE_QUESTION.format(problem=problem),
                     "clue": CLUE_QUESTION.format(problem=problem, clue=run["clue"])}
        todo = missing_answers(run_folder, run, model)
        label = f"[{i}/{len(models)}] {model['name']} ({model['group']})"
        if not todo:
            print(f"{label}: already done")
            continue
        print(f"{label}: {len(todo)} answers ", end="", flush=True)
        start = time.time()
        try:
            for kind, t in todo:
                seed = run["seed"] + t
                body, reply, thinking, answer = ask(host, run["test_settings"], model,
                                                    [{"role": "user", "content": questions[kind]}], seed)
                save_json(run_folder / "answers" / file_name(model, kind, t), {
                    "model": model, "ollama_version": version, "model_digest": digests.get(model["id"]),
                    "group": model["group"], "kind": kind, "try": t, "seed": seed, "question": questions[kind],
                    "answer": answer, "thinking": thinking, "stopped_because": reply.get("done_reason"),
                    "tokens_in": reply.get("prompt_eval_count"), "tokens_out": reply.get("eval_count"),
                    "timestamp": now(), "request": body, "reply": reply})
                print(".", end="", flush=True)
            print(f" done ({time.time() - start:.0f} s)")
        except Exception as e:   # out of memory, Ollama stopped, ...: the next run picks up the rest
            print(f"\n  FAILED: {e}\n  The answers so far are saved; run the same command again to finish.")
        finally:
            unload(host, model["id"])


# ---------- 2. marks ----------

def clean_mark(reply, n_parts):
    """The judge's mark, or None if unreadable: one true/false per part, and a verdict that agrees with them
    (it can't say the idea is used while a part is missing)."""
    if not isinstance(reply, dict) or not isinstance(reply.get("uses_idea"), bool):
        return None
    present = reply.get("parts_present")
    if not (isinstance(present, list) and len(present) == n_parts and all(isinstance(x, bool) for x in present)):
        return None
    if reply["uses_idea"] and not all(present):
        return None
    return {"uses_idea": reply["uses_idea"], "parts_present": present,
            "reason": str(reply.get("reason") or "").strip()}


def judge_answers(run_folder, run):
    """Mark every answer that has no mark yet. Raises DailyLimit (the marks so far are saved)."""
    todo = sorted(p for name, p in answer_files(run_folder, run).items()
                  if not (run_folder / "marks" / name).exists())
    if not todo:
        return
    parts = run_parts(run)
    cfg = load_model_config("judge")
    client = make_client(cfg)
    print(f"\nJudge: marking {len(todo)} answers ", end="", flush=True)
    for path in todo:
        rec = load_json(path)
        problem = run["problems"][rec["group"]]
        # The judge never sees the question, so it can't tell whether the clue was given.
        user = (f"PROBLEM: {problem['problem']}\nKEY IDEA: {run['principle']}\nPARTS:\n"
                + "\n".join(f"{i}. {p}" for i, p in enumerate(parts, 1))
                + f"\nEXPECTED ANSWER: {problem['answer']}"
                f"\n\nANSWER:\n{rec['answer'] or '(empty: the model gave no final answer)'}")
        mark = None
        for attempt in range(JUDGE_ATTEMPTS):
            log, raw = logged_call(client, cfg, JUDGE_PROMPT, user, seed=cfg["seed"] + attempt)
            mark = clean_mark(parse_json_reply(raw), len(parts))
            if mark:
                break
        save_json(run_folder / "marks" / path.name, {"answer_file": show_path(path), "mark": mark,
                                                     "judge_call": log, "timestamp": now()})
        print(".", end="", flush=True)
    print(" done")


# ---------- 3. summary ----------

def count(run_folder, run, model):
    """Per kind: answered, judged (readable marks), solved, unreadable marks, cut off (no final answer)."""
    out = {}
    for kind in KINDS:
        c = dict.fromkeys(("answered", "judged", "solved", "unreadable", "cut_off"), 0)
        for t in range(1, run["tries"] + 1):
            path = answer_path(run_folder, run, file_name(model, kind, t))
            if not path.exists():
                continue
            c["answered"] += 1
            rec = load_json(path)
            c["cut_off"] += rec["stopped_because"] == "length" or not rec["answer"]
            mark_path = run_folder / "marks" / file_name(model, kind, t)
            if mark_path.exists():
                mark = load_json(mark_path)["mark"]
                if mark is None:
                    c["unreadable"] += 1
                else:
                    c["judged"] += 1
                    c["solved"] += mark["uses_idea"]
        out[kind] = c
    return out


def group_verdict(group, pooled, rule, finished):
    """(passed or None if unfinished, words)."""
    if not finished:
        return None, "not finished"
    alone, clue = pooled["alone"], pooled["clue"]
    alone_rate = alone["solved"] / alone["judged"] if alone["judged"] else 1.0
    clue_rate = clue["solved"] / clue["judged"] if clue["judged"] else 0.0
    faults = []
    if alone_rate > rule["alone_max"]:
        faults.append(f"too easy alone ({alone_rate:.0%}, limit {rule['alone_max']:.0%})")
    if group in rule["clue_groups"] and clue_rate < rule["clue_min"]:
        faults.append(f"the clue doesn't help enough ({clue_rate:.0%}, need {rule['clue_min']:.0%})")
    if faults:
        return False, "FAIL: " + "; ".join(faults)
    if group not in rule["clue_groups"]:
        return True, f"hard alone; with the clue {clue_rate:.0%} (only reported for this group)"
    return True, "PASS"


def summarise(run_folder, run, all_models, not_tested, full_tries):
    tries, rule = run["tries"], run["rule"]
    models = [m for m in all_models if m["id"] in run["models"]]
    lines = [f"DIFFICULTY FILTER - {run['title']} ({dt.datetime.now():%Y-%m-%d %H:%M})",
             f"Folder: {show_path(run_folder)}",
             f"{tries} tries alone and {tries} with the clue per model, thinking off. Judge: "
             f"{load_model_config('judge')['model']}. Pass per group: alone at most {rule['alone_max']:.0%}, with "
             f"the clue at least {rule['clue_min']:.0%} (clue required for: {', '.join(rule['clue_groups'])})."]
    if run.get("answers_from"):
        lines.append(f"Re-marked with the current judge prompt. Answers from: {run['answers_from']}")
    lines.append("")
    per_model, groups_out, finished_all = {}, {}, True
    for group in run["groups"]:
        members = [m for m in models if m["group"] == group]
        lines.append(f"{group.upper()} group (problem: {run['problems'][group]['file']})")
        pooled = {k: dict.fromkeys(("judged", "solved"), 0) for k in KINDS}
        finished = bool(members)
        for model in members:
            c = count(run_folder, run, model)
            per_model[model["id"]] = {"group": group, "alone": f"{c['alone']['solved']}/{c['alone']['judged']}",
                                      "with_clue": f"{c['clue']['solved']}/{c['clue']['judged']}", "counts": c}
            for kind in KINDS:
                pooled[kind]["judged"] += c[kind]["judged"]
                pooled[kind]["solved"] += c[kind]["solved"]
                finished &= c[kind]["judged"] + c[kind]["unreadable"] == tries
            notes = [f"{c[k]['cut_off']} {'alone' if k == 'alone' else 'with clue'} cut off" for k in KINDS
                     if c[k]["cut_off"]]
            notes += [f"{c[k]['unreadable']} unreadable judge mark{'s' if c[k]['unreadable'] > 1 else ''} "
                      f"({'alone' if k == 'alone' else 'with clue'})" for k in KINDS if c[k]["unreadable"]]
            lines.append(f"  {model['name']:<20}alone {pct(c['alone']['solved'], c['alone']['judged']):<14}"
                         f"with clue {pct(c['clue']['solved'], c['clue']['judged']):<14}{'; '.join(notes)}")
        skipped = [name for name, g in not_tested if g == group]
        if skipped:
            lines.append(f"  Not tested: {', '.join(skipped)}")
        passed, words = group_verdict(group, pooled, rule, finished) if members else (None, "no models tested")
        finished_all &= finished
        lines.append(f"  Group: alone {pct(pooled['alone']['solved'], pooled['alone']['judged'])}, with clue "
                     f"{pct(pooled['clue']['solved'], pooled['clue']['judged'])}  ->  {words}")
        lines.append("")
        groups_out[group] = {"passed": passed, "verdict": words, "pooled": pooled, "not_tested": skipped}

    command = (f"python src/generate/difficulty_filter.py {run['code']}"
               + (f" --groups {' '.join(run['groups'])}" if run["groups"] != list(GROUPS) else ""))
    if not finished_all:
        result = f"NOT FINISHED. Carry on with the same command: {command}" + (
            " --rejudge" if run.get("answers_from") else "")
    else:
        required = [g for g in groups_out if g in rule["clue_groups"]]
        passed = [g for g in required if groups_out[g]["passed"]]
        failed = [f"{g}: {groups_out[g]['verdict'][len('FAIL: '):]}" for g, v in groups_out.items()
                  if v["passed"] is False]
        reported = [f"{g}: {v['verdict']}" for g, v in groups_out.items() if g not in required and v["passed"]]
        result = " ".join(([f"Passes for: {', '.join(passed) or 'no group'}."] if required else [])
                          + ([f"Fails for {'; '.join(failed)}."] if failed else [])
                          + ([f"Reported only, {'; '.join(reported)}."] if reported else []))
    lines.append(f"RESULT: {result}")
    if finished_all and tries < full_tries:
        lines.append(f"Only {tries} tries per model (the config says {full_tries}). If a group is close to a limit, "
                     f"add the rest to this run: {command} --tries {full_tries}")

    text = "\n".join(lines)
    print("\n" + text)
    (run_folder / "summary.txt").write_text(text + "\n", encoding="utf-8")   # worked out from the saved marks
    with open(run_folder / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"complete": finished_all, "result": result, "groups": groups_out, "models": per_model},
                  f, ensure_ascii=False, indent=2)


# ---------- main ----------

def stop_no_ollama(host):
    print(f"STOP: our Ollama is not running on {host}. In another PowerShell window run this and leave it open:")
    print("  " + START_CMD)
    sys.exit(1)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Is the problem hard alone, but solvable with its clue?")
    parser.add_argument("code", help="statement code, e.g. CHEM-1 (its newest kept problem and clue are tested)")
    parser.add_argument("--groups", nargs="+", choices=GROUPS, help="only these strength groups (default: all)")
    parser.add_argument("--new", action="store_true", help="start a fresh run even if the last one is unfinished")
    parser.add_argument("--yes", action="store_true", help="don't ask before starting a new run")
    parser.add_argument("--tries", type=int, help="tries per model and question (default: the config). Higher than "
                        "the last run's: adds the missing tries to that run")
    parser.add_argument("--rejudge", action="store_true", help="mark the last run's saved answers again with the "
                        "current judge prompt, in a new folder (the old marks stay)")
    args = parser.parse_args()
    if args.tries is not None and args.tries < 1:
        parser.error("--tries must be at least 1")
    if args.rejudge and (args.tries or args.new):
        parser.error("--rejudge marks the last run's answers as they are; it can't be combined with --tries or --new")

    host, settings, all_models = load_test_config()
    info = server_info(host)
    if info is None and not args.rejudge:
        stop_no_ollama(host)
    version, digests = info or (None, {})
    with open(CONFIG_PATH, encoding="utf-8") as f:
        filter_cfg = yaml.safe_load(f)["difficulty_filter"]

    code = args.code.upper()
    folder = problem_folder(code)
    groups = [g for g in GROUPS if g in (args.groups or GROUPS)]
    memory = last_memory_check(code)
    if memory is None:
        print(f"  Note: no memory check found for {code}. Run it first (context.md section 3.2, step 4).")
    elif memory.startswith("DROP"):
        print(f"  WARNING: the memory check says {memory}")

    chosen = [m for m in all_models if m["group"] in groups]
    local = [m for m in chosen if m["route"] == "local" and m["id"] in digests]

    last = None if args.new else last_run(folder, groups)
    run_folder = None
    if args.rejudge:
        if last is None:
            sys.exit(f"No earlier filter run of {code} for these groups ({', '.join(groups)}) to mark again.")
        run = load_run(last)
        if run.get("answers_from") and not is_complete(last):
            run_folder = last
            print(f"Carrying on with the re-marking in {show_path(last)}")
        else:
            gaps = sum(len(missing_answers(last, run, {"id": m})) for m in run["models"])
            if gaps:
                sys.exit(f"{show_path(last)} is missing {gaps} answers. Finish it first: "
                         f"python src/generate/difficulty_filter.py {code}")
            if run["clue"] != load_json(ROOT / run["problem_folder"] / "clue.json")["clue"]:
                sys.exit(f"The clue in {run['problem_folder']} is not the one {show_path(last)} tested.")
            marks = len(answer_files(last, run))
            run = {**{k: v for k, v in run.items() if k != "created"}, "created": now(),
                   "rejudge_of": show_path(last),
                   "answers_from": show_path(answer_path(last, run, "x").parent),
                   "parts": load_json(ROOT / run["problem_folder"] / "clue.json").get("parts")}
            run_parts(run)
            print(f"\n{run['title']}\nMarking the {marks} saved answers of {show_path(last)} again with the "
                  f"current judge prompt, into a new folder ({groq_note(marks)}). The old marks stay.")
            if not confirm(args):
                return
            run_folder = folder / f"filter_{dt.datetime.now():%Y%m%d_%H%M%S}"
            save_json(run_folder / "run.json", run)
            print(f"Saving to {show_path(run_folder)}")
    elif last:
        run = load_run(last)
        if args.tries and args.tries > run["tries"]:
            added = sum(m in digests for m in run["models"]) * len(KINDS) * (args.tries - run["tries"])
            print(f"\n{run['title']}\nAdding tries {run['tries'] + 1}-{args.tries} to {show_path(last)} "
                  f"(tries 1-{run['tries']} are kept): {added} more answers and marks ({groq_note(added)}).")
            if not confirm(args):
                return
            save_json(last / f"more_tries_{dt.datetime.now():%Y%m%d_%H%M%S}.json",
                      {"created": now(), "from": run["tries"], "tries": args.tries})
            run["tries"] = args.tries
            run_folder = last
        elif not is_complete(last):
            run_folder = last
            print(f"Carrying on with {show_path(last)} ({run['tries']} tries per model)")
    if not run_folder:
        statement = load_json(folder / "statement.json")
        problems = {g: problem_for(folder, g) for g in groups}
        run = {"created": now(), "code": code, "title": statement.get("title") or code,
               "problem_folder": show_path(folder), "groups": groups, "tries": args.tries or filter_cfg["tries"],
               "seed": filter_cfg["seed"],
               "rule": {k: filter_cfg[k] for k in ("alone_max", "clue_min", "clue_groups")},
               "models": [m["id"] for m in local], "test_settings": settings,
               "thinking": "off where it can be switched off (OLMo Think always thinks)",
               "principle": statement["principle"], "clue": load_json(folder / "clue.json")["clue"],
               "parts": load_json(folder / "clue.json").get("parts"),
               "problems": {g: {"file": name, "problem": p["problem"], "answer": p["answer"]}
                            for g, (name, p) in problems.items()},
               "questions": {"alone": ALONE_QUESTION, "clue": CLUE_QUESTION}}
        answers = len(local) * len(KINDS) * run["tries"]
        seconds = sum(len(KINDS) * run["tries"] * SECONDS_PER_ANSWER.get(m["thinking"], SECONDS_PER_ANSWER_OTHER)
                      for m in local)
        print(f"\n{run['title']}\nProblem tested: {show_path(folder)}\n"
              f"This run: {run['tries']} tries per model, alone and with the clue. {answers} answers on the lab PC "
              f"(about {seconds // 60} minutes) and {answers} judge marks ({groq_note(answers)}).")
        run_parts(run)
        if not confirm(args):
            return
        run_folder = folder / f"filter_{dt.datetime.now():%Y%m%d_%H%M%S}"
        save_json(run_folder / "run.json", run)
        print(f"Saving to {show_path(run_folder)}")

    models = [m for m in all_models if m["id"] in run["models"]]
    if info is None and any(missing_answers(run_folder, run, m) for m in models):
        stop_no_ollama(host)
    ask_models(run_folder, run, [m for m in models if m["id"] in digests], host, version, digests)
    not_tested = [(f"{m['name']} ({'API not set up yet' if m['route'] != 'local' else 'not in our Ollama'})",
                   m["group"]) for m in chosen if m["id"] not in run["models"]]
    try:
        judge_answers(run_folder, run)
    except DailyLimit as e:
        print(f"\n\nSTOPPED: {e} The marks so far are saved; run the same command later to finish.")
    except Exception as e:   # the API kept failing
        print(f"\n\nSTOPPED: the judge call failed ({str(e)[:300]}). Run the same command again to finish.")
    summarise(run_folder, run, all_models, not_tested, filter_cfg["tries"])


if __name__ == "__main__":
    main()
