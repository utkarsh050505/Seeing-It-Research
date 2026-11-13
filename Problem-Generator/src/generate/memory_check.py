r"""
Memory check (context.md section 3.2, step 4): do any of our test models already know a research finding?

Run it on the lab PC from the repo root, with our own Ollama running in another window:
    $env:OLLAMA_HOST="127.0.0.1:11500"; $env:OLLAMA_MODELS="E:\LLMxPsychology-models"; ollama serve
Then:
    python src/generate/memory_check.py CHEM-1             # a statement from data/sources/statements.csv
    python src/generate/memory_check.py                    # paste a statement instead
    python src/generate/memory_check.py --resume FOLDER    # carry on with a run that stopped halfway
    python src/generate/memory_check.py CHEM-1 --models qwen3.5:4b-bf16 llama3.2:3b-instruct-fp16   # only these
    python src/generate/memory_check.py CHEM-1 --tries 1   # quick try (default: memory_check.tries in config)

A pasted statement looks the same as for make_problem.py (title line, "Statement: ...", "Principle: ...").

How it works:
1. The writer (gpt-oss) turns the statement into three questions. You read them first, to make sure none
   gives the answer away:
     real paper    - names the journal and month and describes the problem, not the fix: "Do you know this study?"
     made-up paper - the same for a study that does not exist, to catch models that claim to know anything
     problem alone - the same problem with no paper named: "How would you solve it?"
2. Every local test model answers each question several times (a fresh chat each time), thinking off.
3. The checker (gpt-oss) marks every answer: does it claim to know the study, does it use the real fix, does it
   give details that only the paper has (not in the question)?
4. Each model gets a verdict (rules in `verdict`). If any model knows the finding, it is dropped for all models.

Everything goes to data/candidates/<code>/memory_<date_time>/: the statement, the questions, every answer, every
mark and the summary. Nothing saved is redone or overwritten, so a run that stops (AnyDesk, restart, API limit)
carries on with --resume.
"""

import argparse
import datetime as dt
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import yaml

from llm import CONFIG_PATH, ROOT, chat, load_model_config, make_client, parse_json_reply
from local_llm import START_CMD, ask, load_test_config, server_info, unload
from statements import get_item

WRITER_PROMPT = ROOT / "src" / "generate" / "prompts" / "memory_questions_system.txt"
JUDGE_PROMPT = ROOT / "src" / "generate" / "prompts" / "memory_judge_system.txt"
OUT_DIR = ROOT / "data" / "candidates"

# Fixed wording around the writer's text, so every statement is asked the same way.
PAPER_QUESTION = (
    "A study reported by {journal} in {month_year} was about this problem:\n\n{problem}\n\n"
    "Do you know this study? If you do, explain how the researchers solved the problem, with any specific details "
    "you remember (methods, materials, numbers, names). If you don't know it, say so plainly instead of guessing.")
ALONE_QUESTION = ("Here is a problem that researchers faced:\n\n{problem}\n\n"
                  "How would you solve it? Give your best solution in a short paragraph.")
KINDS = ("real", "fake", "alone")
KIND_LABEL = {"real": "Real paper", "fake": "Made-up paper", "alone": "Problem alone"}
NO_STUDY = "None. This study was made up for the test; it does not exist."
JUDGE_ATTEMPTS = 4


# ---------- small helpers ----------

def ask_choice(prompt, allowed):
    """Ask until the reply is one of `allowed`, so leftover pasted lines can't answer for you."""
    while True:
        try:
            choice = input(prompt).strip().upper()
        except EOFError:
            return "Q"
        if choice in allowed:
            return choice
        print(f"  Please type {', '.join(allowed[:-1])} or {allowed[-1]}.")


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:   # "x" = fail rather than overwrite
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def show(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def file_name(model, kind, t):
    return f"{model['id'].replace(':', '_').replace('/', '_')}__{kind}__{t}.json"


def gpt_settings(cfg, prompt_text, seed, response):
    usage = getattr(response, "usage", None)
    return {"provider": cfg["provider"], "model": cfg["model"], "model_version": getattr(response, "model", None),
            "temperature": cfg["temperature"], "max_completion_tokens": cfg["max_completion_tokens"],
            "reasoning_effort": cfg.get("reasoning_effort"), "seed": seed,
            "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
            "tokens_in": getattr(usage, "prompt_tokens", None), "tokens_out": getattr(usage, "completion_tokens", None)}


# ---------- 1. questions ----------

def build_questions(parts):
    paper = dict(journal=parts["journal"], month_year=parts["month_year"])
    return {"real": PAPER_QUESTION.format(problem=parts["problem"], **paper),
            "fake": PAPER_QUESTION.format(problem=parts["fake_problem"], **paper),
            "alone": ALONE_QUESTION.format(problem=parts["problem"])}


def write_questions(item):
    """The writer drafts the questions until you accept a version. Returns (questions, record), or None if you quit."""
    cfg = load_model_config("memory_questions")
    client = make_client(cfg)
    system = WRITER_PROMPT.read_text(encoding="utf-8")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"STATEMENT: {item['statement']}\nPRINCIPLE: {item['principle']}"}]
    attempts = []
    while True:
        seed = random.SystemRandom().randrange(2**31)
        print("\n  Writing the questions...")
        response, raw = None, None
        try:
            response = chat(client, cfg, messages, seed)
            raw = response.choices[0].message.content
        except Exception as e:   # network / key / rate limit: report and let the user try again
            print(f"  The API call failed: {e}")
        parts = parse_json_reply(raw)
        ok = isinstance(parts, dict) and all(parts.get(k) for k in ("journal", "month_year", "problem", "fake_problem"))
        attempts.append({"settings": gpt_settings(cfg, system, seed, response), "raw_reply": raw,
                         "parsed": parts if ok else None})
        if not ok:
            if raw:
                print("  The reply wasn't in the expected format.")
            if ask_choice("[N] try again   [Q] quit: ", ["N", "Q"]) == "Q":
                return None
            continue

        questions = build_questions(parts)
        for kind in KINDS:
            print(f"\n[{KIND_LABEL[kind]}]\n{questions[kind]}")
        print("\nCheck that no question gives away how the researchers solved it.")
        choice = ask_choice("[Y] ask the models   [N] write them again   [Q] quit: ", ["Y", "N", "Q"])
        if choice == "Q":
            return None
        if choice == "Y":
            return questions, {"parts": parts, "questions": questions, "accepted_attempt": len(attempts),
                               "attempts": attempts}


# ---------- 2. answers ----------

def ask_models(folder, run, questions, models, host, version, digests):
    print(f"\nAsking {len(models)} models: 3 questions, {run['tries']} {'try' if run['tries'] == 1 else 'tries'} "
          f"each, thinking off.")
    for i, model in enumerate(models, 1):
        todo = [(kind, t) for kind in KINDS for t in range(1, run["tries"] + 1)
                if not (folder / "answers" / file_name(model, kind, t)).exists()]
        label = f"[{i}/{len(models)}] {model['name']} ({model['group']})"
        if not todo:
            print(f"{label}: already done")
            continue
        print(f"{label}: {len(todo)} answers ", end="", flush=True)
        start = time.time()
        try:
            for kind, t in todo:
                seed = run["seed"] + t
                messages = [{"role": "user", "content": questions[kind]}]
                body, reply, thinking, answer = ask(host, run["test_settings"], model, messages, seed)
                save_json(folder / "answers" / file_name(model, kind, t), {
                    "model": model, "ollama_version": version, "model_digest": digests.get(model["id"]),
                    "kind": kind, "try": t, "seed": seed, "question": questions[kind],
                    "answer": answer, "thinking": thinking, "stopped_because": reply.get("done_reason"),
                    "tokens_in": reply.get("prompt_eval_count"), "tokens_out": reply.get("eval_count"),
                    "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                    "request": body, "reply": reply})
                print(".", end="", flush=True)
            print(f" done ({time.time() - start:.0f} s)")
        except Exception as e:   # out of memory, Ollama stopped, ...: --resume picks up the rest
            print(f"\n  FAILED: {e}\n  The answers so far are saved; rerun with --resume to finish.")
        finally:
            unload(host, model["id"])


# ---------- 3. marks ----------

def valid_mark(mark):
    return (isinstance(mark, dict) and isinstance(mark.get("claims_to_know"), bool)
            and isinstance(mark.get("uses_real_fix"), bool) and isinstance(mark.get("matching_details"), list))


def check_answers(folder, item):
    todo = sorted(p for p in (folder / "answers").glob("*.json") if not (folder / "checks" / p.name).exists())
    if not todo:
        return
    cfg = load_model_config("judge")
    client = make_client(cfg)
    system = JUDGE_PROMPT.read_text(encoding="utf-8")
    print(f"\nChecker: marking {len(todo)} answers ", end="", flush=True)
    for path in todo:
        rec = load_json(path)
        study, principle = (NO_STUDY, "None.") if rec["kind"] == "fake" else (item["statement"], item["principle"])
        user = (f"REAL STUDY: {study}\nPRINCIPLE: {principle}\n\nQUESTION:\n{rec['question']}\n\n"
                f"ANSWER:\n{rec['answer'] or '(empty: the model gave no final answer)'}")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt in range(JUDGE_ATTEMPTS):
            seed = cfg["seed"] + attempt
            response, raw, error = None, None, None
            try:
                response = chat(client, cfg, messages, seed)
                raw = response.choices[0].message.content
            except Exception as e:   # rate limit, network, ...
                error = e
            mark = parse_json_reply(raw)
            if valid_mark(mark):
                break
            if attempt + 1 < JUDGE_ATTEMPTS:
                wait = 20 * (attempt + 1)
                print(f"\n  Checker call failed ({error or 'reply not in the expected format'}); "
                      f"trying again in {wait} s ", end="", flush=True)
                time.sleep(wait)
        else:
            print("\n  The checker keeps failing (the daily API limit?). Stopping here; rerun with --resume later.")
            return
        save_json(folder / "checks" / path.name, {
            "answer_file": f"answers/{path.name}", "mark": mark, "raw_reply": raw,
            "settings": gpt_settings(cfg, system, seed, response),
            "timestamp": dt.datetime.now().isoformat(timespec="seconds")})
        print(".", end="", flush=True)
    print(" done")


# ---------- 4. verdicts ----------

def count_marks(folder, model, tries):
    """How many of the tries got each mark, or None if some answers are not marked yet."""
    c = dict.fromkeys(("claims_real", "fix_real", "details_real", "claims_fake", "fix_alone", "details_alone"), 0)
    for kind in KINDS:
        for t in range(1, tries + 1):
            path = folder / "checks" / file_name(model, kind, t)
            if not path.exists():
                return None
            mark = load_json(path)["mark"]
            if kind == "real":
                c["claims_real"] += mark["claims_to_know"]
                c["fix_real"] += mark["uses_real_fix"]
                c["details_real"] += bool(mark["matching_details"])
            elif kind == "fake":
                c["claims_fake"] += mark["claims_to_know"]
            else:
                c["fix_alone"] += mark["uses_real_fix"]
                c["details_alone"] += bool(mark["matching_details"])
    return c


def verdict(c, tries):
    """(symbol, words). Memory shows up as details only the paper has, given more often when the paper is named
    than when it isn't: details a model also gives with no paper named come from reasoning, not memory."""
    most = tries // 2 + 1
    extra_details = c["details_real"] > c["details_alone"]
    if extra_details and c["details_real"] >= most:
        return "✗", "KNOWS IT: gives details only the paper has"
    if extra_details:
        return "?", f"check by hand: gave details only the paper has in {c['details_real']} of {tries}"
    if c["fix_real"] >= most and c["fix_alone"] == 0:
        return "?", "check by hand: gives the fix only when the paper is named"
    if c["claims_real"] >= most and c["fix_real"] == 0:
        return "✓", "makes it up: claims to know it, describes something else"
    if c["fix_real"] or c["fix_alone"]:
        return "✓", "worked it out, not memory (the problem may be easy)"
    return "✓", "doesn't know"


def summarise(folder, item, run, all_models):
    tries = run["tries"]
    lines = [f"MEMORY CHECK - {item['title'] or item['code']} ({dt.datetime.now():%Y-%m-%d %H:%M})",
             f"Folder: {show(folder)}",
             f"3 questions, {tries} {'try' if tries == 1 else 'tries'} each per model, thinking off. "
             f"Checker: {load_model_config('judge')['model']}.",
             "",
             f"  {'Model':<20}{'Group':<8}{'Says it knows':<15}{'Real fix':<10}{'Paper details':<15}"
             f"{'Made-up paper':<15}{'Solved alone':<14}Verdict"]
    rows, unfinished, api_skipped = [], [], []
    for model in all_models:
        row = {"model": model["id"], "name": model["name"], "group": model["group"]}
        start = f"  {model['name']:<20}{model['group']:<8}"
        c = count_marks(folder, model, tries) if model["id"] in run["models"] else None
        if model["id"] not in run["models"]:
            row["verdict"] = "skipped: API not set up yet" if model["route"] == "api" else "not in this run"
            if model["route"] == "api":
                api_skipped.append(model["name"])
            lines.append(f"{start}{'-':<69}{row['verdict']}")
        elif c is None:
            row["verdict"] = "not finished"
            unfinished.append(model["name"])
            lines.append(f"{start}{'-':<69}not finished")
        else:
            row["symbol"], row["verdict"] = verdict(c, tries)
            row["counts"] = c
            n = {k: f"{v}/{tries}" for k, v in c.items()}
            lines.append(f"{start}{n['claims_real']:<15}{n['fix_real']:<10}{n['details_real']:<15}"
                         f"{n['claims_fake']:<15}{n['fix_alone']:<14}{row['symbol']} {row['verdict']}")
        rows.append(row)

    knows = [r["name"] for r in rows if r.get("symbol") == "✗"]
    unsure = [r["name"] for r in rows if r.get("symbol") == "?"]
    if knows:
        result = (f"DROP: {', '.join(knows)} already {'knows' if len(knows) == 1 else 'know'} this finding. "
                  f"Drop it for every model.")
    elif unfinished:
        result = (f"NOT FINISHED: {', '.join(unfinished)}. Carry on with:\n"
                  f"  python src/generate/memory_check.py --resume {show(folder)}")
    elif unsure:
        result = f"CHECK BY HAND: read the answers of {', '.join(unsure)} in {show(folder / 'answers')}"
    else:
        result = "PASS: no model knows this finding."
    lines += ["",
              "  Says it knows  = asked about the real paper, it claims to know the study",
              "  Real fix       = asked about the real paper, its answer uses the real fix",
              "  Paper details  = asked about the real paper, it gives details only the paper has (not in the question)",
              "  Made-up paper  = asked about a study that doesn't exist, it claims to know it",
              "  Solved alone   = given the problem with no paper named, it reaches the real fix",
              "", f"RESULT: {result}"]
    if api_skipped and not knows:
        lines.append(f"Not checked yet: {', '.join(api_skipped)} (API not set up).")

    text = "\n".join(lines)
    print("\n" + text)
    (folder / "summary.txt").write_text(text + "\n", encoding="utf-8")   # worked out from the saved marks
    with open(folder / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"result": result, "tries": tries, "models": rows}, f, ensure_ascii=False, indent=2)


# ---------- main ----------

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Check whether our test models already know a research finding.")
    parser.add_argument("code", nargs="?", help="statement code in data/sources/statements.csv, e.g. CHEM-1 "
                                                "(leave out to paste a statement)")
    parser.add_argument("--resume", metavar="FOLDER", help="carry on with a run that stopped halfway")
    parser.add_argument("--models", nargs="+", metavar="ID", help="new run: only these test models (config ids)")
    parser.add_argument("--tries", type=int, help="new run: tries per question (default: memory_check.tries)")
    args = parser.parse_args()

    host, settings, all_models = load_test_config()
    info = server_info(host)
    if info is None:
        print(f"STOP: our Ollama is not running on {host}. In another PowerShell window run this and leave it open:")
        print("  " + START_CMD)
        sys.exit(1)
    version, digests = info

    if args.resume:
        folder = Path(args.resume).resolve()
        item = load_json(folder / "statement.json")
        run = load_json(folder / "run.json")
        questions = load_json(folder / "questions.json")["questions"]
        print(f"Carrying on with {show(folder)}")
    else:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            memory_cfg = yaml.safe_load(f)["memory_check"]
        known = {m["id"] for m in all_models}
        wanted = args.models or [m["id"] for m in all_models if m["route"] == "local"]
        if set(wanted) - known:
            sys.exit(f"Not in config/models.yaml: {', '.join(sorted(set(wanted) - known))}")

        item = get_item(args.code)
        if item is None:
            return

        written = write_questions(item)
        if written is None:
            print("Nothing asked, nothing saved.")
            return
        questions, record = written
        now = dt.datetime.now()
        folder = OUT_DIR / item["code"] / f"memory_{now:%Y%m%d_%H%M%S}"
        run = {"created": now.isoformat(timespec="seconds"), "tries": args.tries or memory_cfg["tries"],
               "seed": memory_cfg["seed"], "models": wanted, "test_settings": settings}
        save_json(folder / "statement.json", item)
        save_json(folder / "run.json", run)
        save_json(folder / "questions.json", record)
        print(f"\nSaving to {show(folder)}")
        print(f"If this stops, carry on with:  python src/generate/memory_check.py --resume {show(folder)}")

    models = [m for m in all_models if m["id"] in run["models"] and m["route"] == "local"]
    missing = [m["name"] for m in models if m["id"] not in digests]
    if missing:
        print(f"Not in our Ollama, skipped: {', '.join(missing)}")
    ask_models(folder, run, questions, [m for m in models if m["id"] in digests], host, version, digests)
    check_answers(folder, item)
    summarise(folder, item, run, all_models)


if __name__ == "__main__":
    main()
