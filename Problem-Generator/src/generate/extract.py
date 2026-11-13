"""
Turn a pasted research finding into a candidate benchmark item:
a plain-language problem (solution hidden), the gold idea, and clue/decoy story sketches.

Usage (from the repo root):
    python src/generate/extract.py                  # paste findings one after another
    python src/generate/extract.py --file f.txt     # read one finding from a text file
    python src/generate/extract.py --dry-run        # show what would be sent; no API call, nothing saved

Settings come from config/models.yaml (section `writer`). The API key is read from .env or the
environment variable named there (e.g. GROQ_API_KEY).

Every result is saved as a NEW file in data/candidates/ (existing files are never overwritten).
Candidates are not data: they still go through the filters, recall probe and human check (context.md §3.2).
"""

import argparse
import datetime as dt
import hashlib
import json
import sys
import textwrap
import uuid
from pathlib import Path

from llm import ROOT, chat, load_model_config, make_client, parse_json_reply

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "extract_system.txt"
DEFAULT_OUT_DIR = ROOT / "data" / "candidates"

CUTOFF = dt.date(2026, 7, 1)  # findings must be published on or after this date (context.md §3.1)
END_MARKER = "END"
QUIT_MARKER = "QUIT"

REQUIRED_KEYS = [
    "problem_faced", "how_solved", "gold_idea", "principle", "principle_why", "single_idea",
    "single_idea_note", "domain", "problem", "blocked_answers", "clue_ideas", "lure_idea", "warnings",
]


# ---------- input ----------

def read_pasted_text():
    """Read lines until END. Returns None if the user types QUIT."""
    print(f"\nPaste the research finding. Type {END_MARKER} on its own line when done ({QUIT_MARKER} to stop).")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().upper() == QUIT_MARKER and not lines:
            return None
        if line.strip().upper() == END_MARKER:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def parse_date(text):
    try:
        return dt.date.fromisoformat(text.strip())
    except ValueError:
        return None


def ask_source_details():
    title = input("Source title (optional, Enter to skip): ").strip() or None
    url = input("Link / DOI (optional, Enter to skip): ").strip() or None
    while True:
        raw = input("Publication date YYYY-MM-DD (optional, Enter to skip): ").strip()
        if not raw:
            return title, url, None
        if parse_date(raw):
            return title, url, raw
        print("  Could not read that date. Use the form 2026-08-14.")


def check_cutoff(published):
    """True / False against the cutoff, or None when no date was given."""
    if published is None:
        print("  Note: no publication date given. It must be checked before this item can be used.")
        return None
    if parse_date(published) < CUTOFF:
        print(f"  WARNING: published {published}, before the {CUTOFF} cutoff. Models may have seen this finding.")
        return False
    return True


def confirm(question):
    return input(f"{question} [y/N]: ").strip().lower() in ("y", "yes")


# ---------- model call ----------

def build_messages(system_prompt, text, title, url, published):
    header = [f"Source title: {title or 'not given'}",
              f"Link: {url or 'not given'}",
              f"Published: {published or 'not given'}"]
    user = "\n".join(header) + "\n\nFinding:\n" + text
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]


# ---------- saving and showing ----------

def save_record(record, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{record['candidate_id']}.json"
    with open(path, "x", encoding="utf-8") as f:  # "x" = fail rather than overwrite
        json.dump(record, f, ensure_ascii=False, indent=2)
    return path


def wrap(text, indent="  "):
    return textwrap.fill(str(text), width=100, initial_indent=indent, subsequent_indent=indent)


def show(result):
    print("\n" + "=" * 100)
    print(f"Domain:      {result.get('domain')}")
    print(f"Principle:   {result.get('principle')}")
    single = "yes" if result.get("single_idea") else f"NO - {result.get('single_idea_note')}"
    print(f"Single idea: {single}")

    print("\nPROBLEM (what the test model will see)")
    print(wrap(result.get("problem")))
    print("\nGOLD IDEA (hidden from the test model)")
    print(wrap(result.get("gold_idea")))
    print("\nHow the researchers solved it")
    print(wrap(result.get("how_solved")))

    print("\nObvious answers the problem rules out")
    for answer in result.get("blocked_answers") or []:
        print(wrap(answer, indent="  - "))

    print("\nCLUE STORY SKETCHES (same idea, distant setting)")
    for i, clue in enumerate(result.get("clue_ideas") or [], 1):
        print(wrap(f"{i}. [{clue.get('setting')}] {clue.get('sketch')}"))
        mapping = clue.get("mapping") or {}
        if mapping:
            print(wrap("matches: " + "; ".join(f"{p} -> {s}" for p, s in mapping.items()), indent="     "))

    lure = result.get("lure_idea") or {}
    print("\nDECOY SKETCH (same topic, wrong idea)")
    print(wrap(lure.get("sketch")))
    print(wrap(f"why it's wrong: {lure.get('why_wrong')}"))

    warnings = result.get("warnings") or []
    print("\nWARNINGS" if warnings else "\nWarnings: none")
    for w in warnings:
        print(wrap(w, indent="  - "))
    print("=" * 100)


# ---------- one finding, start to finish ----------

def process(text, title, url, published, cfg, client, system_prompt, out_dir, dry_run):
    if not text:
        print("  Nothing was pasted.")
        return
    cutoff_ok = check_cutoff(published)
    if cutoff_ok is False and not confirm("Process it anyway?"):
        return

    messages = build_messages(system_prompt, text, title, url, published)
    if dry_run:
        print("\n--- DRY RUN: this would be sent to", cfg["model"], "via", cfg["provider"], "---")
        print(messages[1]["content"])
        print("--- (system prompt:", PROMPT_PATH.relative_to(ROOT), ") ---")
        return

    print(f"\nAsking {cfg['model']} ({cfg['provider']})... this can take a minute.")
    try:
        response = chat(client, cfg, messages, cfg["seed"])
    except Exception as e:  # network / auth / rate-limit errors: report and keep the session alive
        print(f"  The API call failed: {e}")
        return

    raw = response.choices[0].message.content
    result = parse_json_reply(raw)
    missing = [k for k in REQUIRED_KEYS if result is not None and k not in result]

    now = dt.datetime.now()
    record = {
        "candidate_id": f"cand_{now:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}",
        "status": "candidate",
        "created": now.isoformat(timespec="seconds"),
        "source": {"title": title, "url": url, "published": published},
        "cutoff_ok": cutoff_ok,
        "input_text": text,
        "parse_ok": result is not None and not missing,
        "missing_keys": missing,
        "result": result,
        "writer": {
            "provider": cfg["provider"],
            "base_url": cfg["base_url"],
            "model": cfg["model"],
            "model_version": response.model,
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "temperature": cfg["temperature"],
            "max_completion_tokens": cfg["max_completion_tokens"],
            "reasoning_effort": cfg.get("reasoning_effort"),
            "seed": cfg["seed"],
            "prompt_file": str(PROMPT_PATH.relative_to(ROOT)),
            "prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        },
        "tokens_in": getattr(response.usage, "prompt_tokens", None),
        "tokens_out": getattr(response.usage, "completion_tokens", None),
        "raw_reply": raw,
    }
    path = save_record(record, out_dir)

    if result is None:
        print("  The model's reply was not valid JSON. The raw reply was saved for checking.")
    else:
        show(result)
        if missing:
            print(f"  Note: the reply is missing {', '.join(missing)}.")
    print(f"Saved: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Turn a research finding into a candidate problem + clue.")
    parser.add_argument("--file", type=Path, help="read one finding from this text file instead of pasting")
    parser.add_argument("--title", help="source title (with --file)")
    parser.add_argument("--url", help="source link or DOI (with --file)")
    parser.add_argument("--published", help="publication date YYYY-MM-DD (with --file)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="where to save results")
    parser.add_argument("--dry-run", action="store_true", help="show the request without calling the API")
    args = parser.parse_args()

    if args.published and not parse_date(args.published):
        sys.exit("--published must look like 2026-08-14.")

    cfg = load_model_config("writer")
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    client = None if args.dry_run else make_client(cfg)

    if args.file:
        text = args.file.read_text(encoding="utf-8").strip()
        process(text, args.title, args.url, args.published, cfg, client, system_prompt, args.out, args.dry_run)
        return

    print(f"Writer model: {cfg['model']} via {cfg['provider']}. Results go to {args.out}.")
    while True:
        text = read_pasted_text()
        if text is None:
            print("Done.")
            return
        title, url, published = ask_source_details()
        process(text, title, url, published, cfg, client, system_prompt, args.out, args.dry_run)


if __name__ == "__main__":
    main()
