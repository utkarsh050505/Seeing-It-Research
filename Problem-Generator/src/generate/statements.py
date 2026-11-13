"""
Where the tools get a statement (context.md section 3.2, step 2): a row of data/sources/statements.csv, picked by
its code (e.g. CHEM-1), or a statement pasted into the terminal.

Edit the CSV in Google Sheets or a text editor. Excel tends to rewrite the dates (2026-07-15 -> 15-07-2026).
"""

import csv
import datetime as dt
import re
import sys

from llm import ROOT

STATEMENTS_CSV = ROOT / "data" / "sources" / "statements.csv"
REQUIRED = ("title", "published", "source", "statement", "principle")
CUTOFF = dt.date(2026, 7, 1)  # findings must be published on or after this date (context.md §3.1)

DATE_PATTERN = re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3})[a-z]*\.?\s+(\d{4})\b")  # e.g. "15 Jul 2026"
CODE_PATTERN = re.compile(r"^([A-Za-z]+-?\d+)\b")   # "CHEM-1" from "CHEM-1. Choose afterwards ...", or "C1"
PRINCIPLE_PATTERN = re.compile(r"\bprinciple\s*:", re.IGNORECASE)  # may sit mid-line when pasted from a table


class StatementError(ValueError):
    """A CSV row that can't be used yet (missing field, unreadable date, unknown code)."""


def get_item(code=None):
    """The statement with this code from the CSV, or a pasted one if no code is given. None if the user quits.
    Both come back in the same form: code, title, statement (venue and date in brackets at the end), principle,
    trap, published, cutoff_ok."""
    if not code:
        return paste_statement()
    try:
        return read_statement(code)
    except StatementError as e:
        sys.exit(str(e))


# ---------- from the CSV ----------

def load_statements():
    """{code: row} for every filled row of the CSV."""
    with open(STATEMENTS_CSV, encoding="utf-8-sig", newline="") as f:   # -sig: Excel adds a BOM
        rows = [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]
    statements = {}
    for line, row in enumerate(rows, 2):   # line 1 is the header
        if not any(row.values()):
            continue
        code = row["code"].upper()
        if not code:
            sys.exit(f"{STATEMENTS_CSV.name} line {line}: the row has no code.")
        if code in statements:
            sys.exit(f"{STATEMENTS_CSV.name} line {line}: code {code} is used twice.")
        statements[code] = row
    return statements


def parse_published(text):
    """'2026-07-15' -> (date, '15 Jul 2026'). '2026-09' (day not known yet) -> (1 Sep 2026, 'Sep 2026').
    Anything else -> (None, None)."""
    try:
        d = dt.datetime.strptime(text, "%Y-%m-%d").date()
        return d, f"{d.day} {d:%b %Y}"
    except ValueError:
        pass
    try:
        d = dt.datetime.strptime(text, "%Y-%m").date()
        return d, f"{d:%b %Y}"
    except ValueError:
        return None, None


def read_statement(code, statements=None, quiet=False):
    """One CSV row as an item, plus the whole row for the record. Raises StatementError if the row can't be used.
    quiet=True skips printing the statement and its warnings."""
    statements = statements or load_statements()
    row = statements.get(code.upper())
    if row is None:
        raise StatementError(f"No statement {code} in {STATEMENTS_CSV.relative_to(ROOT)}. "
                             f"Codes: {', '.join(statements)}")
    code = code.upper()
    missing = [k for k in REQUIRED if not row[k]]
    if missing:
        raise StatementError(f"{code}: fill in {', '.join(missing)} in {STATEMENTS_CSV.name} first.")
    published, date_words = parse_published(row["published"])
    if published is None:
        raise StatementError(f"{code}: can't read the date '{row['published']}'. "
                             f"Write it as 2026-07-15, or 2026-07 if the day isn't known yet.")

    item = {"code": code, "title": f"{code}. {row['title']}",
            "statement": f"{row['statement']} ({row['source']}, {date_words})", "principle": row["principle"],
            "trap": row.get("trap", ""), "published": row["published"], "cutoff_ok": published >= CUTOFF,
            "csv_row": row}
    if quiet:
        return item
    print(f"\n{item['title']}\nStatement: {item['statement']}\nPrinciple: {item['principle']}")
    if not item["cutoff_ok"]:
        print(f"  WARNING: published {row['published']}, before the {CUTOFF} cutoff. Models may know this finding.")
    if row.get("verdict", "").lower() != "keep":
        note = f" ({row['verdict_note']})" if row.get("verdict_note") else ""
        print(f"  Note: the verdict for this statement is '{row.get('verdict') or 'blank'}'{note}, not Keep.")
    return item


# ---------- pasted ----------

def read_item():
    """Read a pasted block up to the 'Principle:' part. Returns None if the user quits."""
    print("\nPaste the statement and principle (Q to quit). It starts once it reads the 'Principle:' part:")
    title_lines, statement_lines, principle = [], [], None
    in_statement = False
    while principle is None:
        try:
            line = input().strip()
        except EOFError:
            return None
        if not line:
            continue
        if line.upper() in ("Q", "QUIT") and not (title_lines or statement_lines):
            return None
        match = PRINCIPLE_PATTERN.search(line)
        if match:
            principle = line[match.end():].strip()
            line = line[:match.start()].strip()
            if not line:
                continue
        label, _, rest = line.partition(":")
        if label.strip().lower() == "statement":
            in_statement = True
            statement_lines.append(rest.strip())
        elif in_statement:
            statement_lines.append(line)  # statement wrapped onto several lines
        else:
            title_lines.append(line)

    statement = " ".join(s for s in statement_lines if s)
    title = " ".join(title_lines) or None
    if not statement and title:  # pasted without a "Statement:" label
        statement, title = title, None
    return {"title": title, "statement": statement, "principle": principle}


def find_published_date(text):
    """First 'DD Mon YYYY' date in the text, as a date, or None."""
    for day, month, year in DATE_PATTERN.findall(text):
        try:
            return dt.datetime.strptime(f"{day} {month[:3].title()} {year}", "%d %b %Y").date()
        except ValueError:
            continue
    return None


def check_cutoff(statement):
    published = find_published_date(statement)
    if published is None:
        print("  Note: no publication date found in the statement. Check it is on or after", CUTOFF)
        return None, None
    if published < CUTOFF:
        print(f"  WARNING: published {published}, before the {CUTOFF} cutoff. Models may know this finding.")
        return published.isoformat(), False
    return published.isoformat(), True


def paste_statement():
    """A pasted statement as an item. The code comes from the title line ("CHEM-1. ..."), else 'item'."""
    item = read_item()
    if item is None:
        return None
    if not item["statement"] or not item["principle"]:
        sys.exit("Need both a statement and a principle.")
    published, cutoff_ok = check_cutoff(item["statement"])
    code_match = CODE_PATTERN.match(item["title"] or "")
    item.update(code=code_match.group(1).upper() if code_match else "item", trap="", published=published,
                cutoff_ok=cutoff_ok)
    return item
