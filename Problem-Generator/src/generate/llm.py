"""Shared helpers for calling the writer model (gpt-oss via an OpenAI-compatible provider)."""

import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "models.yaml"

# API keys live in .env at the repo root (git-ignored). A key already set in the environment wins.
load_dotenv(ROOT / ".env", override=False)


def load_model_config(section):
    """Settings from config/models.yaml[section], merged with the chosen provider's details."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    cfg = dict(config[section])
    provider = cfg["provider"]
    if provider not in config["providers"]:
        sys.exit(f"Provider '{provider}' is not listed under `providers` in {CONFIG_PATH}.")
    cfg.update(config["providers"][provider])
    return cfg


# Several API keys (e.g. one per Groq account) take turns, call by call, so their per-minute and daily limits
# add up. All of this is by key NAME (GROQ_API_KEY_3); the keys themselves are never printed or saved.
USED_UP = set()        # keys that reached their daily limit during this run
RESTING = {}           # key name -> time (time.monotonic) when its per-minute limit frees up again
ACCOUNT = {}           # key name -> the Groq account it belongs to, read from Groq's refusal messages
LAST_KEY = None        # the key that answered the last call (saved with every record)
_TURN = [0]            # which key goes next
_NOTED = set()         # notes already printed once
RATE_LIMIT_TRIES = 60  # per-minute refusals for one call (all keys together) before giving up on it


def key_names(cfg, environ=os.environ):
    """The variable names in the provider's api_key_env. A name ending in * means every variable that starts
    with it, in number order: GROQ_API_KEY* = GROQ_API_KEY, GROQ_API_KEY_2, ..., GROQ_API_KEY_10."""
    listed = cfg["api_key_env"]
    listed = [listed] if isinstance(listed, str) else list(listed)
    names = []
    for entry in listed:
        if entry.endswith("*"):
            matches = [n for n in environ if n.startswith(entry[:-1])]
            names += sorted(matches, key=lambda n: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", n)])
        else:
            names.append(entry)
    return list(dict.fromkeys(names))


def make_client(cfg):
    """{key name: API client}, one per key named in the provider's api_key_env that is set (in .env or the
    environment). chat() lets them take turns. The same key pasted under two names counts once."""
    names, found, values = key_names(cfg), [], set()
    for n in names:
        value = os.environ.get(n, "").strip()
        if value and value not in values:
            found.append(n)
            values.add(value)
    if not found:
        first = names[0] if names else str(cfg["api_key_env"]).rstrip("*")
        sys.exit(
            f"\nNo API key found for provider '{cfg['provider']}'.\n"
            f"Add this line to the .env file in the repo root:  {first}=your-key-here"
        )
    notes = [n for n in names if n not in found]
    if tuple(found) not in _NOTED:
        _NOTED.add(tuple(found))
        skipped = f" ({', '.join(notes)}: empty or the same key again, skipped)" if notes else ""
        if len(found) > 1 or notes:
            using = f"{len(found)} keys take turns: " if len(found) > 1 else "using "
            print(f"  {cfg['provider']}: {using}{', '.join(found)}{skipped}")
    from openai import OpenAI  # imported here so dry runs work without the package
    # max_retries=0: chat() handles refusals itself (it moves on to another key instead of waiting on this one)
    return {n: OpenAI(base_url=cfg["base_url"], api_key=os.environ[n].strip(), max_retries=0) for n in found}


def _retry_seconds(text, default=30.0):
    """'... Please try again in 1m2.5s ...' -> 62.5"""
    m = re.search(r"try again in ([\d.hms]+)", text)
    total = sum(float(num) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
                for num, unit in re.findall(r"([\d.]+)(ms|h|m|s)", m.group(1))) if m else 0
    return total or default


def _note_account(name, text):
    """Groq's refusals name the account. Two keys from one account share one limit, so say so once."""
    m = re.search(r"organization `?(org_\w+)", text)
    if not m:
        return
    ACCOUNT[name] = m.group(1)
    same = tuple(sorted(n for n, a in ACCOUNT.items() if a == m.group(1)))
    if len(same) > 1 and same not in _NOTED:
        _NOTED.add(same)
        print(f"\n  Note: {' and '.join(same)} are from the same Groq account, so they share one limit. ",
              end="", flush=True)


def _hiccup(e):
    """A network problem or a server error: worth trying again."""
    status = getattr(e, "status_code", None)
    return type(e).__name__ in ("APIConnectionError", "APITimeoutError") or (status is not None and status >= 500)


def chat(client, cfg, messages, seed):
    """One call. The keys take turns; a key at its per-minute limit rests until Groq says it frees up while the
    others carry on (we only wait when every key is resting); a key at its daily limit is skipped for the rest
    of the run. Raises the daily-limit error once every key has reached it."""
    global LAST_KEY
    params = dict(
        model=cfg["model"],
        messages=messages,
        temperature=cfg["temperature"],
        max_completion_tokens=cfg["max_completion_tokens"],
        seed=seed,
    )
    if cfg.get("reasoning_effort"):
        params["reasoning_effort"] = cfg["reasoning_effort"]
    names = list(client)
    refusals, hiccups, error = 0, 0, None
    while True:
        live = [n for n in names if n not in USED_UP]
        if not live:
            raise error or RuntimeError(f"tokens per day: every key ({', '.join(names)}) reached its daily limit")
        now = time.monotonic()
        ready = [n for n in live if RESTING.get(n, 0) <= now]
        if not ready:
            wait = min(RESTING[n] for n in live) - now
            if wait >= 20:
                print(f"\n  Every key is at its per-minute limit; waiting {wait:.0f} s ", end="", flush=True)
            time.sleep(wait)
            continue
        start = _TURN[0] % len(names)
        name = next(n for n in names[start:] + names[:start] if n in ready)
        _TURN[0] = names.index(name) + 1
        try:
            response = client[name].chat.completions.create(**params)
            LAST_KEY = name
            return response
        except Exception as e:
            text, error = str(e), e
            if "per day" in text:   # tokens or requests per day: this key is done for this run
                USED_UP.add(name)
                _note_account(name, text)
                if any(n not in USED_UP for n in names):
                    print(f"\n  {name} reached its daily limit; going on with the other keys ", end="", flush=True)
                continue
            if getattr(e, "status_code", None) == 429 or "Error code: 429" in text:   # per minute: rest this key
                _note_account(name, text)
                refusals += 1
                if refusals > RATE_LIMIT_TRIES:
                    raise
                RESTING[name] = time.monotonic() + _retry_seconds(text) + 0.5
                continue
            if _hiccup(e) and hiccups < 2:
                hiccups += 1
                time.sleep(2 * hiccups)
                continue
            raise


API_WAITS = (20, 40, 60)    # seconds before retrying a failed call (the Groq free plan allows 8,000 tokens a minute)


class DailyLimit(Exception):
    """The provider's tokens-per-day limit (Groq free plan: 200,000): waiting a minute won't help."""


def logged_call(client, cfg, prompt_path, user, seed=None):
    """One call with the system prompt in `prompt_path`, retried after a wait if the API fails.
    Returns (log record with every setting, raw reply text). Raises DailyLimit, or the error if the API keeps
    failing. Without a seed: the config's fixed seed (the judge), else a fresh random one (the writer)."""
    system = Path(prompt_path).read_text(encoding="utf-8")
    if seed is None:
        seed = cfg.get("seed") or random.SystemRandom().randrange(2**31)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for wait in (*API_WAITS, None):
        try:
            response = chat(client, cfg, messages, seed)
            break
        except Exception as e:   # rate limit, network, ...
            if "per day" in str(e):
                retry_in = re.search(r"try again in ([\dhms.]+)", str(e))
                wait_text = f"; it frees up enough in {retry_in.group(1).rstrip('.')}" if retry_in else ""
                keys = f" on every key ({', '.join(client)})" if len(client) > 1 else ""
                raise DailyLimit(f"Groq's daily token limit is reached{keys}{wait_text}.") from None
            if wait is None:
                raise
            print(f"\n  The API call failed ({str(e)[:150]}); trying again in {wait} s ", end="", flush=True)
            time.sleep(wait)
    raw = response.choices[0].message.content
    record = {
        "settings": {
            "provider": cfg["provider"],
            "api_key_name": LAST_KEY,   # which key answered (the name in .env, never the key)
            "model": cfg["model"],
            "model_version": response.model,
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "temperature": cfg["temperature"],
            "max_completion_tokens": cfg["max_completion_tokens"],
            "reasoning_effort": cfg.get("reasoning_effort"),
            "seed": seed,
            "prompt_file": str(Path(prompt_path).resolve().relative_to(ROOT)),
            "prompt_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
        },
        "tokens_in": getattr(response.usage, "prompt_tokens", None),
        "tokens_out": getattr(response.usage, "completion_tokens", None),
        "user_message": user,
        "raw_reply": raw,
    }
    return record, raw


def parse_json_reply(raw):
    """Pull the JSON object out of a reply, tolerating code fences or stray text around it."""
    if not raw:
        return None
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None
