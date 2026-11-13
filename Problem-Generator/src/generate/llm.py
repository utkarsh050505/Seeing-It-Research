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


USED_UP = set()        # keys that reached their daily limit during this run (by name, e.g. GROQ_API_KEY)
_NOTED_MISSING = set()


def make_client(cfg):
    """{key name: API client}, one per key listed in the provider's api_key_env that is set in .env.
    chat() uses them in order and moves on to the next when one reaches its daily limit."""
    names = cfg["api_key_env"]
    names = [names] if isinstance(names, str) else list(names)
    found = [n for n in names if os.environ.get(n)]
    if not found:
        sys.exit(
            f"\nNo API key found for provider '{cfg['provider']}'.\n"
            f"Add this line to the .env file in the repo root:  {names[0]}=your-key-here"
        )
    missing = [n for n in names if n not in found and n not in _NOTED_MISSING]
    if missing:
        print(f"  Note: {', '.join(missing)} not in .env, so only {', '.join(found)} will be used.")
        _NOTED_MISSING.update(missing)
    from openai import OpenAI  # imported here so dry runs work without the package
    return {n: OpenAI(base_url=cfg["base_url"], api_key=os.environ[n]) for n in found}


def current_key(client):
    """The key the next call will use (the first one that hasn't reached its daily limit), or None."""
    return next((n for n in client if n not in USED_UP), None)


def chat(client, cfg, messages, seed):
    params = dict(
        model=cfg["model"],
        messages=messages,
        temperature=cfg["temperature"],
        max_completion_tokens=cfg["max_completion_tokens"],
        seed=seed,
    )
    if cfg.get("reasoning_effort"):
        params["reasoning_effort"] = cfg["reasoning_effort"]
    error = None
    while (name := current_key(client)) is not None:
        try:
            return client[name].chat.completions.create(**params)
        except Exception as e:
            if "per day" not in str(e):
                raise
            USED_UP.add(name)
            error = e
            if current_key(client):
                print(f"\n  {name} reached its daily limit; going on with {current_key(client)} ", end="", flush=True)
    raise error or RuntimeError(f"tokens per day: every key ({', '.join(client)}) reached its daily limit")


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
            "api_key_name": current_key(client),   # which account answered (the name in .env, never the key)
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
