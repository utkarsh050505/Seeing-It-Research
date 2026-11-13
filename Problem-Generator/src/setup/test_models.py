r"""
Quick test of our test models on the lab PC: does each one load, fit fully on the GPU, and answer?

Needs only Python 3.8+; nothing to install. Our own Ollama must be running first, in its own window:
    $env:OLLAMA_HOST="127.0.0.1:11500"; $env:OLLAMA_MODELS="E:\LLMxPsychology-models"; ollama serve
Then, in a second window:
    python test_models.py                         # all 8 models, about 10-15 minutes
    python test_models.py --models qwen3.5:9b-bf16 qwen3.5:9b-q8_0
    python test_models.py --ctx 16384             # test with room for a shorter chat

It first checks whether someone else is using the GPU and asks before going on. Then, for each model, it loads it
with room for a long chat (--ctx tokens), asks one short question and reports: load time, how much of the model sits
on the GPU (anything under 100% runs much slower), speed in tokens per second, and the start of the answer. Models
that can think are asked twice, thinking on and off, to check the switch works. Each model is unloaded before the
next one. Nothing is downloaded or changed. The report is saved as a .txt, and every request and reply as a .json,
in the folder you run it from.
"""

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

# The 8 models download_models.ps1 put in our folder (context.md section 4). Qwen3.5 27B and Gemma 4 31B run
# through an API, so they are not tested here.
MODELS = [
    "llama3.2:3b-instruct-fp16",
    "qwen3.5:4b-bf16",
    "olmo-3:7b-instruct-fp16",
    "olmo-3:7b-think-fp16",
    "llama3.1:8b-instruct-fp16",
    "qwen3.5:9b-bf16",
    "gemma4:12b-it-q8_0",
    "qwen3:14b-q8_0",
]
# 32768 tokens is our guess at the longest chat (8 failed tries + 4 stories + the model's thinking).
# Ollama reserves GPU room for the whole chat length when it loads a model, so this tests the worst case.
DEFAULT_CTX = 32768
QUESTION = "In two sentences: why does ice float on water?"
SETTINGS = {"temperature": 0.7, "seed": 42, "num_predict": 1024}   # same temperature as the experiment
START_CMD = '$env:OLLAMA_HOST="127.0.0.1:11500"; $env:OLLAMA_MODELS="E:\\LLMxPsychology-models"; ollama serve'
BUSY_MIB = 2000   # more GPU memory than this in use before we start = someone else is probably on the GPU
BUSY_UTIL = 10    # same for % of the time the GPU is busy


# ---------- small helpers ----------

def run(cmd, timeout=15):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (out.stdout or out.stderr or "").strip()
    except (OSError, subprocess.SubprocessError):
        return None


def gpu_now():
    """(memory used MiB, memory total MiB, % busy) from nvidia-smi, or None if it can't be read."""
    out = run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"])
    try:
        used, total, util = (int(x) for x in out.splitlines()[0].split(","))
        return used, total, util
    except (AttributeError, ValueError, IndexError):
        return None


def gpu_programs():
    out = run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    return [line.strip() for line in (out or "").splitlines() if line.strip()]


# ---------- talking to our Ollama ----------

def api(host, path, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(f"http://{host}{path}", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:   # Ollama puts the reason in {"error": "..."}
        try:
            msg = json.load(e).get("error", "")
        except ValueError:
            msg = ""
        raise RuntimeError(msg or f"HTTP {e.code}") from None


def loaded(host):
    return api(host, "/api/ps").get("models", [])


def unload(host, model):
    try:
        api(host, "/api/generate", {"model": model, "keep_alive": 0}, timeout=60)
        for _ in range(30):
            if not any(model in (m.get("name"), m.get("model")) for m in loaded(host)):
                return
            time.sleep(1)
    except Exception:
        pass


def ask(host, model, ctx, think):
    body = {"model": model, "messages": [{"role": "user", "content": QUESTION}], "stream": False,
            "options": dict(SETTINGS, num_ctx=ctx),
            "keep_alive": "2m"}   # if this script stops halfway, the model still leaves the GPU after 2 minutes
    if think is not None:
        body["think"] = think
    reply = api(host, "/api/chat", body, timeout=900)
    reply.pop("context", None)
    return body, reply


def split_reply(reply):
    """(words of thinking, answer). Thinking comes back on its own, or inside <think> tags in the answer."""
    msg = reply.get("message", {})
    thinking, answer = msg.get("thinking") or "", msg.get("content") or ""
    if "</think>" in answer:
        inside, answer = answer.split("</think>", 1)
        thinking += inside.replace("<think>", "")
    return len(thinking.split()), " ".join(answer.split())


# ---------- one model ----------

def test_model(host, model, ctx, say):
    info = api(host, "/api/show", {"model": model})
    details = info.get("details", {})
    res = {"model": model, "stored_as": details.get("quantization_level", "?"),
           "params": details.get("parameter_size", "?"),
           "can_think": "thinking" in info.get("capabilities", []), "runs": []}
    say(f"  stored as {res['stored_as']}, {res['params']} parameters | can think: {'yes' if res['can_think'] else 'no'}")

    for think in ([True, False] if res["can_think"] else [None]):
        body, reply = ask(host, model, ctx, think)
        if not res["runs"]:   # the first question loaded the model: see where it sits
            entry = next((m for m in loaded(host) if model in (m.get("name"), m.get("model"))), {})
            size, vram = entry.get("size", 0), entry.get("size_vram", 0)
            res["all_on_gpu"] = size > 0 and vram >= size
            res["on_gpu_pct"] = int(100 * vram / size) if size else 0
            res["loaded_gb"] = round(size / 1e9, 1)
            res["context_length"] = entry.get("context_length") or ctx
            gpu = gpu_now()
            res["gpu_used_gb"] = round(gpu[0] / 1024, 1) if gpu else None
            say(f"  loaded in {reply.get('load_duration', 0) / 1e9:.0f} s | {res['on_gpu_pct']}% on GPU "
                f"({res['loaded_gb']} GB with room for {res['context_length']} tokens) | "
                f"GPU memory in use now: {res['gpu_used_gb'] if gpu else '?'} GB")

        words, answer = split_reply(reply)
        seconds = reply.get("eval_duration", 0) / 1e9
        tokens = reply.get("eval_count", 0)
        one = {"think": think, "tokens": tokens, "speed": round(tokens / seconds, 1) if seconds else 0,
               "thinking_words": words, "answer": answer, "stopped_because": reply.get("done_reason"),
               "request": body, "reply": reply}
        res["runs"].append(one)
        mode = {True: "thinking on ", False: "thinking off", None: "answer      "}[think]
        cut = " (hit the token limit)" if one["stopped_because"] == "length" else ""
        short = answer if len(answer) <= 70 else answer[:67] + "..."
        say(f"  {mode}: {one['speed']:.0f} tokens/s, {tokens} tokens, thought {words} words{cut} | \"{short}\"")
    return res


def thinking_note(res):
    if not res.get("can_think") or len(res.get("runs", [])) < 2:
        return "-"
    on, off = res["runs"][0]["thinking_words"], res["runs"][1]["thinking_words"]
    if on and not off:
        return "switch works"
    if off:
        return "thinks even when off"
    return "never thinks"


def verdict(res):
    if "error" in res:
        return "FAILED: " + res["error"]
    if not res.get("all_on_gpu"):
        return "PART ON CPU (slow): try 8-bit or a smaller --ctx"
    if not any(r["answer"] for r in res["runs"]):
        return "NO ANSWER"
    return "OK"


# ---------- report ----------

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Load each test model on our Ollama and check it fits and answers.")
    parser.add_argument("--host", default="127.0.0.1:11500", help="our Ollama (default 127.0.0.1:11500)")
    parser.add_argument("--ctx", type=int, default=DEFAULT_CTX, help=f"chat length to make room for (default {DEFAULT_CTX})")
    parser.add_argument("--models", nargs="+", default=MODELS, help="test only these (default: all 8)")
    args = parser.parse_args()

    try:
        version = api(args.host, "/api/version", timeout=5).get("version", "?")
    except Exception:
        print(f"STOP: our Ollama is not running on {args.host}. In another PowerShell window run this and leave it open:")
        print("  " + START_CMD)
        sys.exit(1)

    have = {m.get("name") for m in api(args.host, "/api/tags").get("models", [])}
    missing = [m for m in args.models if m not in have]
    models = [m for m in args.models if m in have]
    if missing:
        print("Not in our Ollama, skipped: " + ", ".join(missing))
    if not models:
        sys.exit(1)

    # Free the GPU of anything our own Ollama still holds, then see whether someone else is using it.
    for m in loaded(args.host):
        unload(args.host, m.get("name"))
    gpu = gpu_now()
    if gpu is None:
        gpu_line = "GPU: can't read it (nvidia-smi not found)"
    else:
        used, total, util = gpu
        gpu_line = f"GPU before we start: {used / 1024:.1f} of {total / 1024:.1f} GB memory in use, {util}% busy"
        if used > BUSY_MIB or util > BUSY_UTIL:
            print(gpu_line)
            print("Someone else seems to be using the GPU. Programs on it:")
            for p in gpu_programs() or ["(none listed)"]:
                print("  " + p)
            print("Our models need almost the whole GPU. If we start now they may not fit, the timings will be wrong,")
            print("and we could slow down or crash the other job.")
            if input("Run anyway? (y/N) ").strip().lower() != "y":
                sys.exit(0)

    start = dt.datetime.now()
    out = []

    def say(line):
        print(line, flush=True)
        out.append(line)

    say(f"MODEL TEST - {start:%Y-%m-%d %H:%M}")
    say(f"Our Ollama: {args.host}, version {version} | room for {args.ctx} tokens per chat | settings {SETTINGS}")
    say(gpu_line)
    say(f"Question: {QUESTION}")
    say(f"Testing {len(models)} models, one at a time. About 1-2 minutes each.")

    results = []
    for i, model in enumerate(models, 1):
        say(f"\n[{i}/{len(models)}] {model}")
        try:
            res = test_model(args.host, model, args.ctx, say)
        except Exception as e:   # out of memory, too slow, ...: note it and go on to the next model
            res = {"model": model, "error": str(e) or type(e).__name__}
            say(f"  FAILED: {res['error']}")
        finally:
            unload(args.host, model)
        results.append(res)

    say("\nSUMMARY")
    say(f"  {'Model':<27}{'Stored as':<11}{'On GPU':<8}{'Speed':<12}{'Thinking':<22}Result")
    for res in results:
        runs = res.get("runs", [])
        speed = max(runs, key=lambda r: r["tokens"])["speed"] if runs else 0
        on_gpu = f"{res['on_gpu_pct']}%" if "on_gpu_pct" in res else "-"
        say(f"  {res['model']:<27}{res.get('stored_as', '-'):<11}{on_gpu:<8}{f'{speed:.0f} tok/s':<12}"
            f"{thinking_note(res):<22}{verdict(res)}")
    say("\nEvery model should say OK with 100% on GPU.")

    stamp = f"{start:%Y%m%d_%H%M}"
    txt_path = os.path.abspath(f"model_test_{stamp}.txt")
    json_path = os.path.abspath(f"model_test_{stamp}.json")
    say(f"Saved this report to: {txt_path}")
    say(f"Every request and reply: {json_path}")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"date": start.isoformat(timespec="seconds"), "host": args.host, "ollama_version": version,
                   "ctx": args.ctx, "question": QUESTION, "settings": SETTINGS, "gpu_at_start": gpu,
                   "results": results}, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
