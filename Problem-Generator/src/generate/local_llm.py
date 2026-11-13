"""Shared helpers for calling the test models that run on our own Ollama (lab PC, port in config/models.yaml)."""

import json
import time
import urllib.error
import urllib.request

import yaml

from llm import CONFIG_PATH

START_CMD = '$env:OLLAMA_HOST="127.0.0.1:11500"; $env:OLLAMA_MODELS="E:\\LLMxPsychology-models"; ollama serve'


def load_test_config():
    """(Ollama host, settings for every test-model call, list of test models) from config/models.yaml."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["ollama"]["host"], config["test_settings"], config["test_models"]


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


def server_info(host):
    """(Ollama version, {model id: digest}) or None if our Ollama is not running."""
    try:
        version = api(host, "/api/version", timeout=5).get("version")
    except Exception:
        return None
    digests = {m.get("name"): m.get("digest") for m in api(host, "/api/tags").get("models", [])}
    return version, digests


def unload(host, model_id):
    """Free the GPU before the next model. Waits up to 30 s for Ollama to let go."""
    try:
        api(host, "/api/generate", {"model": model_id, "keep_alive": 0}, timeout=60)
        for _ in range(30):
            if not any(model_id in (m.get("name"), m.get("model")) for m in api(host, "/api/ps").get("models", [])):
                return
            time.sleep(1)
    except Exception:
        pass


def split_reply(reply):
    """(thinking, final answer). Thinking comes back on its own, or inside <think> tags in the answer."""
    msg = reply.get("message", {})
    thinking, answer = msg.get("thinking") or "", msg.get("content") or ""
    if "</think>" in answer:
        inside, answer = answer.split("</think>", 1)
        thinking += inside.replace("<think>", "")
    return thinking.strip(), answer.strip()


def ask(host, settings, model, messages, seed):
    """One chat call to a local test model, thinking off where it can be switched off.
    Returns (request body, raw reply, thinking, final answer)."""
    always = model["thinking"] == "always"
    body = {"model": model["id"], "messages": messages, "stream": False,
            "options": {"temperature": settings["temperature"], "seed": seed, "num_ctx": settings["num_ctx"],
                        "num_predict": settings["max_tokens_thinking"] if always else settings["max_tokens"]},
            "keep_alive": "5m"}   # if the script stops halfway, the model still leaves the GPU after 5 minutes
    if model["thinking"] == "switch":
        body["think"] = False
    reply = api(host, "/api/chat", body, timeout=1800)
    reply.pop("context", None)
    thinking, answer = split_reply(reply)
    return body, reply, thinking, answer
