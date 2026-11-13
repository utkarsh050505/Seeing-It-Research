r"""
Check which language models are already on this computer (meant for the lab PC).

Needs only Python 3.8+; nothing to install. Run it from any folder:
    python check_lab_pc.py                 # normal check: looks 5 folders deep on every drive
    python check_lab_pc.py --deep          # looks through every folder on every drive (slower)
    python check_lab_pc.py --folder E:\x   # also search this folder fully (repeatable)

It looks where models usually live:
  - Ollama's model folder (also custom ones, e.g. on E:)
  - the Hugging Face download folder
  - loose .gguf files (llama.cpp, LM Studio)
It also prints the GPU, free disk space, and which of our 10 test models are here and how compressed they are
(full size, 8-bit, 4-bit, ...). Nothing is downloaded or changed. The report is also saved as a .txt file in the
folder you run it from, so it can be sent back.
"""

import argparse
import datetime as dt
import importlib.util
import json
import os
import platform
import re
import shutil
import string
import subprocess
import sys
import urllib.request

# Our test models and how we want each one on the lab PC (context.md section 4).
# (name shown in the report, family, size, variant that must appear in the name, wanted level)
NEEDED = [
    ("Llama 3.2 3B Instruct", "llama3.2", "3b", None, "full"),
    ("Qwen3.5 4B", "qwen3.5", "4b", None, "full"),
    ("Llama 3.1 8B Instruct", "llama3.1", "8b", None, "full"),
    ("Qwen3.5 9B", "qwen3.5", "9b", None, "full"),
    ("OLMo 3 7B Instruct", "olmo3", "7b", "instruct", "full"),
    ("OLMo 3 7B Think", "olmo3", "7b", "think", "full"),
    ("Gemma 4 12B", "gemma4", "12b", None, "8-bit"),
    ("Qwen3 14B", "qwen3", "14b", None, "8-bit"),
    ("Qwen3.5 27B", "qwen3.5", "27b", None, "api"),
    ("Gemma 4 31B", "gemma4", "31b", None, "api"),
]
WANT_TEXT = {"full": "full size", "8-bit": "8-bit or better", "api": "via API"}
LEVEL_RANK = {"full": 4, "8-bit": 3, "6-bit": 2.5, "5-bit": 2, "4-bit": 1, "3-bit or less": 0.5, "unknown": 0,
              "no weights": -1}
WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".pth", ".gguf")

SKIP_DIRS = {"windows", "program files", "program files (x86)", "system volume information", "node_modules",
             ".git", "__pycache__", "winsxs", "$recycle.bin", "recovery", "msocache"}
SIZE_RE = re.compile(r"[-_:/ ](\d+(?:\.\d+)?)b(?![a-z0-9])", re.I)
QUANT_RE = re.compile(r"(?i)(?:^|[-_.:])(IQ[1-4]_[A-Z]+|Q[2-8]_K(?:_[SML])?|Q[4-8]_[01]|BF16|FP16|F16|F32|FP8)(?=[-_.]|$)")


# ---------- small helpers ----------

def gb(n_bytes):
    return f"{n_bytes / 1e9:.1f} GB"


def level_from_quant(quant):
    """Turn a compression label (Q4_K_M, Q8_0, BF16, ...) into plain words."""
    q = (quant or "").upper()
    if q in ("F32", "F16", "BF16", "FP16", "BFLOAT16", "FLOAT16", "FLOAT32"):
        return "full"
    if q.startswith(("Q8", "FP8", "INT8")):
        return "8-bit"
    if q.startswith("Q6"):
        return "6-bit"
    if q.startswith("Q5"):
        return "5-bit"
    if q.startswith(("Q4", "IQ4")):
        return "4-bit"
    if q.startswith(("Q3", "Q2", "IQ3", "IQ2", "IQ1")):
        return "3-bit or less"
    return "unknown"


def quant_in_name(text):
    m = QUANT_RE.search(text or "")
    return m.group(1) if m else None


def family_size(label):
    """'qwen3.5:27b-q4_K_M' -> ('qwen3.5', '27b', '-q4_k_m'); 'meta-llama/Llama-3.2-3B-Instruct' -> ('llama3.2', '3b', '-instruct')."""
    s = label.lower().replace("\\", "/").split("/")[-1]
    m = SIZE_RE.search(s)
    if not m:
        return None, None, s
    family = re.sub(r"[-_ ]", "", s[:m.start()]).replace("metallama", "llama")
    return family, m.group(1) + "b", s[m.end():]


def matches(label, family, size, variant):
    fam, sz, rest = family_size(label)
    if fam is None or sz != size:
        return False
    fam_ok = fam == family or (family == "olmo3" and fam.startswith("olmo3"))
    return fam_ok and (variant is None or variant in rest)


def folder_size(path):
    total = 0
    for dirpath, _, files in os.walk(path, onerror=lambda e: None):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def run(cmd, timeout=15):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (out.stdout or out.stderr or "").strip()
    except (OSError, subprocess.SubprocessError):
        return None


# ---------- where to look ----------

def user_homes():
    homes = {os.path.expanduser("~")}
    users = os.path.join(os.environ.get("SystemDrive", "C:") + os.sep, "Users") if os.name == "nt" else "/home"
    if os.path.isdir(users):
        for d in os.listdir(users):
            p = os.path.join(users, d)
            if os.path.isdir(p):
                homes.add(p)
    return sorted(homes)


def fixed_drives():
    if os.name != "nt":
        return ["/"]
    import ctypes
    drives = []
    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        if os.path.exists(root) and ctypes.windll.kernel32.GetDriveTypeW(root) == 3:  # 3 = local fixed disk
            drives.append(root)
    return drives


def ollama_folders_from_settings():
    """OLLAMA_MODELS from this window, from Windows user/system settings, and each user's default folder."""
    found = []
    if os.environ.get("OLLAMA_MODELS"):
        found.append(os.environ["OLLAMA_MODELS"])
    if os.name == "nt":
        import winreg
        places = [(winreg.HKEY_CURRENT_USER, r"Environment"),
                  (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")]
        for hive, key in places:
            try:
                with winreg.OpenKey(hive, key) as k:
                    found.append(os.path.expandvars(winreg.QueryValueEx(k, "OLLAMA_MODELS")[0]))
            except OSError:
                pass
    found += [os.path.join(h, ".ollama", "models") for h in user_homes()]
    return found


def hf_folders_from_settings():
    found = [os.environ[v] for v in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE") if os.environ.get(v)]
    if os.environ.get("HF_HOME"):
        found.append(os.path.join(os.environ["HF_HOME"], "hub"))
    found += [os.path.join(h, ".cache", "huggingface", "hub") for h in user_homes()]
    return found


def is_ollama_folder(path):
    return os.path.isdir(os.path.join(path, "manifests")) and os.path.isdir(os.path.join(path, "blobs"))


def is_hf_hub(path):
    try:
        return any(d.startswith("models--") for d in os.listdir(path))
    except OSError:
        return False


def search(roots, max_depth, ollama, hubs, ggufs):
    """Walk folders looking for Ollama folders, Hugging Face hubs and .gguf files."""
    for root in roots:
        base = root.rstrip("\\/").count(os.sep)
        for dirpath, dirs, files in os.walk(root, onerror=lambda e: None):
            if is_ollama_folder(dirpath):
                ollama.add(os.path.normpath(dirpath))
                dirs[:] = []
                continue
            if os.path.basename(dirpath).lower() == "hub" and is_hf_hub(dirpath):
                hubs.add(os.path.normpath(dirpath))
                dirs[:] = []
                continue
            for f in files:
                if f.lower().endswith(".gguf"):
                    ggufs.add(os.path.normpath(os.path.join(dirpath, f)))
            depth = dirpath.rstrip("\\/").count(os.sep) - base
            dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS and not d.startswith("$")]
            if max_depth is not None and depth >= max_depth:
                dirs[:] = []


# ---------- reading what's there ----------

def read_ollama(folder):
    models = []
    man_root = os.path.join(folder, "manifests")
    for dirpath, _, files in os.walk(man_root, onerror=lambda e: None):
        for f in files:
            path = os.path.join(dirpath, f)
            parts = os.path.relpath(path, man_root).split(os.sep)
            if len(parts) < 3:
                continue
            host, ns, model, tag = parts[0], (parts[1] if len(parts) >= 4 else ""), parts[-2], parts[-1]
            if host == "registry.ollama.ai" and ns == "library":
                name = f"{model}:{tag}"
            else:
                name = "/".join(p for p in (host, ns, model) if p) + ":" + tag
            try:
                with open(path, encoding="utf-8") as fh:
                    manifest = json.load(fh)
            except (OSError, ValueError):
                continue
            size = sum(l.get("size", 0) for l in manifest.get("layers", []) if "image.model" in l.get("mediaType", ""))
            quant = params = None
            digest = (manifest.get("config") or {}).get("digest")
            if digest:
                try:
                    with open(os.path.join(folder, "blobs", digest.replace(":", "-")), encoding="utf-8") as fh:
                        cfg = json.load(fh)
                    quant, params = cfg.get("file_type"), cfg.get("model_type")
                except (OSError, ValueError):
                    pass
            quant = quant or quant_in_name(name)
            models.append({"where": "Ollama", "name": name, "quant": quant, "level": level_from_quant(quant),
                           "params": params, "size": size, "path": folder})
    return models


def read_hf_hub(hub):
    models = []
    for d in sorted(os.listdir(hub)):
        if not d.startswith("models--"):
            continue
        repo = d[len("models--"):].replace("--", "/")
        base = os.path.join(hub, d)
        snapshots = os.path.join(base, "snapshots")
        gguf_files, config, has_weights = [], {}, False
        for dirpath, _, files in os.walk(snapshots, onerror=lambda e: None):
            for f in files:
                has_weights = has_weights or f.lower().endswith(WEIGHT_EXT)
                if f.lower().endswith(".gguf"):
                    gguf_files.append(os.path.join(dirpath, f))
                elif f == "config.json" and not config:
                    try:
                        with open(os.path.join(dirpath, f), encoding="utf-8") as fh:
                            config = json.load(fh)
                    except (OSError, ValueError):
                        pass
        if gguf_files:
            for g in gguf_files:
                quant = quant_in_name(os.path.basename(g))
                models.append({"where": "Hugging Face", "name": f"{repo} ({os.path.basename(g)})", "quant": quant,
                               "level": level_from_quant(quant), "params": None, "size": os.path.getsize(g), "path": g})
            continue
        qc = config.get("quantization_config") or {}
        if qc:
            bits = qc.get("bits") or (4 if qc.get("load_in_4bit") else 8 if qc.get("load_in_8bit") else None)
            quant = f"{qc.get('quant_method', 'quantized')} {bits}-bit" if bits else qc.get("quant_method", "quantized")
            level = "8-bit" if bits == 8 or qc.get("quant_method") == "fp8" else "4-bit" if bits == 4 else "unknown"
        else:
            quant = config.get("torch_dtype") or config.get("dtype")
            level = level_from_quant(quant) if quant else "unknown"
        if not has_weights:  # only config/tokenizer files were downloaded, not the model itself
            quant, level = "only small files", "no weights"
        # On Windows the Hugging Face cache often keeps files in snapshots/ and leaves blobs/ empty.
        size = folder_size(os.path.join(base, "blobs")) or folder_size(snapshots)
        models.append({"where": "Hugging Face", "name": repo, "quant": quant, "level": level, "params": None,
                       "size": size, "path": base})
    return models


def read_gguf(path):
    quant = quant_in_name(os.path.basename(path))
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    return {"where": "GGUF file", "name": os.path.basename(path), "quant": quant, "level": level_from_quant(quant),
            "params": None, "size": size, "path": path}


# ---------- the machine itself ----------

def machine_lines():
    lines = [f"Computer: {platform.node()} | {platform.system()} {platform.release()} | Python {sys.version.split()[0]}"]

    gpu = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"])
    lines.append(f"GPU: {gpu}" if gpu else "GPU: nvidia-smi not found (no NVIDIA driver, or not on PATH)")

    disks = []
    for d in fixed_drives():
        try:
            u = shutil.disk_usage(d)
            disks.append(f"{d.rstrip(os.sep)} {gb(u.free)} free of {gb(u.total)}")
        except OSError:
            pass
    lines.append("Free disk: " + " | ".join(disks))

    exe = shutil.which("ollama")
    default_exe = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe")
    if not exe and os.path.isfile(default_exe):
        exe = default_exe
    version = run([exe, "--version"]) if exe else None
    version = version.splitlines()[-1] if version else None
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=2) as r:
            running = "yes, version " + json.load(r).get("version", "?")
    except Exception:
        running = "no"
    lines.append(f"Ollama: {'installed (' + (version or exe) + ')' if exe else 'not found'} | running now: {running}")

    lines.append(f"llama.cpp server (llama-server): {shutil.which('llama-server') or 'not found'}")

    packages = {"openai": "openai", "pyyaml": "yaml", "python-dotenv": "dotenv", "huggingface_hub": "huggingface_hub"}
    have = [f"{name} {'yes' if importlib.util.find_spec(mod) else 'no'}" for name, mod in packages.items()]
    lines.append("Python packages our pipeline uses: " + ", ".join(have))
    return lines


# ---------- report ----------

def status_for(need, models):
    label, family, size, variant, want = need
    hits = [m for m in models if matches(m["name"], family, size, variant)]
    best = max(hits, key=lambda m: LEVEL_RANK[m["level"]], default=None)
    if best:
        have = f"{best['level']} ({best['where']}: {best['name']}, {gb(best['size'])})"
        if len(hits) > 1:
            have += f" + {len(hits) - 1} more copy(ies)"
    else:
        have = "not on this PC"
    if want == "api":
        todo = "plan: API" + ("; local copy is a backup" if best else "")
    elif best and LEVEL_RANK[best["level"]] >= LEVEL_RANK[want]:
        todo = "OK"
    else:
        todo = f"NEED {WANT_TEXT[want]} version"
    return label, WANT_TEXT[want], have, todo, hits


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="List the language models already on this computer.")
    parser.add_argument("--deep", action="store_true", help="search every folder on every drive (slower)")
    parser.add_argument("--depth", type=int, default=5, help="how many folders deep to look on each drive (default 5)")
    parser.add_argument("--folder", action="append", default=[], help="also search this folder fully (repeatable)")
    args = parser.parse_args()

    print("Looking for models. This can take a minute or two...")
    ollama = {os.path.normpath(p) for p in ollama_folders_from_settings() if is_ollama_folder(p)}
    hubs = {os.path.normpath(p) for p in hf_folders_from_settings() if os.path.isdir(p) and is_hf_hub(p)}
    ggufs = set()
    lmstudio = [os.path.join(h, sub) for h in user_homes() for sub in (os.path.join(".lmstudio", "models"),
                                                                     os.path.join(".cache", "lm-studio", "models"))]
    search([p for p in lmstudio + args.folder if os.path.isdir(p)], None, ollama, hubs, ggufs)
    search(fixed_drives(), None if args.deep else args.depth, ollama, hubs, ggufs)

    models = []
    for folder in sorted(ollama):
        models += read_ollama(folder)
    for hub in sorted(hubs):
        models += read_hf_hub(hub)
    hub_paths = [h.lower() for h in hubs]
    models += [read_gguf(g) for g in sorted(ggufs) if not any(g.lower().startswith(h) for h in hub_paths)]

    now = dt.datetime.now()
    out = [f"LAB PC MODEL CHECK - {now:%Y-%m-%d %H:%M}", ""] + machine_lines() + ["", "Model folders found:"]
    out += [f"  Ollama: {f} ({sum(m['path'] == f for m in models)} models)" for f in sorted(ollama)] or ["  Ollama: none"]
    out += [f"  Hugging Face: {h}" for h in sorted(hubs)] or ["  Hugging Face: none"]
    out.append(f"  Loose .gguf files: {sum(m['where'] == 'GGUF file' for m in models)}")
    out.append("  (searched " + ("every folder" if args.deep else f"up to {args.depth} folders deep") + " on each drive)")

    out += ["", "OUR 10 TEST MODELS", f"  {'Model':<24}{'Want':<18}{'To do':<36}Have"]
    used = set()
    for need in NEEDED:
        label, want, have, todo, hits = status_for(need, models)
        used.update(id(m) for m in hits)
        out.append(f"  {label:<24}{want:<18}{todo:<36}{have}")

    others = [m for m in models if id(m) not in used]
    out += ["", f"EVERYTHING ELSE FOUND ({len(others)})"]
    for m in sorted(others, key=lambda m: (m["where"], m["name"].lower())):
        quant = m["quant"] or "?"
        name = m["name"] if len(m["name"]) <= 56 else m["name"][:53] + "..."
        out.append(f"  {m['where']:<13}{name:<58}{m['level']:<14}{quant:<17}{gb(m['size']):>9}")
    if not others:
        out.append("  (nothing)")

    report_path = os.path.abspath(f"lab_pc_check_{now:%Y%m%d_%H%M}.txt")
    out += ["", f"Saved this report to: {report_path}"]
    text = "\n".join(out)
    print("\n" + text)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
