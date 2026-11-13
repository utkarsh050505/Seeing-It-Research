# Problem Generator

This folder builds the test items (the benchmark) for our study **"Seeing It, or Seeing It Everywhere?"**. The study asks whether AI models that are stuck on a problem notice a hint hidden in an unrelated story, and whether they also start "seeing" hints that aren't there.

Each test item is:
- a **problem** built from a real discovery published on or after 1 July 2026, so no model we test can have learned the answer;
- a **short hint**: the same idea, happening in everyday life.

Before an item is kept, it must pass two checks: no model already knows the discovery, and models fail the problem alone but solve it with the hint.

The full plan, rules and every decision are in [context.md](context.md). The running experiment will live in the `Experimentation` folder.

---

## How an item is made

| Step | What happens | Script |
|---|---|---|
| 1. Statement | The team writes one row per discovery in `data/sources/statements.csv`: <ul><li>what the problem was and how it was solved</li><li>the one idea behind the fix (the "principle")</li><li>the tempting wrong answer (the "trap")</li></ul> | written by hand |
| 2. Memory check | Each test model is asked about the discovery. If any model already knows it, the item is dropped. | `src/generate/memory_check.py` |
| 3. Problem + hint | An AI writer (gpt-oss-120b) writes the problem, the expected answer and the hint. A second AI call checks them, and the writer tries again if the check fails. A person still reads every one. | `src/generate/make_problem.py` |
| 4. Difficulty test | Every test model tries the problem 10 times alone and 10 times with the hint. An AI marker checks each answer. Keep the item only if it's solved alone at most 30% of the time and with the hint at least 80% of the time. | `src/generate/difficulty_filter.py` |
| 5. Stories, human check, freeze | Not built yet (see context.md section 3.2). | |

**Example (CHEM-1):**
- **Discovery:** electrons land on whichever molecule they meet first, but the unwanted molecule goes back to how it was, while the wanted one becomes the product.
- **Idea:** let everything in, let the wrong ones undo themselves, and let only the right one lock in.
- **Hint:** a downpour soaks the whole garden; the dandelions dry back, while the tomato vines keep the water and their fruit swells.

## Setup

1. **Python 3.10 or newer**, then:
   ```
   pip install -r requirements.txt
   ```
2. **API keys.** Make a file called `.env` in this folder with your Groq keys (free at console.groq.com):
   ```
   GROQ_API_KEY=...
   GROQ_API_KEY_2=...
   ```
   - Every key whose name starts with `GROQ_API_KEY` is used, and they take turns, so their limits add up. Limits are per Groq account, so keys from different accounts help; two keys from one account don't.
   - **Never commit `.env`.** It is in `.gitignore`.
3. **Test models (lab PC only).** We run our own Ollama on port 11500, because the lab PC is shared and someone else uses the usual port. Start it in its own PowerShell window and leave it open:
   ```
   $env:OLLAMA_HOST="127.0.0.1:11500"; $env:OLLAMA_MODELS="E:\LLMxPsychology-models"; ollama serve
   ```
   - First time only: download the models with `src/setup/download_models.ps1`.
   - Check they all load and fit on the GPU with `python src/setup/test_models.py`.

## Running it

Run everything from this folder. `CHEM-1` stands for any code in `statements.csv`.

```
python src/generate/memory_check.py CHEM-1                # step 2 (needs Ollama)
python src/generate/make_problem.py --auto CHEM-1         # step 3 (only the Groq keys needed)
python src/generate/difficulty_filter.py CHEM-1           # step 4 (needs Ollama)
```

Useful extras:
```
python src/generate/make_problem.py --auto                # problem + hint for every statement that needs one
python src/generate/make_problem.py --auto --again CHEM-1 # a fresh problem + hint (the old ones stay saved)
python src/generate/make_problem.py CHEM-1                # choose or edit the problem yourself
python src/generate/difficulty_filter.py CHEM-1 --rejudge # mark the last test's answers again (no Ollama needed)
python src/generate/difficulty_filter.py CHEM-1 --groups weak   # test only one strength group
```

- **Before starting:** every script that runs many calls prints an estimate (time and Groq tokens) and asks first.
- **If a run stops:** run the same command again. This covers a closed window, a restart, or every key reaching its daily limit; the run carries on where it stopped.
- **More options:** the top of each script lists all of them.

## Where results go

Everything is saved under `data/candidates/<CODE>/` and is never overwritten. A new run makes a new folder named with the date and time.

```
data/candidates/CHEM-1/
  memory_<date>/          memory check: questions, every answer, every mark, summary.txt
  problem_<date>/         one problem + hint attempt
    versions/             every problem the writer made, with the checker's notes
    kept.json             the problem that was kept
    clues/, clue.json     every hint, and the kept one (with the idea split into parts)
    filter_<date>/        a difficulty test: run.json, answers/, marks/, summary.txt
data/candidates/problems_review_<date>.csv   every statement's problem and hint, for the team to read
```

The newest kept problem and hint count.

## Settings

All model names and settings are in [config/models.yaml](config/models.yaml), never in the code. This includes:
- which models are tested, and their strength group (strong / middle / weak);
- temperatures, seeds and token limits;
- the difficulty-test limits (10 tries, 30%, 80%).

The AI prompts are plain text files in [src/generate/prompts/](src/generate/prompts/).

## Rules

- **Discoveries:** only those first published on or after **1 July 2026**.
- **Writer and marker:** gpt-oss-120b, and it must never be one of the tested models. No Llama, Qwen, Gemma or OLMo as writer or marker.
- **Saved runs:** never edit or delete them. Make a new run instead.
- **Keep off GitHub (this repo is public):** `.env` and `data/`. About 30% of the final items must never be posted online, so models can't learn them.
- **People check:** a person reads every problem and hint, and the AI marker must be checked against hand marks before its numbers are trusted.

## Files

```
config/models.yaml          models, settings, limits
src/generate/               the pipeline scripts (above) and their prompts
  llm.py                    calls the writer/marker; the Groq keys take turns
  local_llm.py              calls the test models on our Ollama
  statements.py             reads statements.csv
  extract.py                an older first tool, no longer part of the pipeline
src/setup/                  lab PC setup: find, download and test the models
context.md                  the full research plan and decisions
```
