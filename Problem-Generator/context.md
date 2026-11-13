# Project context: "Seeing It, or Seeing It Everywhere?"

Read this before every task. This repo is a **research project**, not a product. Every script exists to produce clean, reproducible evidence for a paper. When a prompt is vague, assume the goal is: correct data → clear graph/table → plain-English takeaway.

The design below is the **current plan, not final**. If a prompt conflicts with it, follow the prompt, but say which part of this file it changes.

---

## 1. The research question

> **Does working longer on an unsolved problem make language models better at noticing a relevant clue in (superficially) unrelated information — or just more likely to see connections that aren't there — and do they show the effects psychology has documented in humans?**

Inspiration: Newton's apple / the shower moment. A person is stuck for hours, takes a break, and something unrelated makes it click.

- **RQ1 (seeing it):** Does being stuck (more failed tries) increase real use of a hidden clue? → *hit rate*
- **RQ2 (seeing it everywhere):** Does it increase claimed connections to stories that don't help? → *false-alarm rate*
- **RQ3 (human comparison):** Which human findings does the model reproduce? (see §8)

How to read results:

| Being stuck gives… | Meaning |
|---|---|
| more hits, same false alarms | real noticing ("seeing it") |
| more hits and more false alarms | just more eager to connect everything |
| more false alarms only | "seeing it everywhere" |
| no change / worse | no human-like effect (check long-chat confound) |

---

## 2. One trial = one chat in four parts

1. **Work:** model gets a hard problem; we reply "That wouldn't work. Try another idea." 0, 2 or 8 times.
2. **Break:** model summarises 4 short stories. The problem is **never mentioned** here.
    - Real-clue set: 1 clue story (same deep idea, different topic) + 1 lure (same topic/words, wrong idea) + 2 fillers.
    - No-clue set: 1 lure + 3 fillers.
3. **Return:** "Back to the problem. What's your best solution now?"
4. **Hint:** "One of the stories you read may help. Try again."

Example: Problem = cut a bolt deep inside an engine without burning nearby parts. Clue = a village turns a heavy mill wheel with many small streams meeting at it. Gold idea = many weak beams from different sides meeting at the bolt.

### The 10 versions (same chat, one change each)

| # | Part 1 (Work) | Part 2 (Stories) |
|---|---|---|
| 1 | skipped (problem first seen in Part 3) | real clue |
| 2 | skipped | no clue |
| 3 | 2 failed tries | real clue |
| 4 | 2 failed tries | no clue |
| 5 | 8 failed tries | real clue |
| 6 | 8 failed tries | no clue |
| 7 | unrelated filler chat, same number of turns/length | real clue |
| 8 | unrelated filler chat, same number of turns/length | no clue |
| 9 | fresh chat + one-line note of what was tried | real clue |
| 10 | fresh chat + one-line note | no clue |

Versions 7–10 answer the reviewer question: *"is it being stuck, or just a longer chat?"* Version 9 vs 5 tests the human "forgetting the wrong idea" theory of breaks.

### Hard rules for running chats

- **Never** mention the problem during the break.
- **Never** ask "which story helped?" before Part 4. That question is itself a hint.
- **Exclusion rule:** if any Part-1 answer already uses the gold idea, drop that chat (our "that wouldn't work" lied to it). Log how many were dropped per version.
- Shuffle story order in every chat and **log the order**.
- Same temperature, max tokens and system prompt across all versions of a model.
- The running script only sends messages and saves replies. It does **not** judge anything live.
- Never overwrite raw chats. Runs must be resumable (skip chats already saved).

---

## 3. Dataset (the benchmark)

### 3.1 Idea: build problems from real findings newer than every model

Each problem is built from a **real finding published after the newest test model was released**, so no model can have memorised it. Each finding must have a clear "problem the researchers faced → how they solved it" and a solution that rests on **one** transferable deep idea (a TRIZ principle).

- **Cutoff rule:** use only findings first published **on or after 1 July 2026**. The newest test model was released in June 2026, and several Qwen cutoffs are not published, so the release date is our safe line.
- **Domains** (problems come from these; aim for 8–12 items each):
    1. Astrophysics & Particle Detection
    2. Chemistry & Autocatalytic Systems
    3. Neuroscience & Biological Control
    4. Economics & Complex Adaptive Systems
    5. Swarm Robotics & Decentralised Intelligence
    6. Marine Biology & Ecosystem Restoration
    7. Materials Science & Self-Healing Systems
    8. Cybersecurity & Software Supply Chains

### 3.2 Pipeline

**Collect findings → team writes statement + principle → memory check → writer makes problem + short clue → difficulty filter → write stories → automatic filters → human check → freeze.**

The short clue is tested on the models *before* any story is written, so we only spend effort on stories for items that already work.

1. **Collect:** gather candidate findings (papers, preprints, official press releases). Save the source title, URL/DOI and publication date. Paraphrase; never copy text.
2. **Statement + principle** (team, by hand): for each finding, write a short plain statement (the problem the researchers faced and how they solved it) and a one-line principle: the deep idea behind the solution. Drop any finding whose solution doesn't reduce to one clean idea. The principle is the item's gold idea; also tag the closest TRIZ principle.
    - Each finding is one row of `data/sources/statements.csv`: `code, topic, title, published, source, paper_title, link, statement, principle, trap, added_by, verdict, verdict_note`. Codes are topic + number (`CHEM-1`, `CYB-5`, `NEU-1` ...), the same codes as the Nov quick check; a dropped code is never reused. `published` is `2026-07-15`, or `2026-07` until the day is known.
    - The tools take a code (`python src/generate/memory_check.py CHEM-1`). Pasting a statement still works, in this format:
    ```
    CHEM-1. Choose afterwards, not at the door
    Statement: A catalyst fires electrons into the solvent ... (Nature, 15 Jul 2026)
    Principle: Let everything in, then make the wrong ones undo themselves.
    ```
3. **Problem + short clue** (`src/generate/make_problem.py --auto CHEM-1` for one statement, `--auto` for all that need them).
    - **Problem** (2–4 sentences): the situation told as a puzzle someone is stuck on.
        - **Hide the solution.** The model sees the problem, never the answer. Otherwise it's just reading comprehension. If the principle is about where the harm comes from, the problem must not say where: it describes what is seen and asks what is going on and what would stop it.
        - Remove jargon, so a failure means "didn't notice," not "didn't know the field."
        - Block the obvious answers in the text, starting with the statement's Trap.
        - Use only facts from the statement.
    - The writer also gives the **answer** (what a correct reply says, in the problem's own words) and the **reason** (why it works and how it follows the principle), so judging has a concrete reference.
    - A **checker** (gpt-oss, temperature 0) reads each problem next to the team's statement and principle and flags a problem that gives the idea away, rules out the real answer, says something the statement doesn't, uses specialist words, or asks a question that doesn't fit its own facts (13 Nov: CHEM-1's said only A ends up as product, then asked how to make it selective); the writer then retries, up to 3 times. In the 11 Nov tests the writer alone leaked the key fact in most tries (it lists what to hide, then uses it anyway), so the checker stays. It is not perfect either (it missed 1 of 6 on CHEM-2), so a person still reads every problem.
    - **Short clue:** the same principle happening as an event in a *distant* setting (currently everyday life; see §10). Every part of the principle must appear in it, in the same roles. Never stated as a lesson. Written right after a problem is kept: the writer first splits the principle into parts (who does what) and maps each to an everyday event (this is the item's `gold_mapping`), then writes 2–3 sentences. The clue writer and checker also get the statement's Trap. The checker first writes what a stranger would take the scene's trick to be, then flags a scene that reads as the Trap (13 Nov: CHEM-1's coin machine "change-detector" read as a filter at the door, the exact trap), a missing or misplaced part (e.g. someone else does what should happen by itself, or the wrong ones die when they should go back to how they were), a stated lesson, or surface likeness to the problem; up to 3 tries. The clue does not have to be realistic (decided 12 Nov): these checks only catch obvious mistakes, and whether a clue works is decided by the difficulty filter (step 5: problem + clue ≥ 80%).
    - **Automatic** (`--auto`): every statement with verdict Keep that wasn't dropped by the memory check gets a problem (the first version that passes its checker) and then a clue (the same way). Statements where nothing passed in 3 tries are listed (problem: for a person; clue: rerun with `--retry`). It stops at Groq's daily limit and carries on when rerun. `--auto --again CHEM-1` makes a fresh problem and clue for one that has them (the old ones stay saved; the newest counts). Each run writes `data/candidates/problems_review_<date>.csv` with every statement's problem, clue and status, for the team to read through.
    - **By hand** (`make_problem.py CHEM-1`), for the leftovers or to replace one: see each problem version and keep one, or write or edit one yourself (`E`; the answer, reason and key step carry over, and the checker still reads it); the clue is then written automatically. Every version is saved; the kept problem is `kept.json`, its clue `clue.json`, and the newest kept problem counts.
4. **Memory check** (`src/generate/memory_check.py`), run on every test model right after the statement is written, before effort goes into the problem and clue (so it now comes before step 3): ask about the finding without giving the answer. If a model already knows how it was solved, drop the item for all models. The audit model (OLMo, open training data) lets us also search the training data directly.
    - The writer turns the statement into 3 questions, which a teammate reads to make sure none gives the fix away: **real paper** (journal + month + the problem, "Do you know this study?"), **made-up paper** (same, for a study that doesn't exist: catches models that claim to know anything), **problem alone** (no paper named, "How would you solve it?").
    - Each model answers each question 3 times, thinking off. The judge marks every answer: claims to know the study / uses the real fix / gives details that only the paper has (not in the question).
    - Verdict per model: ✗ **knows it** (paper-only details in most tries, and more often than with no paper named) · ? **check by hand** (that once, or the fix only when the paper is named) · ✓ **makes it up** · ✓ **worked it out** (not memory, but the problem may be easy) · ✓ **doesn't know**.
5. **Difficulty filter** (short-clue stage), run on the test models, **10 tries** per check (decided 11 Nov 2026). With 5, a problem a model really solves 40% of the time still passes "≤ 1 of 5" about 1 time in 3. **12 Nov:** to fit Groq's free daily allowance, runs start with 5 tries per model (`--tries 5`, pooled per group: 10 answers for the 2 local strong models, 20 for middle). A group within about 10 points of a limit is topped up to 10 (`--tries 10`, the first 5 are kept). At 5 tries pooled, a problem the strong group really solves 40% of the time alone still passes about 2 times in 5 (1 in 4 at 10).
    - Problem alone → reaches the gold idea in **≤ 30%** of tries.
    - Problem + short clue + "use this" → reaches it in **≥ 80%** of tries.
    - Judged per **strength group**, from the quick check (`group` in `config/models.yaml`): strong = Qwen3.5 4B, Gemma 4 12B, Qwen3.5 27B, Gemma 4 31B; middle = Qwen3.5 9B, Qwen3 14B, Llama 3.1 8B, OLMo Think; weak = OLMo Instruct, Llama 3.2 3B. Look at the groups again after the first statements.
    - **Thinking off** first. Rerun only the items that pass with thinking on.
    - "Reaches the gold idea" is decided by the same judge as §5, so the judge must be built before this step.
    - **Built** (`src/generate/difficulty_filter.py CHEM-1`, 12 Nov): every local test model answers the newest kept problem alone and with the clue ("This short story may help you solve it … Use the story"), 10 tries each, thinking off. The judge (gpt-oss-120b, `prompts/solve_judge_system.txt`) marks each answer against the principle and the expected answer, without seeing the question, so it can't tell whether the clue was given. Tries are pooled per strength group, and each group gets its own verdict. The clue is required (≥ 80%) for the strong and middle groups; the weak group's clue result is only reported (decided 12 Nov). It prints an estimate and asks before starting, saves every answer and mark next to the problem it tested, and carries on when rerun (Groq's free plan allows ~1 statement of judging a day: ~160 marks × ~1,300 tokens). Its judge prompt is new and still needs the hand-check in §5.
    - **Judge marks the parts (13 Nov).** On CHEM-1's first run the judge was too generous: it said "uses the idea" when B was removed or destroyed instead of going back to how it was, or when only A's lock-in was there (about 12 of 80 marks clearly wrong, all too generous, in one reading by Claude). It now gets the idea's parts (saved with the clue) and marks each part true/false; "uses the idea" needs every part as the main fix, with the same kind of change and the same actor. A reply that says "uses it" while a part is missing is refused and asked again. `--rejudge` marks the last run's saved answers again into a new folder (old marks kept, no Ollama needed). The §5 scoring judge should work the same way.
    - **Per-group problems** (the 6 Nov plan: one clue, a problem per group): a group answers its own version if one was kept (`kept_<group>.json` next to `kept.json`), else the shared one. Plan (12 Nov): start with one shared problem; make a group its own version only when it fails (harder if too easy alone, easier if the clue doesn't help), and retest only that group (`--groups`). Writing those versions is not built yet.
6. **Write the stories** (only for items that passed step 5; details to come):
    - **Clue story:** the short clue turned into a plain narrative, with the idea hidden in the events. The solution happens as an event, not as a stated lesson.
    - **Lure:** the *same* domain and words as the problem, but a different idea.
    - **Fillers:** 3, from other domains. The real-clue set uses 2; the no-clue set uses all 3, with the third taking the clue story's place (§2).
7. **Automatic filters:**
    - Low word/meaning overlap between problem and clue story.
    - High overlap between problem and lure.
    - The clue story must not mirror the problem's sentence order or length.
8. **Human check** (two teammates, ~3 min per item):
    - The problem and gold idea match the source.
    - The clue story shares the idea.
    - The lure is same-topic but has the wrong idea.
    - The problem is fair.

The dataset pipeline ends here: no test model sees the stories during dataset generation. How well models use the clue story is measured in the experiment itself (Part 4, `solved_after_hint`).

Writer and judge model: `gpt-oss-120b` (the writer may change, see §10). The writer does not need to be newer than the test models, because the statement already gives it everything it needs. It must write well and must **not** come from a tested family (Llama, Qwen, Gemma, OLMo). No gpt-oss model is in the test set. Keep it that way.

Generate 3–5× more candidates than needed (target: 60–100 kept items). Filter runs are **not data**; only fresh runs on the frozen set count. After freezing, `dataset_v1` is read-only and any change creates `dataset_v2`. Keep ~30% of items as a private split, never posted online.

### 3.3 Item schema (one JSON object per problem)

```json
{
  "id": "P017",
  "domain": "Cybersecurity & Software Supply Chains",
  "source": {"title": "...", "url": "...", "published": "2026-08-14"},
  "statement": "...",
  "principle": "TRIZ: ...",
  "problem": "...",
  "gold_idea": "...",
  "short_clue": "...",
  "gold_mapping": {"problem_part": "story_part"},
  "blocked_answers": ["...", "..."],
  "stories": {
    "clue": {"text": "...", "domain": "everyday life"},
    "lure": {"text": "...", "domain": "Cybersecurity & Software Supply Chains"},
    "fillers": [{"text": "...", "domain": "..."}, {"text": "...", "domain": "..."}, {"text": "...", "domain": "..."}]
  },
  "fresh_note": "You tried ... they failed because ...",
  "recall_probe": {"model_id": "knew_it: true|false"},
  "difficulty_filter": {"model_id": {"alone": "1/10", "with_clue": "9/10"}},
  "split": "public | private",
  "dataset_version": "dataset_v1"
}
```

`statement` and `gold_idea` are the team's hand-written statement and one-line principle (§3.2 step 2); `principle` is the closest TRIZ tag.

---

## 4. Models and settings

Final set: 8 test models + 1 audit model. These are the team's figures; **verify release dates, cutoffs and thinking modes against each model card** before the paper.

| Tier | Model | Family | Size | Released | Stated cutoff | Thinking mode |
|---|---|---|---|---|---|---|
| Very small | Llama-3.2-3B-Instruct | Meta Llama | 3B | Sep 2024 | Dec 2023 | No |
| Very small | Qwen3.5-4B | Alibaba Qwen | 4B | Mar 2026 | Not published | On/off |
| Small | Llama-3.1-8B-Instruct | Meta Llama | 8B | Jul 2024 | Dec 2023 | No |
| Small | Qwen3.5-9B | Alibaba Qwen | 9B | Mar 2026 | Not published | On/off |
| Medium | Gemma-4-12B | Google Gemma | 12B | Jun 2026 | Jan 2025 | On/off (works on Ollama, tested 11 Nov) |
| Medium | Qwen3-14B | Alibaba Qwen | 14B | Apr 2025 | Not published | On/off |
| Large | Qwen3.5-27B | Alibaba Qwen | 27B | Feb 2026 | Not published | On/off |
| Large | Gemma-4-31B | Google Gemma | 31B | Apr 2026 | Jan 2025 | To check |
| Audit | OLMo-3-7B (Instruct + Think) | Ai2 OLMo | 7B | Nov 2025 | Dec 2024 | Separate Think model |

- **Audit model role:** OLMo's training data is open, so it is our contamination check, and Instruct vs Think is a clean reasoning comparison on the same base.
- **Thinking mode** is a factor: compare thinking off vs on for the Qwen models and Instruct vs Think for OLMo. Record the mode in every chat record.
- Exact model IDs and settings live in `config/models.yaml`. Don't hard-code them.
- Temperature 0.7 for all models. Repeats per chat are set with the scale below. Log seed, model ID, version string, thinking mode and date for every run.
- **Scale: X chats in total, not fixed yet.** X = problems × versions × repeats × model/mode combinations, and not every block needs every model. Set X after the pilot, with a quick power check that uses the pilot's real hit rates.
    - Never cut the number of problems or the 0 / 2 / 8 tries levels: they carry the main claims.
    - Cut instead: fewer repeats, the long-chat controls (versions 7–10) on a few representative models, thinking-off on a few Qwen models.
    - For scale: the full grid (every model and mode, all 10 versions, 5 repeats) is ≈ 42,000 chats; a trimmed design (all 10 model configs × versions 1–6 × 60 problems × 3 repeats, plus controls on 3 models and thinking-off on 2 Qwen models) is ≈ 13,000.
- **Where models run (proposed, not final):**
    - Lab PC (RTX A5000 24 GB, 128 GB RAM, Xeon W-2295, Windows) runs every model up to 14B: full precision (BF16) up to 9B, 8-bit for Gemma-4-12B and Qwen3-14B. Served with our own Ollama (port 11500, models in E:\LLMxPsychology-models), which runs llama.cpp inside. Test on 11 Nov 2026 (`src/setup/test_models.py`): all 8 fit fully on the GPU with room for a 32k-token chat, at 34-94 tokens/s. The thinking switch works for Qwen3.5 4B/9B, Qwen3 14B and Gemma 4 12B; OLMo Think always thinks. Qwen3.5 thinks at length, so thinking-on answers need a big token limit.
    - Qwen3.5-27B and Gemma-4-31B don't fit at full or 8-bit precision on 24 GB, so they run through an API at full precision, from one pinned provider with automatic fallback off.
    - Each model stays on the same route for the difficulty filter and the main run. Run one model both locally and through the API on the same ~50 prompts once, to show the route doesn't change results.
    - Log provider, precision and the returned model name for every answer.
    - The Nov 2026 quick check (the Replicate app) used 4-bit Ollama builds for every model. It was only an integrity check of the statements, not this setup.
- Before any large run, print estimated tokens, time and cost, and wait for confirmation.
- Always do a **dry run** first: 3 problems × 10 versions × 1 repeat × 1 model = 30 chats, which the team reads by hand.

### Chat record (one JSON file per chat)

`model, model_version, thinking_mode, problem_id, version (1–10), repeat, seed, temperature, story_order, messages[], tokens_in, tokens_out, cost, timestamp`

---

## 5. Scoring (after all chats are done)

A judge model (`gpt-oss-120b`, a different family from every tested model) reads each saved chat and answers:

1. `drop`: did any Part-1 answer already use the gold idea?
2. `solved_before_hint`: does the Part-3 answer use the gold idea? (judge it against `gold_idea`, not "is it correct")
3. `story_credited`: which story did Part 3 say helped? `clue | lure | filler | none`
4. `solved_after_hint`: does the Part-4 answer use the gold idea?

Show the judge only what it needs (the answer, `gold_idea`, story labels), and hide which version the chat came from. For thinking-mode chats, score the final answer, not the hidden reasoning.

**Validation:** two teammates hand-score the same 200 random chats blind. If judge–human agreement is < 85%, fix the judge prompt and rescore everything. Report agreement in the paper.

All scores go into one table: `results/scores.csv`, one row per chat.

---

## 6. What we calculate

- **Hit rate** (real-clue chats): % where the Part-3 answer used the clue's idea.
- **False-alarm rate** (no-clue chats): % where the model claimed a story helped.
- **Lure errors** (real-clue chats): % crediting the lure. Report separately.
- **Spotting score** d′ = Z(H) − Z(F): telling real clue vs no clue apart.
- **Eagerness score** c = −(Z(H) + Z(F)) / 2: lower c = more eager to connect.
    - Use the log-linear correction: H = (hits + 0.5)/(n + 1), F = (FA + 0.5)/(n + 1).
    - Compute per model × thinking mode × condition pair: (1,2), (3,4), (5,6), (7,8), (9,10).
- **Hint gap:** solved after hint − solved before hint ("could use it, didn't notice it").
- **Key comparisons:**
    - 0 vs 2 vs 8 failed tries.
    - 8 tries vs same-length chat vs fresh note.
    - Before vs after hint.
    - Thinking off vs on.
    - Very small vs small vs medium vs large.
    - By domain.
- **Statistics:** 95% confidence intervals (Wilson for proportions, bootstrap over problems for d′/c). One mixed-effects logistic regression per model, e.g. `outcome ~ stuck_level * clue_present + (1 | problem_id)`. Report n everywhere.

---

## 7. Output conventions (graphs, tables, files)

When asked for results or analysis, by default produce: **a figure + a CSV table + a 3–5 line plain-English summary of what it means for RQ1–RQ3.**

The main figures for the paper:

1. Hit rate and false-alarm rate vs failed tries (0/2/8), 95% CI, one panel per model.
2. Spotting (d′) and eagerness (c) vs failed tries, by model size and thinking mode.
3. 8 tries vs same-length chat vs fresh note (hits and false alarms).
4. Before vs after hint (hint gap) per condition.
5. Table: human-effect scorecard (shown / not shown / opposite).

Style:
- matplotlib, error bars on everything, plain-word axis labels ("Failed tries before the break", not `stuck_level`).
- Colours match the slides: real clue / hits `#FFB020` (orange), false alarms `#FF5A52` (red), controls grey `#8E8E93`.
- Save every figure as PNG (300 dpi) and PDF in `results/figures/`.

### Suggested repo layout

```
config/        models.yaml, run.yaml (temperature, repeats, seeds)
data/          sources/ (findings + metadata), candidates/, dataset_v1/ (frozen), private/ (git-ignored)
src/generate/  extraction + writer prompts + filters + recall probe
src/run/       chat runner (resumable, logs everything)
src/score/     judge prompts + scorer
src/analysis/  metrics, stats, plots
runs/          raw chat JSONs (never edited)
results/       scores.csv, tables/, figures/
paper/         drafts, bib
```

---

## 8. Human findings we compare against (RQ3)

| Human effect | Source | Our test |
|---|---|---|
| Distant clue rarely noticed alone (~30%), noticed with hint (~75–92%) | Gick & Holyoak 1980 | before vs after hint |
| Surface similarity drives recall; structure drives judgment | Gentner, Rattermann & Forbus 1993 | lure credit vs clue use |
| Being stuck at an impasse helps pick up clues | Moss, Kotovsky & Cagan 2011; Seifert et al. 1995 | 0 vs 2 vs 8 tries |
| Hidden clues unused unless told to look | Dodds, Smith & Ward 2002; Smith et al. 2012 | hint gap |
| Longer preparation, bigger incubation benefit | Sio & Ormerod 2009 | 2 vs 8 tries |
| Breaks help by letting the wrong idea fade | Smith & Blankenship 1991 | fresh note (9) vs full chat (5) |
| "Aha" feelings often come with wrong answers | Danek & Wiley 2017 | insight words vs correctness |

Key LLM papers already reviewed: Webb, Holyoak & Lu 2023 (*Nature Human Behaviour*); Lewis & Mitchell 2025 (TMLR); Webb et al. 2025 (*PNAS Nexus*); Chen et al. 2025; Li et al. 2024 (COLM); Laban et al. 2025; Gema et al. 2025 (TMLR); Gonen et al. 2025 (NAACL); Sen, Workiewicz & Puranam 2026 (*Strategy Science*).

---

## 9. Working style for Claude Code

- Python by default. Small, readable scripts with clear names; configs in YAML; API keys only from environment variables.
- **Explain things to us in simple terms.** The team dislikes jargon. Code comments and the paper can use proper terms; chat explanations should be plain.
- Before changing anything in the experiment design, the frozen dataset or the scoring rules, stop and ask.
- Flag any request that would break a rule in §2–§5 (cueing, solution leaked into a problem, sources older than the cutoff, unlogged randomness, judge from a tested family, overwriting raw data).
- Prefer reproducibility over speed: fixed seeds, versioned data, every run logged.
- In paper text, describe behaviour, not minds: "the model's answers used the clue more often", not "the model had an insight".

---

## 10. Open decisions (to settle as a team)

- Do the Gemma-4 models have a thinking mode we can switch? Gemma 4 12B: yes, the switch works on Ollama (tested 11 Nov). Gemma 4 31B (API): still to check.
- **Total chats (X) and repeats:** set after the pilot with a power check (§4). Which models get the long-chat controls (proposal: one per size group), and which Qwen models get thinking-off (proposal: 2)?
- **Where models run:** confirm the lab PC + API split in §4, and pick the API provider after checking which ones serve Qwen3.5-27B and Gemma-4-31B at full precision.
- Clue stories: always everyday life, or sometimes one of the other 7 domains? This changes how "far" the clue is.
- Final count per domain (target 8–12).
- ~~**Difficulty-filter numbers**~~ Decided 11 Nov: alone ≤ 30%, with clue ≥ 80%, 10 tries, per strength group (§3.2 step 5).
- **Same items for every model, or per model?** If an item must pass on all 8 models, the very small models (3B/4B) will remove most items. If each model keeps its own items, models are tested on different problems and sizes can't be compared fairly. Proposal: one shared item set; "hard alone" checked on every model, "works with the clue" checked on the medium and large models only; the very small models' clue use is just reported. **12 Nov:** one shared statement, clue and (to start with) problem; the filter judges each strength group on its own; a group gets its own problem version only if it fails; the weak group's clue result is only reported. Still open: whether size comparisons may use different problem versions, or only the items that pass with the shared problem.
- **Thinking mode in the filter:** must "hard alone" also hold with thinking on? Thinking-on solves more alone, so requiring it removes more items. For now: filter with thinking off, then rerun the passing items with thinking on and decide.
- **Items whose clue story doesn't work:** if an item's `solved_after_hint` is very low across models, keep it and report it, or drop it? Write the rule down before the main run; deciding after seeing the results would look like cherry-picking.
- **Writer model:** keep `gpt-oss-120b` or switch to a stronger writer such as Claude (via the API, not the CLI, so the system prompt, settings and exact model ID are logged)? Decide with a blind side-by-side on 5–10 statements. The judge stays `gpt-oss-120b` unless decided otherwise, since open weights let reviewers rerun the scoring.

---

## 11. Team, timeline, and later work

- **Team (BMS College of Engineering, Bengaluru):**
    - Utkarsh: architecture and flow
    - Samrudh: builder
    - Kartikey: implementation
    - Aarushi: research and details
    - Zaid, Aakanshya: execution
- Previous paper (ICTCS): correct-to-incorrect answer degradation during extended reasoning in small LMs.
- **Timeline:** Nov build + pilot → Dec main run → Feb–Mar full paper for a top venue (EMNLP / NeurIPS Datasets & Benchmarks or similar). No short CogSci-style paper: don't cut scope to fit one. → Summer: Paper 2.
- **Paper 2 (don't build unless asked):** "Prepared-Mind Agent". It writes down *why* each attempt failed, checks new text for a match in two steps (quick similarity, then a careful check that the parts really line up), and only reports a connection when confident. It succeeds only if it catches more real clues **without** more false alarms.
