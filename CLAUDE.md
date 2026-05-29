# CLAUDE.md — dengue-bulletin-extraction

This file is read by Claude Code at the start of every session. It contains
everything needed to work in this codebase without asking for background.

---

## What this project is

Part of an MSc thesis (Statistics & Data Science, UFBA) building a multimodal
neural forecasting system for dengue outbreaks across Brazilian states.

The other modalities — SINAN/SIVEP case counts, climate variables, Google
Trends, EBC news — are already scraped. This module handles the one that
isn't covered in the literature: **turning the Ministério da Saúde weekly
arbovirus bulletins (PDFs) into a structured, temporally-aligned feature
matrix** that feeds the final forecasting model alongside the numeric channels.

The research contribution is specifically that these bulletins carry qualitative
signal (risk framing, serotype intel, forward-looking alerts) that is NOT in the
case-count series. This module exists to make that signal machine-readable.

---

## Critical domain constraint — temporal leakage

**This is the most important rule in the codebase. Never violate it.**

The bulletins often restate the very case counts we are predicting. If you
extract a "number of cases this week" from a bulletin and use it as a model
feature, you have leaked the target. The entire value of this modality comes
from the fields that carry information *beyond* the contemporaneous count:

- Serotype mentions (especially re-emerging or new serotypes in a region)
- The narrative trend the bulletin *describes* (not numbers — prose framing)
- Official alert/risk level assigned to a region
- Forward-looking warnings about upcoming weeks or the coming season
- Recommended action intensity

**Any time you add or modify extraction fields, ask: does this field contain
information that was genuinely available before the forecast target was
realised? If no, do not extract it.**

The `epi_week_reported` field is the week the bulletin *reports on*, not the
publication date. Always check and store both, because bulletins publish with a
lag. The forecasting model must only see features from weeks strictly before the
prediction horizon.

---

## Repo structure

```
dengue-bulletin-extraction/
│
├── CLAUDE.md               ← you are here
├── pyproject.toml          ← deps, tool config
├── .env.example            ← required env vars, no secrets
│
├── data/
│   ├── raw/                ← PDFs exactly as downloaded, never modified
│   ├── passages/           ← parsed text units as JSONL (source_file, pub_date,
│   │                          epi_week, text) — output of src/parse.py
│   └── extracted/          ← validated ExtractionRecord JSONL — output of pipeline.py
│
├── src/
│   ├── schema.py           ← Pydantic models: BulletinSignal + ExtractionRecord (the spec)
│   ├── prompts.py          ← SYSTEM_PROMPT string and PROMPT_VERSION constant
│   ├── client.py           ← builds the Instructor-patched OpenAI client
│   ├── extract.py          ← single-passage extraction (one function, one concern)
│   ├── pipeline.py         ← batch runner: fingerprinting, idempotency, record assembly
│   ├── __main__.py         ← CLI entry point: `python src/ [--input X] [--output Y]`
│   ├── parse.py            ← PDFs → passage units  [NEXT TO BUILD]
│   └── eval.py             ← agreement metrics, Cohen's κ vs oracle [FUTURE]
│
├── prompts/
│   └── extraction_v1.md    ← versioned extraction prompt (source of truth for SYSTEM_PROMPT
│                              in src/prompts.py — keep in sync, bump version on any change)
│
├── notebooks/
│   └── explore.ipynb       ← EDA only, never runs pipeline code
│
└── tests/
    └── test_schema.py      ← schema validation, sentinel values, enum coverage
```

### Module responsibilities (one concern per file)

| File | Owns |
|---|---|
| `schema.py` | Pydantic data contracts — `BulletinSignal`, `ExtractionRecord`, all enums |
| `prompts.py` | The system prompt string and its version tag |
| `client.py` | Building the Instructor-patched LLM client from env vars |
| `extract.py` | Calling the LLM for one passage → one `BulletinSignal` |
| `pipeline.py` | Batch loop, fingerprinting, idempotency, provenance stamping |
| `__main__.py` | CLI argument parsing and demo mode |

---

## Setup

```bash
# 1. Clone and enter
git clone <repo-url> && cd dengue-bulletin-extraction

# 2. Create environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install instructor openai pydantic pymupdf python-dotenv

# 4. Copy env template and fill in
cp .env.example .env
```

`.env.example`:
```
# Local vLLM (reproducible, paper-grade path)
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=not-needed-for-local
LLM_MODEL=Qwen/Qwen3-8B

# Frontier oracle (label bootstrapping and ceiling benchmark only)
# ORACLE_BASE_URL=https://chat.maritaca.ai/api
# ORACLE_API_KEY=your-maritaca-key
# ORACLE_MODEL=sabia-4
```

**To serve Qwen3-8B locally with constrained decoding:**
```bash
pip install vllm
vllm serve Qwen/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes
```
Or with Ollama (simpler, slower):
```bash
ollama pull qwen3:8b
# set LLM_BASE_URL=http://localhost:11434/v1
```

---

## Model strategy (two-tier, do not collapse this into one)

**Workhorse — local open model (Qwen3-4B or 8B, Apache 2.0)**
- Runs locally, weights pinned to a specific version
- Every paper result must be reproducible from these weights
- Served via vLLM with XGrammar constrained decoding for schema guarantees

**Oracle — frontier API (Sabiá-4 or equivalent)**
- Used *only* to bootstrap a small gold/silver validation set (~50–100 records)
- Used to compute ceiling agreement (how close does the local model get?)
- Results cited as "agreement with frontier oracle: κ = X.XX"
- Never used as the production extractor — it cannot be pinned or reproduced

This design is what makes the pipeline academically defensible. Do not merge
the two roles into one model. The distinction must be preserved in code
(separate config keys, logged separately in provenance).

---

## Coding conventions

This codebase follows a strict philosophy: **code is a communication tool**.

- **Names reveal intent.** `bulletin_publication_date` not `pub_dt`. `extract_signal` not `run`.
- **Comments explain *why*, never *what*.** If a comment describes what the code does,
  rewrite the code until it's obvious, then delete the comment.
- **Functions do one thing.** If you can describe a function with "and", split it.
- **No abstraction before it is reused.** Do not create base classes, factories, or
  plugin systems for hypothetical future requirements. Add them when the second
  use case appears.
- **Every record carries full provenance.** `model_id`, `prompt_version`,
  `schema_version`, `extracted_at` are stamped in code, never by the model.
  A record without provenance is not a valid record.

---

## Schema rules (src/schema.py)

The schema is the contract between the extraction step and the forecasting model.
Treat it as carefully as a database migration.

- **Enums over free strings for every categorical field.** The model must not
  be able to invent values that downstream code cannot handle.
- **Every ordinal enum must include a `nao_informado` member.** This gives the
  model a valid abstain path and is the primary hallucination reducer.
- **`requires_human_review: bool` must always be present.** When True, the record
  is written to a separate review queue, not directly to the features table.
- **`evidence_span: str` (max 280 chars) is mandatory.** Auditability of every
  extraction is non-negotiable for a thesis.
- **Schema version is a constant in schema.py, not in pyproject.toml.** Bump it
  on any field addition, removal, or type change. The version is stored in every
  output record.

Before modifying the schema, check that the change is leakage-safe (see above).

---

## Prompt versioning (prompts/extraction_v1.md)

`prompts/extraction_v1.md` is the human-readable source of truth for the system
prompt. `src/prompts.py` contains the same text as a Python string.

Both must be kept in sync. Both the filename and `PROMPT_VERSION` in `prompts.py`
must be bumped together on any wording change, even a typo fix. This is required
for provenance integrity: a row in the output must be fully rerunnable by checking
out the corresponding prompt version from git.

---

## Running the pipeline

```bash
# Step 1: parse PDFs into passage units (once parse.py is built)
python src/parse.py --input data/raw/ --output data/passages/passages.jsonl

# Step 2: extract signals (idempotent, safe to re-run)
python src/ --input data/passages/passages.jsonl \
            --output data/extracted/signals.jsonl

# Step 3 (once oracle labels exist): compute agreement
python src/eval.py --extracted data/extracted/signals.jsonl \
                   --gold data/extracted/gold_sample.jsonl

# Demo mode (no passages file needed — uses built-in synthetic passage):
python src/
```

---

## What to build next

Priority order, do not skip ahead:

1. **`src/parse.py`** — PDF to passage units. Each bulletin has a "Situação
   Epidemiológica" section per disease/region; that section is the target passage.
   Use PyMuPDF (fitz) for text extraction. Output: JSONL with fields
   `{source_file, bulletin_publication_date, epi_week_reported, uf, text}`.
   One row per (bulletin × UF) combination where a UF-level section exists, plus
   one national row.

2. **`tests/test_schema.py`** — validate that every enum member roundtrips through
   JSON serialisation, that `requires_human_review` defaults to False, and that
   a missing `evidence_span` raises a validation error.

3. **`src/eval.py`** — agreement harness. Load a hand-labelled gold sample
   (50–100 records), compute per-field accuracy and Cohen's κ between the local
   model and (a) the human labels and (b) the oracle labels. This is the paper's
   Table 2.

---

## What not to do

- Do not add LangChain. It adds hidden prompts, version churn, and reproducibility
  noise. Instructor + plain Python is the right level of abstraction.
- Do not extract numeric case counts. They are already in SINAN/SIVEP and would
  leak the target (see leakage section above).
- Do not call the frontier oracle model in the main extraction loop. It is for
  label bootstrapping only.
- Do not modify `data/raw/`. It is immutable. All transformations happen in
  later stages.
- Do not commit `.env`. It is gitignored. Share config via `.env.example` only.

---

## Research context (condensed, for domain questions)

- **Thesis:** Multimodal neural forecasting of dengue outbreaks across Brazilian
  states. Integrates SINAN/SIVEP epidemiological baselines, climate variables,
  Google Trends, EBC news, and this module's bulletin signal.
- **Target model:** neural forecaster (LSTM or equivalent) trained at the
  (UF, epidemiological_week) granularity.
- **Key prior work:** Chen & Moraga 2026 (LSTM + mobility + climate, 10 cities);
  Borges et al. 2026 Scientific Data (SINAN + GT benchmark dataset). Neither
  uses official narrative text as a modality — that is the contribution gap
  this module fills.
- **Epidemiological week convention:** Brazilian SINAN uses SE (semana
  epidemiológica) numbered 1–53. The epi-week string format in this codebase
  is `YYYY-SENN` (e.g. `2024-SE18`).
- **UF codes:** 2-letter IBGE abbreviations (BA, SP, RJ, ...). The national
  aggregate row uses `uf = None`.
