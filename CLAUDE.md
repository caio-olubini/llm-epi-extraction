# CLAUDE.md — dengue-bulletin-extraction

This file is read by Claude Code at the start of every session. It contains
everything needed to work in this codebase without asking for background.

---

## What this project is

Part of an MSc thesis (Statistics & Data Science, UFBA) building a multimodal
neural forecasting system for dengue outbreaks across Brazilian states.

The other modalities — SINAN/SIVEP case counts, climate variables, Google
Trends, EBC news — are already scraped. This module handles the one that
isn't covered in the literature: **turning the Ministério da Saúde bulletin
corpus (PDFs) into a structured, temporally-aligned feature matrix** that feeds
the final forecasting model alongside the numeric channels.

The corpus is fed in **unfiltered** — all Ministério da Saúde bulletins, not
just arbovirus ones. Deciding relevance is the model's job (`is_arbovirus_related`
field), and that classification is part of the research contribution. The
qualitative signal of interest (risk framing, serotype intel, forward-looking
alerts) is NOT in the case-count series. This module exists to make it
machine-readable.

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
├── requirements.txt        ← deps (no pyproject.toml)
├── config.yaml             ← all result-affecting run config (model, sampling, prompt, paths)
├── .env.example            ← connection + secret vars, no values
│
├── data/
│   ├── raw/                ← PDFs exactly as downloaded, never modified
│   ├── passages/           ← parsed text units as JSONL — output of src/parse.py
│   └── extracted/          ← validated ExtractionRecord JSONL — output of pipeline.py
│
├── src/
│   ├── config.py           ← loads config.yaml + .env; typed Settings singleton
│   ├── schema.py           ← Pydantic models: BulletinSignal + ExtractionRecord (the spec)
│   ├── prompts.py          ← loads SYSTEM_PROMPT + PROMPT_VERSION from the active .md file
│   ├── client.py           ← builds the Instructor-patched OpenAI client
│   ├── extract.py          ← single-passage extraction (one function, one concern)
│   ├── pipeline.py         ← batch runner: fingerprinting, idempotency, record assembly
│   ├── __main__.py         ← CLI entry point: `python src/ [--input X] [--output Y]`
│   ├── parse.py            ← PDFs → passage units (one passage per bulletin)
│   └── eval.py             ← agreement metrics, Cohen's κ vs oracle [FUTURE]
│
└── prompts/
    └── extraction_v2.md    ← versioned prompt file; fenced block = system prompt sent to model;
                               first heading carries the version tag; design notes outside the
                               fence are for humans only and never reach the API
```

### Module responsibilities (one concern per file)

| File | Owns |
|---|---|
| `config.py` | Loading config.yaml and .env into a typed `Settings` singleton |
| `schema.py` | Pydantic data contracts — `BulletinSignal`, `ExtractionRecord`, all enums |
| `prompts.py` | Loading `SYSTEM_PROMPT` and `PROMPT_VERSION` from the active prompt file |
| `client.py` | Building the Instructor-patched LLM client from env vars |
| `extract.py` | Calling the LLM for one passage → one `BulletinSignal` |
| `pipeline.py` | Batch loop, fingerprinting, idempotency, provenance stamping |
| `__main__.py` | CLI argument parsing and demo mode |
| `parse.py` | PDF corpus → passages JSONL (one row per bulletin, content-agnostic) |

---

## Configuration

Everything that affects an extraction run's result lives in **`config.yaml`**
(committed, no secrets). Connection credentials live in **`.env`** (gitignored).

```yaml
# config.yaml
llm:
  model: gemini-2.5-flash   # model identifier passed to the API
  temperature: 0.0
  top_p: 1.0
  max_tokens: 2048
  max_retries: 2

prompt:
  path: prompts/extraction_v2.md

data:
  passages: data/passages/passages.jsonl
  signals: data/extracted/signals.jsonl
  raw_pdfs: data/raw
```

```
# .env  (copy from .env.example, never commit)
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_API_KEY=your-key-here
```

`config.py` calls `load_dotenv()` on import, so `.env` is loaded for the whole
process regardless of entry point (CLI or test). `src/schema.py` is NOT
configurable — `SCHEMA_VERSION` is a code constant, not a tunable.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in LLM_BASE_URL and LLM_API_KEY
```

**How constrained decoding is wired (read before touching client.py):**
The client is built with `instructor.Mode.JSON_SCHEMA`. This sends the Pydantic
schema in the `response_format={"type":"json_schema", ...}` envelope, which vLLM
and Ollama feed to their XGrammar backend to mask token logits during decoding.
The output is schema-valid by construction, not by retry. Do **not** switch to
`Mode.JSON` or `Mode.TOOLS`.

---

## Prompt versioning

The prompt file (`prompts/extraction_v2.md`) is the single source of truth:

- The **fenced code block** inside the file is the exact text sent to the model.
  Everything outside the fence (design notes, changelog) is human documentation
  and never reaches the API.
- The **first heading** carries the version tag:
  `# Extraction prompt — version 2026-05-extraction-v2`
- `src/prompts.py` reads both the text and the version from that file at import
  time. There is no second copy to sync.

**To create a new prompt version:**
1. Copy the file → `prompts/extraction_v3.md`, bump the version in the heading.
2. Edit the fenced block.
3. Update `config.yaml` → `prompt.path: prompts/extraction_v3.md`.

That's it. Every output record's `prompt_version` field then reflects the new tag.

Current version: `2026-05-extraction-v2` (schema 1.1.0).

---

## Schema rules (src/schema.py)

The schema is the contract between the extraction step and the forecasting model.
Treat it as carefully as a database migration.

- **Enums over free strings for every categorical field.**
- **Every ordinal enum must include a `nao_informado` member** — valid abstain path, primary hallucination reducer.
- **`GeographicScope` also includes `nao_se_aplica`** for off-topic bulletins.
- **`is_arbovirus_related: bool` is the first field the model fills.** When False, every other field takes its abstain value.
- **`requires_human_review: bool` must always be present.**
- **`evidence_span: str` (max 280 chars) is mandatory.** Auditability is non-negotiable.
- **`SCHEMA_VERSION` is a constant in `schema.py`.** Bump on any field addition, removal, or type change.

Before modifying the schema, check that the change is leakage-safe (see above).

---

## Running the pipeline

```bash
# Parse PDFs into passages (defaults come from config.yaml)
python src/parse.py

# Extract signals (idempotent, safe to re-run)
python src/

# Override paths via CLI flags
python src/ --input data/passages/passages.jsonl --output data/extracted/signals.jsonl

# Demo mode (no input file — uses a built-in synthetic passage)
python src/
```

**Corpus:** 202 PDFs across `data/raw/2019/` – `data/raw/2026/`. One bulletin
(`2021/boletim_hanseniase_internet_-2.pdf`) is image-only; it gets empty text
and an off-topic classification.

---

## What to build next

1. **Fix idempotency in `pipeline.py`** — `run_corpus` skips already-done passages
   by reading a `_passage_text` field back from the output file, but never writes
   that field. A re-run will reprocess everything and append duplicates. The
   fingerprint must be written into the record on first pass.

2. **`tests/test_schema.py`** — validate enum roundtrips, `requires_human_review`
   default, missing `evidence_span` raises, over-long `evidence_span` clips and
   flips the flag.

3. **`src/eval.py`** — agreement harness: per-field accuracy and Cohen's κ between
   the model and hand-labelled gold sample (50–100 records).

---

## What not to do

- Do not add LangChain.
- Do not extract numeric case counts (leaks the forecast target).
- Do not modify `data/raw/` — it is immutable.
- Do not commit `.env`.
- Do not add oracle/two-tier model config — the current single-model config is intentional.

---

## Research context

- **Thesis:** Multimodal neural forecasting of dengue outbreaks across Brazilian states.
- **Target model:** neural forecaster (LSTM or equivalent) at (UF, epi-week) granularity.
- **Key prior work:** Chen & Moraga 2026; Borges et al. 2026 Scientific Data. Neither uses official narrative text as a modality — that is the contribution gap this module fills.
- **Epi-week format:** `YYYY-SENN` (e.g. `2024-SE18`), Brazilian SINAN convention.
- **UF codes:** 2-letter IBGE abbreviations. National aggregate uses `uf = None`.
