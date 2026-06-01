"""Batch pipeline: run extraction over a corpus of passages idempotently.

This module owns three concerns that belong together:
    1. Fingerprinting — each (model, prompt, passage) triple maps to a unique
       hash so we can skip work that was already done.
    2. Idempotency — reading back the output file to find already-processed
       passages, so an interrupted run resumes from where it stopped.
    3. Record assembly — wrapping the model's BulletinSignal with the provenance
       fields (model_id, prompt_version, schema_version, extracted_at) that the
       code stamps on. The model never produces provenance; the pipeline does.

The model and its sampling parameters come from config.yaml.

Usage:
    from pathlib import Path
    from pipeline import run_corpus

    run_corpus(passages, Path("data/extracted/signals.jsonl"))

Each item in `passages` must be a dict with these keys:
    source_file               — filename of the source PDF
    bulletin_publication_date — ISO date string, e.g. "2024-05-10"
    epi_week_reported         — e.g. "2024-SE18"
    text                      — the passage text to extract from
"""

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

from client import build_client
from config import get_settings
from extract import extract_signal
from prompts import load_active_prompt
from schema import SCHEMA_VERSION, ExtractionRecord


def run_corpus(passages: list[dict], output_path: Path) -> None:
    """Extract signals for every passage, writing validated records as JSONL.

    The output file is appended to, not overwritten. Passages whose fingerprint
    is already present in the file are skipped, so the function is safe to
    call multiple times on the same corpus -- it will only do new work.

    Args:
        passages:    List of passage dicts (see module docstring for required keys).
        output_path: Path to the JSONL file where records are appended.
    """
    llm = get_settings().llm
    system_prompt, prompt_version = load_active_prompt()
    client = build_client()

    already_done = _load_done_fingerprints(output_path, llm.model, prompt_version)

    with output_path.open("a", encoding="utf-8") as sink:
        for item in passages:
            fingerprint = _passage_fingerprint(item["text"], llm.model, prompt_version)

            if fingerprint in already_done:
                continue  # resume: this passage was already extracted

            signal = extract_signal(
                client,
                llm.model,
                item["text"],
                system_prompt=system_prompt,
                temperature=llm.temperature,
                top_p=llm.top_p,
                max_tokens=llm.max_tokens,
                max_retries=llm.max_retries,
            )

            # The parser writes a null date when a cover yields no parseable one
            # (the corpus is unfiltered, so some bulletins simply don't have it).
            raw_date = item.get("bulletin_publication_date")
            publication_date = date.fromisoformat(raw_date) if raw_date else None

            record = ExtractionRecord(
                source_file=item["source_file"],
                bulletin_publication_date=publication_date,
                epi_week_reported=item.get("epi_week_reported"),
                signal=signal,
                model_id=llm.model,
                prompt_version=prompt_version,
                schema_version=SCHEMA_VERSION,
                extracted_at=datetime.now(timezone.utc),
            )

            # Write immediately and flush so a crash does not lose the record.
            sink.write(record.model_dump_json() + "\n")
            sink.flush()


def _passage_fingerprint(passage: str, model: str, prompt_version: str) -> str:
    """Return a short hash that uniquely identifies one unit of work.

    The fingerprint covers the passage text, the model, and the prompt version.
    This means:
    - The same passage reprocessed with a different model gets a new fingerprint
      (because model outputs differ).
    - A prompt version bump also invalidates the fingerprint (because the
      extraction instructions changed), forcing a re-run.

    Only the first 16 hex characters are kept (64 bits of collision resistance),
    which is more than enough for a corpus of thousands of passages.
    """
    content = f"{model}|{prompt_version}|{passage}"
    digest = hashlib.sha256(content.encode()).hexdigest()
    return digest[:16]


def _load_done_fingerprints(output_path: Path, model: str, prompt_version: str) -> set[str]:
    """Return the set of fingerprints already written to the output file.

    Called once at the start of run_corpus so we can skip finished work.
    Returns an empty set if the file does not exist yet.

    Note: records written before the _passage_text field was introduced will
    have an empty passage_text and will not match any fingerprint, so they
    will be re-extracted if the corpus is re-run. That is intentional: a
    record without a fingerprint cannot be deduped safely.
    """
    if not output_path.exists():
        return set()

    done: set[str] = set()
    with output_path.open(encoding="utf-8") as fh:
        lines = [l for l in fh if l.strip()]
    for line in lines:
        record = json.loads(line)
        passage_text = record.get("_passage_text", "")
        if passage_text:
            done.add(_passage_fingerprint(passage_text, model, prompt_version))
    return done
