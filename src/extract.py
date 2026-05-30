"""Extract the qualitative signal from a single bulletin passage.

This module handles exactly one concern: turning one text passage into one
validated BulletinSignal. It does not know about files, corpora, or provenance
-- those belong in pipeline.py.

Keeping this function isolated makes it easy to test with a single passage
and to swap the model or prompt without touching the batch logic.
"""

import instructor

from schema import BulletinSignal
from prompts import SYSTEM_PROMPT


def extract_signal(
    client: instructor.Instructor,
    model: str,
    passage: str,
) -> BulletinSignal:
    """Extract the qualitative signal from one bulletin passage.

    Args:
        client:  Instructor client built by client.py (Mode.JSON_SCHEMA).
        model:   Model identifier string, e.g. "Qwen/Qwen3-8B".
        passage: Raw text of one bulletin section (one UF or national summary).

    Returns:
        A validated BulletinSignal instance.

    Where constrained decoding is applied:
        Not here -- it is configured once in client.py via Mode.JSON_SCHEMA.
        That mode makes instructor send BulletinSignal's JSON Schema in the
        response_format envelope, and the server (Ollama/vLLM) masks token
        logits so the output cannot violate the schema. Passing response_model
        here is what supplies that schema and what validates the result; the
        same Pydantic class is therefore both the decoding constraint and the
        validation contract, so they can never drift apart.

    Why max_retries is still set:
        With constrained decoding the structure is guaranteed, so retries only
        cover transport hiccups or a backend that silently ignores the schema.
        It is a safety net, not the primary correctness mechanism.

    Why temperature=0:
        Determinism matters more than variety here. The same passage fed to the
        same pinned model must produce the same extraction, so re-runs are
        idempotent and paper results are reproducible.
    """
    return client.chat.completions.create(
        model=model,
        response_model=BulletinSignal,
        max_retries=2,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": passage},
        ],
    )
