"""Extract the qualitative signal from a single bulletin passage.

This module handles exactly one concern: turning one text passage into one
validated BulletinSignal. It does not know about files, corpora, or provenance
-- those belong in pipeline.py.

Keeping this function isolated makes it easy to test with a single passage
and to swap the model or prompt without touching the batch logic.
"""

from openai import OpenAI

from schema import BulletinSignal
from prompts import SYSTEM_PROMPT


def extract_signal(
    client: OpenAI,
    model: str,
    passage: str,
) -> BulletinSignal:
    """Extract the qualitative signal from one bulletin passage.

    Args:
        client:  Instructor-patched OpenAI client (built by client.py).
        model:   Model identifier string, e.g. "Qwen/Qwen3-8B".
        passage: Raw text of one bulletin section (one UF or national summary).

    Returns:
        A validated BulletinSignal instance.

    How validation works:
        Instructor intercepts the model response and parses it against
        BulletinSignal. If parsing fails (e.g. the model returned an invalid
        enum value), Instructor re-prompts with the validation error attached
        so the model can self-correct. max_retries=2 allows two such attempts
        before raising.

    Why temperature=0:
        Determinism matters more than variety here. The same passage fed to
        the same model must produce the same extraction so that re-runs are
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
