"""Turn one bulletin passage into one validated BulletinSignal.

Files, corpora, and provenance belong in pipeline.py. The model, sampling
parameters, and prompt are passed in (sourced from config.yaml), so this function
hardcodes nothing.
"""

import instructor

from schema import BulletinSignal


def extract_signal(
    client: instructor.Instructor,
    model: str,
    passage: str,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_retries: int,
) -> BulletinSignal:
    """Extract the qualitative signal from one bulletin passage.

    Constrained decoding is configured once in client.py (Mode.JSON_SCHEMA);
    passing response_model here supplies the schema and validates the result.
    max_retries only covers transport errors -- structure is already guaranteed.
    """
    return client.chat.completions.create(
        model=model,
        response_model=BulletinSignal,
        max_retries=max_retries,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": passage},
        ],
    )
