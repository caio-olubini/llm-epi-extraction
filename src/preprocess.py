"""Optional passage-selection stage: reduce a bulletin to its relevant spans.

This runs *before* extraction when `preprocess.enabled` is true in config.yaml.
A dedicated model copies out, verbatim, only the spans that carry arbovirus
signal (alerts, trend, serotypes, regions of concern, forward-looking warnings);
the extraction model then reads those spans instead of the whole document.

The model selects, it never transforms. To keep that guarantee enforceable
rather than merely requested, `verbatim_filter` drops any returned span that is
not actually present in the source -- so a paraphrased or invented span can never
reach the extractor. An empty selection is a valid answer (off-topic bulletin)
and lets the pipeline short-circuit without a second model call.

Like extract.py, this module hardcodes nothing about files, corpora, or config:
the model, sampling parameters, and prompt are passed in by the pipeline.
"""

import re

import instructor
from pydantic import BaseModel, Field

from schema import (
    ActionIntensity,
    AlertLevel,
    BulletinSignal,
    GeographicScope,
    ReportedTrend,
)

# Collapse any run of whitespace (spaces, newlines, tabs from PDF extraction)
# to a single space so a span and its source compare on content, not layout.
_WHITESPACE = re.compile(r"\s+")


class RelevantPassages(BaseModel):
    """The selector model's output: verbatim spans relevant to arboviruses.

    Internal intermediate contract -- it never leaves the pipeline, so it lives
    here rather than in schema.py (which owns the extraction/output contract).
    """

    passages: list[str] = Field(
        default_factory=list,
        description=(
            "Trechos copiados LITERALMENTE do boletim que sejam relevantes para "
            "arboviroses (alertas, tendencia, sorotipos, regioes de preocupacao, "
            "avisos sobre semanas/temporada futuras). Lista vazia se nada for "
            "relevante. Nao reescreva, resuma nem traduza."
        ),
    )


def select_passages(
    client: instructor.Instructor,
    model: str,
    text: str,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_retries: int,
) -> list[str]:
    """Return the spans the selector model judges relevant (still unverified).

    Constrained decoding (Mode.JSON_SCHEMA, configured in client.py) guarantees
    the list-of-strings structure; it does NOT guarantee the strings are verbatim
    -- run the result through `verbatim_filter` before trusting it.
    """
    result = client.chat.completions.create(
        model=model,
        response_model=RelevantPassages,
        max_retries=max_retries,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
    )
    return result.passages


def verbatim_filter(spans: list[str], source: str) -> list[str]:
    """Keep only spans that genuinely occur in the source (verbatim guard).

    Both sides are whitespace-normalized first, so PDF line breaks inside a span
    don't cause a false drop. A span whose normalized form is not a contiguous
    substring of the normalized source was paraphrased or invented by the model
    and is discarded -- this is what enforces "the model must not touch the text".
    Empty or whitespace-only spans are dropped too.
    """
    normalized_source = _WHITESPACE.sub(" ", source)
    kept: list[str] = []
    for span in spans:
        normalized_span = _WHITESPACE.sub(" ", span).strip()
        if normalized_span and normalized_span in normalized_source:
            kept.append(span.strip())
    return kept


def off_topic_signal() -> BulletinSignal:
    """The abstain signal emitted when the selector finds nothing relevant.

    Lets the pipeline record an off-topic bulletin without a second (extraction)
    model call. Mirrors the abstain values the extraction model would itself
    produce for an off-topic passage.
    """
    return BulletinSignal(
        is_arbovirus_related=False,
        geographic_scope=GeographicScope.not_applicable,
        uf=None,
        reported_trend=ReportedTrend.not_stated,
        alert_level=AlertLevel.not_stated,
        serotypes_mentioned=[],
        emerging_or_new_serotype=False,
        forward_looking_warning=False,
        regions_of_concern=[],
        recommended_action_intensity=ActionIntensity.not_stated,
        evidence_span="Nenhum trecho relevante a arboviroses identificado no pre-processamento.",
        requires_human_review=False,
    )
