"""Versioned extraction prompt.

This module is the single place that owns the prompt string and its version tag.
Both must be bumped together whenever any wording changes -- even a typo fix --
because the version is written into every output record as provenance.

The canonical human-readable version lives in prompts/extraction_v1.md.
The string below must match that file exactly.
"""

# Version tag written into every ExtractionRecord for provenance.
# Format: YYYY-MM-<topic>-v<N>
PROMPT_VERSION = "2026-05-extraction-v1"

# The system prompt is written in Portuguese to match the source corpus and
# reduce translation ambiguity when the model parses Brazilian health bulletins.
#
# Two-layer hallucination guard:
#   1. "nunca invente" — explicit instruction never to invent values.
#   2. `nao_informado` enum member — a legal abstain path so the model is never
#      forced to guess a categorical value that is absent from the text.
SYSTEM_PROMPT = (
    "Voce extrai sinais qualitativos de boletins epidemiologicos do Ministerio "
    "da Saude sobre arboviroses. Preencha o schema com base APENAS no que o texto "
    "afirma. Quando o boletim nao mencionar um campo, use o valor 'nao_informado' "
    "ou deixe a lista vazia -- nunca invente. Prefira marcar requires_human_review "
    "a chutar."
)
