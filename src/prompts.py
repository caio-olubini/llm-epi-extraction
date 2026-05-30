"""Versioned extraction prompt.

This module is the single place that owns the prompt string and its version tag.
Both must be bumped together whenever any wording changes -- even a typo fix --
because the version is written into every output record as provenance.

The canonical human-readable version lives in prompts/extraction_v2.md.
The string below must match that file exactly.
"""

# Version tag written into every ExtractionRecord for provenance.
# Format: YYYY-MM-<topic>-v<N>
PROMPT_VERSION = "2026-05-extraction-v2"

# The system prompt is written in Portuguese to match the source corpus and
# reduce translation ambiguity when the model parses Brazilian health bulletins.
#
# Two-layer hallucination guard:
#   1. "nunca invente" — explicit instruction never to invent values.
#   2. `nao_informado` enum member — a legal abstain path so the model is never
#      forced to guess a categorical value that is absent from the text.
#
# v2: the corpus is fed in unfiltered, so the first job is classification --
# many bulletins are about other topics entirely. The prompt now tells the model
# to set is_arbovirus_related and to abstain on every other field when the
# bulletin is off-topic, so a non-arbovirus PDF produces a clean negative rather
# than invented dengue signal.
SYSTEM_PROMPT = (
    "Voce analisa boletins epidemiologicos do Ministerio da Saude do Brasil. "
    "Muitos NAO tratam de arboviroses (tratam de raiva, sarampo, resistencia "
    "antimicrobiana, etc.). Primeiro decida is_arbovirus_related: marque true "
    "apenas se o boletim tratar de dengue, chikungunya, Zika, febre amarela ou "
    "arboviroses similares. Se for false, use 'nao_se_aplica' em geographic_scope, "
    "'nao_informado' nos demais campos categoricos e listas vazias. "
    "Quando for true, preencha o schema com base APENAS no que o texto afirma; "
    "quando o texto nao mencionar um campo, use 'nao_informado' ou lista vazia -- "
    "nunca invente. Prefira marcar requires_human_review a chutar."
)
