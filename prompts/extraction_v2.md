# Extraction prompt — version 2026-05-extraction-v2

This file is the **source of truth** for the system prompt used during extraction.
The string in `src/prompts.py` must match this content exactly.

Bump both the filename and `PROMPT_VERSION` in `src/prompts.py` together whenever
any wording changes, even a typo fix. This is required for provenance integrity:
a row in the output must be fully reproducible by checking out the corresponding
prompt version from git.

---

## System prompt (Portuguese)

```
Voce analisa boletins epidemiologicos do Ministerio da Saude do Brasil. Muitos NAO tratam de arboviroses (tratam de raiva, sarampo, resistencia antimicrobiana, etc.). Primeiro decida is_arbovirus_related: marque true apenas se o boletim tratar de dengue, chikungunya, Zika, febre amarela ou arboviroses similares. Se for false, use 'nao_se_aplica' em geographic_scope, 'nao_informado' nos demais campos categoricos e listas vazias. Quando for true, preencha o schema com base APENAS no que o texto afirma; quando o texto nao mencionar um campo, use 'nao_informado' ou lista vazia -- nunca invente. Prefira marcar requires_human_review a chutar.
```

---

## Design notes

- Written in Portuguese to reduce translation ambiguity against the source corpus.
- **Classification first.** The corpus is fed in unfiltered — every Ministério da
  Saúde bulletin, not just arbovirus ones — so the model's first task is to decide
  `is_arbovirus_related`. Deciding relevance is the experiment; the parser never
  pre-filters. When the bulletin is off-topic the model abstains on every other
  field (`nao_se_aplica` / `nao_informado` / empty list) instead of inventing
  dengue signal.
- The key instruction "nunca invente" ("never invent") + the `nao_informado` enum
  member are the two-layered hallucination guard: the prompt says don't invent,
  the schema provides a legal abstain value so there is no pressure to guess.
- "Prefira marcar requires_human_review a chutar" sets the model's loss function:
  marking uncertainty is cheaper than a wrong extraction.

---

## Changelog

- **v2 (2026-05):** Added classification-first framing for the unfiltered corpus
  (`is_arbovirus_related`, `nao_se_aplica` abstain). Pairs with schema 1.1.0.
- **v1 (2026-05):** Initial arbovirus signal-extraction prompt.
