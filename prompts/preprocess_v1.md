# Passage selection prompt — version 2026-06-preprocess-v1

This file is the **source of truth** for the system prompt used during the optional
passage-selection (preprocessing) stage. The fenced code block below is the exact
text sent to the model; everything outside the fence is human documentation.

Bump both the filename and the version tag in the first heading together whenever any
wording changes, even a typo fix. The version is stamped onto every output record's
`preprocess_prompt_version` field, so a row must be reproducible by checking out the
matching prompt from git.

---

## System prompt (Portuguese)

```
Voce recebe o texto integral de um boletim epidemiologico do Ministerio da Saude do Brasil. Sua UNICA tarefa e SELECIONAR e COPIAR, palavra por palavra, os trechos relevantes para arboviroses (dengue, chikungunya, Zika, febre amarela e similares).

Selecione trechos que tragam: niveis de alerta ou risco; tendencia narrada (alta, queda, estavel); sorotipos de dengue citados (DENV-1 a DENV-4), especialmente novos ou reemergentes; municipios, estados ou regioes destacados como preocupantes; e avisos sobre as proximas semanas ou a temporada que vem.

REGRAS ABSOLUTAS:
- COPIE os trechos exatamente como aparecem no texto. NAO reescreva, NAO resuma, NAO traduza, NAO corrija erros de digitacao, NAO una trechos distantes em um so.
- Cada item da lista deve ser uma substring literal e contigua do texto original.
- IGNORE todo conteudo que nao seja sobre arboviroses (raiva, sarampo, resistencia antimicrobiana, cabecalhos administrativos, metodologia, etc.).
- Se o boletim NAO tratar de arboviroses ou nao houver nenhum trecho relevante, retorne uma lista vazia.

Retorne apenas os trechos selecionados, sem comentarios.
```

---

## Design notes

- **Selection, not transformation.** The model's job is to reduce, never to rewrite.
  Keeping spans verbatim preserves auditability and lets the downstream extractor see
  the original prose (and lets `verbatim_filter` drop anything the model paraphrased).
- Written in Portuguese to match the source corpus and reduce translation ambiguity,
  consistent with `prompts/extraction_v2.md`.
- **Empty list is a first-class answer.** The corpus is unfiltered and mostly
  off-topic; an empty selection triggers the pipeline's off-topic short-circuit, so
  the extraction model is never called on irrelevant bulletins.

---

## Changelog

- **v1 (2026-06):** Initial passage-selection prompt for the optional preprocessing
  stage. Pairs with schema 1.2.0.
