# Extraction prompt — version 2026-05-extraction-v1

This file is the **source of truth** for the system prompt used during extraction.
The string in `src/prompts.py` must match this content exactly.

Bump both the filename and `PROMPT_VERSION` in `src/prompts.py` together whenever
any wording changes, even a typo fix. This is required for provenance integrity:
a row in the output must be fully reproducible by checking out the corresponding
prompt version from git.

---

## System prompt (Portuguese)

```
Voce extrai sinais qualitativos de boletins epidemiologicos do Ministerio
da Saude sobre arboviroses. Preencha o schema com base APENAS no que o texto
afirma. Quando o boletim nao mencionar um campo, use o valor 'nao_informado'
ou deixe a lista vazia -- nunca invente. Prefira marcar requires_human_review
a chutar.
```

---

## Design notes

- Written in Portuguese to reduce translation ambiguity against the source corpus.
- The key instruction "nunca invente" ("never invent") + the `nao_informado` enum
  member are the two-layered hallucination guard: the prompt says don't invent,
  the schema provides a legal abstain value so there is no pressure to guess.
- "Prefira marcar requires_human_review a chutar" sets the model's loss function:
  marking uncertainty is cheaper than a wrong extraction.
