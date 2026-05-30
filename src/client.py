"""Build the LLM client used throughout the extraction pipeline.

The client is backend-agnostic: the same code runs against a local vLLM
server, Ollama, Maritaca, or OpenAI -- only the LLM_BASE_URL and LLM_MODEL
env vars change. This separation means extract.py never constructs the raw
OpenAI client itself; it only receives the wrapped client.

Required environment variables:
    LLM_BASE_URL  — OpenAI-compatible endpoint, e.g. http://localhost:8000/v1
    LLM_API_KEY   — API key; any non-empty string for local vLLM/Ollama
    LLM_MODEL     — model identifier, e.g. Qwen/Qwen3-8B


Constrained decoding -- the point of this whole setup
-----------------------------------------------------
There are three ways a library can get JSON out of a model, and only one of
them gives the schema *guarantee* this thesis relies on:

  1. Tool-calling + validate + retry (instructor's DEFAULT, Mode.TOOLS).
     The model emits a tool call, instructor parses it, and re-prompts on
     failure. Any single attempt can still be malformed -- correctness comes
     from retrying, not from the decoder.

  2. Generic JSON mode (Mode.JSON). The model is told "return JSON" via
     response_format={"type": "json_object"} and the schema is pasted into the
     prompt. The output is JSON, but nothing stops the model from inventing an
     enum value or omitting a field -- the schema is a suggestion, not a
     constraint. This is NOT constrained decoding.

  3. True constrained decoding (Mode.JSON_SCHEMA -- what we use).
     The schema is sent in the OpenAI-standard structured-output envelope
     response_format={"type": "json_schema", "json_schema": {...}}. Both vLLM
     and Ollama recognise this and hand the schema to their XGrammar backend,
     which masks token logits at every decoding step so a token that would
     break the schema cannot be sampled. The first attempt is valid by
     construction; instructor's retries become a transport-error safety net.

We therefore pick Mode.JSON_SCHEMA. It is the one mode that is BOTH portable
across our two backends (Ollama now, vLLM for the paper) AND actually
constrains the decoder rather than just asking nicely.

Why not the vLLM-only `guided_json` field? Because Mode.JSON_SCHEMA already
sends the schema in the standard envelope that vLLM and Ollama both constrain
on, so a backend-specific field would be redundant on vLLM and ignored by
Ollama. One portable mechanism beats two divergent ones.

vLLM is the paper-grade path because its weights can be pinned to an exact
version; Ollama is the convenient local path. The constraint mechanism is
identical for both.
"""

import os

import instructor
from openai import OpenAI


def build_client() -> instructor.Instructor:
    """Return an Instructor client configured for constrained JSON decoding.

    The returned client behaves like a normal OpenAI client except that
    client.chat.completions.create(..., response_model=SomePydanticClass)
    returns a validated instance of that class instead of raw JSON.

    Mode.JSON_SCHEMA makes instructor send the schema in the OpenAI-standard
    response_format={"type": "json_schema", ...} envelope. Both Ollama and vLLM
    feed that schema to their XGrammar backend and constrain the decoder, so the
    output is schema-valid by construction (see the module docstring for why
    this mode and not the others).
    """
    raw_client = OpenAI(
        base_url=os.environ["LLM_BASE_URL"],
        # Local vLLM/Ollama ignores the key, but the OpenAI client requires one.
        api_key=os.environ.get("LLM_API_KEY", "not-needed"),
    )
    return instructor.from_openai(raw_client, mode=instructor.Mode.JSON_SCHEMA)
