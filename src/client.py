"""Build the LLM client used throughout the extraction pipeline.

The client is backend-agnostic: the same code runs against a local vLLM
server, Maritaca, or OpenAI -- only the LLM_BASE_URL and LLM_MODEL env vars
change. This separation means extract.py never imports OpenAI or Instructor
directly; it only knows about the client interface.

Required environment variables:
    LLM_BASE_URL  — OpenAI-compatible endpoint, e.g. http://localhost:8000/v1
    LLM_API_KEY   — API key; use any non-empty string for local vLLM
    LLM_MODEL     — model identifier, e.g. Qwen/Qwen3-8B

Instructor wraps the raw OpenAI client and adds two things:
    1. Automatic validation: the response is parsed against a Pydantic model.
    2. Automatic retries: if parsing fails, Instructor re-prompts with the
       validation errors attached so the model can self-correct.
"""

import os

import instructor
from openai import OpenAI


def build_client() -> OpenAI:
    """Return an Instructor-patched OpenAI client ready for structured extraction.

    The returned client behaves like a normal OpenAI client except that
    client.chat.completions.create(..., response_model=SomePydanticClass)
    returns a validated instance of that class instead of raw JSON.

    instructor.patch() is the 0.x API (Python 3.9 compatible).
    The 1.x API (instructor.from_openai) requires Python 3.10+.
    """
    raw_client = OpenAI(
        base_url=os.environ["LLM_BASE_URL"],
        # Local vLLM/Ollama does not require a real key; the env var still must exist.
        api_key=os.environ.get("LLM_API_KEY", "not-needed"),
    )
    return instructor.patch(raw_client)
