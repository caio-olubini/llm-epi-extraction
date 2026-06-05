"""Load run config from config.yaml and secrets from .env.

config.yaml holds everything that affects the result (model, sampling, prompt,
paths); .env holds only the endpoint and API key. load_dotenv() runs here on
import, so any module reading os.environ has .env loaded first.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"

load_dotenv(_PROJECT_ROOT / ".env")


class LLMConfig(BaseModel):
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    max_retries: int


class PromptConfig(BaseModel):
    path: Path


class PreprocessConfig(BaseModel):
    """Optional passage-selection stage that runs before extraction.

    Mirrors LLMConfig (its own model + sampling, so it can use a different model
    from the extraction step) and carries its own prompt file. enabled gates the
    whole stage; when False the pipeline feeds the full passage straight to
    extraction, exactly as before.
    """
    enabled: bool = False
    model: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    max_retries: int = 2
    prompt: PromptConfig


class DataConfig(BaseModel):
    passages: Path
    signals: Path
    raw_pdfs: Path


class Settings(BaseModel):
    llm: LLMConfig
    prompt: PromptConfig
    data: DataConfig
    # Optional: an absent `preprocess` block means the stage is disabled.
    preprocess: Optional[PreprocessConfig] = None

    @property
    def prompt_path(self) -> Path:
        return _PROJECT_ROOT / self.prompt.path

    @property
    def preprocess_prompt_path(self) -> Path:
        return _PROJECT_ROOT / self.preprocess.prompt.path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(**yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")))
