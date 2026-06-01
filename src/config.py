"""Load run config from config.yaml and secrets from .env.

config.yaml holds everything that affects the result (model, sampling, prompt,
paths); .env holds only the endpoint and API key. load_dotenv() runs here on
import, so any module reading os.environ has .env loaded first.
"""

from functools import lru_cache
from pathlib import Path

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


class DataConfig(BaseModel):
    passages: Path
    signals: Path
    raw_pdfs: Path


class Settings(BaseModel):
    llm: LLMConfig
    prompt: PromptConfig
    data: DataConfig

    @property
    def prompt_path(self) -> Path:
        return _PROJECT_ROOT / self.prompt.path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(**yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")))
