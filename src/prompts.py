"""Load the active extraction prompt named in config.yaml.

The prompt text now lives in exactly one place: the fenced code block of the
prompt file (prompts/extraction_vN.md). PROMPT_VERSION is parsed from the version
tag in that file's first heading. This removes the old failure mode where the
string in this module and the .md file could silently drift apart -- there is
only one copy now, and the version travels with it for provenance.

SYSTEM_PROMPT and PROMPT_VERSION are kept as module-level names so existing
importers (extract.py, pipeline.py) need no change.
"""

import re
from pathlib import Path

from config import get_settings

# Fenced code block holding the verbatim prompt text. The language tag is optional.
_FENCE_PATTERN = re.compile(r"```[\w-]*\n(.*?)```", re.DOTALL)

# "version 2026-05-extraction-v2" inside the file's first heading.
_VERSION_PATTERN = re.compile(r"version\s+(\S+)", re.IGNORECASE)


def load_active_prompt(prompt_path: Path | None = None) -> tuple[str, str]:
    """Return (system_prompt_text, prompt_version) from a prompt file.

    Defaults to the extraction prompt named in config.yaml. Pass an explicit
    path to load any other prompt file that follows the same convention (a
    fenced block holding the verbatim text, a `version <tag>` in the first
    heading) -- e.g. the optional preprocess prompt.
    """
    prompt_file = prompt_path or get_settings().prompt_path
    text = prompt_file.read_text(encoding="utf-8")

    fence = _FENCE_PATTERN.search(text)
    if fence is None:
        raise ValueError(f"No fenced prompt block found in {prompt_file}")

    version_match = _VERSION_PATTERN.search(text.splitlines()[0])
    if version_match is None:
        raise ValueError(f"No 'version <tag>' in the first heading of {prompt_file}")

    return fence.group(1).strip(), version_match.group(1)


SYSTEM_PROMPT, PROMPT_VERSION = load_active_prompt()
