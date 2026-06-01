"""Turn raw bulletin PDFs into passage units for the extraction pipeline.

This module owns one concern: reading every PDF under a directory tree and
emitting one passage row per bulletin. It is deliberately **content-agnostic** --
it never decides whether a bulletin is about arboviroses. Deciding relevance is
the model's job (the `is_arbovirus_related` field), and the whole experiment
depends on feeding the corpus in unfiltered. The parser only extracts text and
the provenance the LLM cannot know (source path, publication date).

Granularity: one passage per bulletin (the full document text). Most bulletins
in this corpus are general Ministerio da Saude editions with no per-UF
"Situacao Epidemiologica" structure, so a per-state split does not generalise;
the model reads the whole document and assigns scope/UF itself.

Output JSONL matches what pipeline.py consumes -- one object per line:
    source_file               — path relative to the input root (POSIX slashes)
    bulletin_publication_date — ISO date parsed from the cover, or null
    epi_week_reported         — always null here (see note below)
    text                      — full document text

Why epi_week_reported is left null: most bulletins carry no SE, and inferring
the *reported* week from a multi-week report is a downstream temporal-alignment
concern, not the parser's. The field is carried through so the contract is
stable and a later step can fill it.

Usage:
    python src/parse.py --input data/raw/ --output data/passages/passages.jsonl
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

import fitz  # PyMuPDF

from config import get_settings


# Portuguese month names, full and abbreviated, collapse to the same 3-letter
# key once accents are stripped (e.g. "março", "marco", "Mar." -> "mar"). This
# lets one small table cover every spelling seen across the 2019-2026 covers.
_MONTH_TO_NUMBER = {
    "jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12,
}

# day [º|°] [de] month [.] [de] year — tolerant of the layout variants in the
# covers: "18 de dezembro de 2019", "22 nov. 2023", "13 de maio de 2022".
_DATE_PATTERN = re.compile(
    r"(\d{1,2})\s*[º°]?\s*(?:de\s+)?([A-Za-zçÇãáéíóúâêô]{3,9})\.?\s*(?:de\s+)?(\d{4})",
    re.IGNORECASE,
)


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def _parse_publication_date(cover_text: str, folder_year: int | None) -> str | None:
    """Return the bulletin's publication date as an ISO string, or None.

    Covers often print more than one date -- a template/sidebar artifact from a
    previous issue alongside the real one (e.g. a stale "10 Jan. 20232" next to
    "10 dez. 2025"). The folder year is the disambiguator: the bulletin lives in
    a year subfolder, so we prefer the candidate whose year matches it. Only
    tokens whose month word is an actual Portuguese month are accepted, which
    also rejects incidental number-word-number sequences.
    """
    candidates: list[tuple[int, int, int]] = []  # (year, month, day)
    for day_str, month_word, year_str in _DATE_PATTERN.findall(cover_text):
        month = _MONTH_TO_NUMBER.get(_strip_accents(month_word).lower()[:3])
        if month is None:
            continue
        day = int(day_str)
        if not 1 <= day <= 31:
            continue
        candidates.append((int(year_str), month, day))

    if not candidates:
        return None

    if folder_year is not None:
        year_matches = [c for c in candidates if c[0] == folder_year]
        if year_matches:
            candidates = year_matches

    year, month, day = candidates[0]
    return f"{year:04d}-{month:02d}-{day:02d}"


def _folder_year(pdf_path: Path, input_root: Path) -> int | None:
    """Return the year encoded in the PDF's subfolder name, if it is one."""
    for part in pdf_path.relative_to(input_root).parts:
        if re.fullmatch(r"(19|20)\d{2}", part):
            return int(part)
    return None


def parse_bulletin(pdf_path: Path, input_root: Path) -> dict:
    """Read one PDF into a passage dict (see module docstring for the contract)."""
    document = fitz.open(pdf_path)
    try:
        full_text = "\n\n".join(page.get_text() for page in document)
        cover_text = document[0].get_text() if document.page_count else ""
    finally:
        document.close()

    return {
        "source_file": pdf_path.relative_to(input_root).as_posix(),
        "bulletin_publication_date": _parse_publication_date(
            cover_text, _folder_year(pdf_path, input_root)
        ),
        "epi_week_reported": None,
        "text": full_text,
    }


def parse_corpus(input_root: Path, output_path: Path) -> None:
    """Parse every PDF under input_root, writing one passage per line to output."""
    pdf_paths = sorted(input_root.rglob("*.pdf"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    parsed = 0
    undated = 0
    with output_path.open("w", encoding="utf-8") as sink:
        for pdf_path in pdf_paths:
            passage = parse_bulletin(pdf_path, input_root)
            if passage["bulletin_publication_date"] is None:
                undated += 1
            sink.write(json.dumps(passage, ensure_ascii=False) + "\n")
            parsed += 1

    print(f"Parsed {parsed} bulletins -> {output_path}")
    if undated:
        print(f"  {undated} had no parseable publication date (written with null date).")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse Ministerio da Saude bulletin PDFs into passage units."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Directory tree of PDFs to parse (default: data.raw_pdfs from config.yaml).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL file to write (default: data.passages from config.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    input_root = args.input or settings.data.raw_pdfs
    output_path = args.output or settings.data.passages
    parse_corpus(input_root, output_path)


if __name__ == "__main__":
    main()
