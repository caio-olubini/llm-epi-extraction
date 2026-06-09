"""Sort raw bulletin PDFs into dengue_related / not_dengue_related subdirectories.

Classification is based on keywords found in the first page of each PDF.
Dengue/arbovirus bulletins consistently title themselves with terms like
"arbovirose", "arboviral", "dengue", "chikungunya", "zika", or "COE".

The sort is non-destructive: it **copies** files rather than moving them, so
the original year-folder structure under data/raw/ is untouched. Subdirectories
are created as:

    data/raw/dengue_related/<year>/<filename>
    data/raw/not_dengue_related/<year>/<filename>

Re-running is idempotent: existing destination files are skipped.

Usage:
    python src/sort_corpus.py
    python src/sort_corpus.py --input data/raw/ --dry-run
"""

import argparse
import re
import shutil
import unicodedata
from pathlib import Path

import fitz  # PyMuPDF

from config import get_settings

# Keywords that identify an arbovirus/dengue bulletin. Matching is done on
# the accent-stripped, lowercased first-page text so diacritic variants are
# covered without enumerating them.
# Patterns that identify an arbovirus/dengue bulletin.  Most are simple
# substrings (long enough to be unambiguous); "coe" must be a whole word
# because it appears as a substring in common words like "orientacoes".
_KEYWORD_PATTERN = re.compile(
    r"arbovirose|arboviral|dengue|chikungunya|zika|febre amarela|\bcoe\b"
)


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def _is_dengue_related(pdf_path: Path) -> bool:
    """Return True if the PDF's first page mentions dengue/arbovirus keywords."""
    doc = fitz.open(pdf_path)
    try:
        cover_text = doc[0].get_text() if doc.page_count else ""
    finally:
        doc.close()

    normalised = _strip_accents(cover_text).lower()
    return bool(_KEYWORD_PATTERN.search(normalised))


def sort_corpus(input_root: Path, dry_run: bool = False) -> None:
    """Copy every PDF under input_root into dengue_related or not_dengue_related."""
    pdf_paths = sorted(input_root.rglob("*.pdf"))

    dengue_root = input_root / "dengue_related"
    other_root = input_root / "not_dengue_related"

    # Exclude PDFs that are themselves inside the output directories so that
    # re-runs do not re-classify already-sorted copies.
    pdf_paths = [
        p for p in pdf_paths
        if dengue_root not in p.parents and other_root not in p.parents
    ]

    counts = {"dengue": 0, "other": 0, "skipped": 0}

    for pdf_path in pdf_paths:
        # Preserve the year subfolder structure in the destination.
        relative = pdf_path.relative_to(input_root)
        is_dengue = _is_dengue_related(pdf_path)
        dest_dir = (dengue_root if is_dengue else other_root) / relative.parent
        dest = dest_dir / pdf_path.name
        label = "dengue_related" if is_dengue else "not_dengue_related"

        if dest.exists():
            print(f"  skip  {relative}  [{label}]")
            counts["skipped"] += 1
            continue

        print(f"  {'(dry) ' if dry_run else ''}copy  {relative}  -> {label}/")
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pdf_path, dest)

        if is_dengue:
            counts["dengue"] += 1
        else:
            counts["other"] += 1

    total = counts["dengue"] + counts["other"]
    print(
        f"\nDone: {total} classified "
        f"({counts['dengue']} dengue-related, {counts['other']} other, "
        f"{counts['skipped']} skipped)."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sort bulletin PDFs into dengue_related / not_dengue_related."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Root directory containing year-subfolders of PDFs (default: data.raw_pdfs from config.yaml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without actually copying anything.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    input_root = args.input or settings.data.raw_pdfs
    sort_corpus(input_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
