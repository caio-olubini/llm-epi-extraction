"""Entry point for running the extraction pipeline from the command line.

Usage:
    python src/ --input data/passages/passages.jsonl \
                --output data/extracted/signals.jsonl

    # Or with a synthetic demo passage (no --input flag needed):
    python src/

The demo mode runs a single hard-coded passage so you can verify the full
pipeline is wired correctly before you have real PDF passages to process.
Set LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL first (see .env.example).
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (one directory above src/).
# This must run before any module that reads os.environ (client.py, pipeline.py).
load_dotenv(Path(__file__).parent.parent / ".env")

from pipeline import run_corpus


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract qualitative signals from dengue bulletin passages."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="JSONL file produced by parse.py. Omit to run the built-in demo passage.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/extracted/signals.jsonl"),
        help="JSONL file where ExtractionRecords are appended (default: data/extracted/signals.jsonl).",
    )
    return parser.parse_args()


# A minimal synthetic passage used when no --input file is provided.
# Replace with real parsed-PDF passages once parse.py is built.
_DEMO_PASSAGES = [
    {
        "source_file": "boletim_se18_2024.pdf",
        "bulletin_publication_date": "2024-05-10",
        "epi_week_reported": "2024-SE18",
        "text": (
            "Na Bahia, observa-se tendencia de alta no numero de casos provaveis "
            "de dengue nas ultimas semanas, com circulacao predominante do sorotipo "
            "DENV-3, reintroduzido no estado. Recomenda-se intensificacao das acoes "
            "de controle vetorial. Alerta para as proximas semanas no periodo chuvoso."
        ),
    }
]


def main() -> None:
    args = _parse_args()

    if args.input is not None:
        with args.input.open(encoding="utf-8") as fh:
            passages = [json.loads(line) for line in fh if line.strip()]
    else:
        print("No --input provided. Running demo passage.")
        passages = _DEMO_PASSAGES

    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_corpus(passages, args.output)
    print(f"Records written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
