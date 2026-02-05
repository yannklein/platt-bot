"""CLI entry point: python -m corpus_builder"""

import argparse
from pathlib import Path

_DEFAULT_RAW_DIR = Path(__file__).parent.parent / "corpus" / "raw"
_DEFAULT_OUTPUT = Path(__file__).parent.parent / "corpus" / "input.jsonl"


def main():
    parser = argparse.ArgumentParser(
        description="Build German input corpus from OpenSubtitles and Tatoeba",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=_DEFAULT_RAW_DIR,
        help=f"Directory for raw downloads (default: {_DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=5,
        help="Minimum words per sentence (default: 5)",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=30,
        help="Maximum words per sentence (default: 30)",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Cap total sentences (default: no limit)",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["opensubtitles", "tatoeba"],
        default=["opensubtitles", "tatoeba"],
        help="Which sources to fetch (default: both)",
    )
    args = parser.parse_args()

    # --- Extract: fetch raw data ---
    raw_files = []

    if "tatoeba" in args.sources:
        from .sources.tatoeba import fetch as fetch_tatoeba

        raw_files.append(fetch_tatoeba(args.raw_dir))

    if "opensubtitles" in args.sources:
        from .sources.opensubtitles import fetch as fetch_opensubtitles

        raw_files.append(fetch_opensubtitles(args.raw_dir))

    # --- Load: filter, dedupe, write JSONL ---
    from .pipeline import build_input

    build_input(
        raw_files,
        args.output,
        min_words=args.min_words,
        max_words=args.max_words,
        max_sentences=args.max_sentences,
    )


if __name__ == "__main__":
    main()
