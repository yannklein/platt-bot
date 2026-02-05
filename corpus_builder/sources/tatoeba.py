"""Fetch German sentences from Tatoeba via HuggingFace datasets."""

import sys
from pathlib import Path

from datasets import load_dataset


def fetch(output_dir: Path) -> Path:
    """Download German sentences from Tatoeba.

    Returns path to a raw text file (one sentence per line).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "tatoeba_de.txt"

    if raw_path.exists():
        print(f"Tatoeba: already downloaded → {raw_path}", file=sys.stderr)
        return raw_path

    print("Tatoeba: loading German sentences from HuggingFace...", file=sys.stderr)
    ds = load_dataset(
        "tatoeba",
        lang1="de",
        lang2="en",
        split="train",
        trust_remote_code=True,
    )

    count = 0
    with open(raw_path, "w", encoding="utf-8") as out:
        for row in ds:
            sentence = row["translation"]["de"]
            if sentence:
                out.write(sentence.strip() + "\n")
                count += 1

    print(f"Tatoeba: {count} sentences → {raw_path}", file=sys.stderr)
    return raw_path
