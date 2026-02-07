"""Fetch German sentences from Tatoeba (direct TSV download)."""

import bz2
import csv
import sys
from pathlib import Path

import requests

# TSV columns: sentence_id \t lang \t text
_URL = "https://downloads.tatoeba.org/exports/per_language/deu/deu_sentences.tsv.bz2"


def fetch(output_dir: Path) -> Path:
    """Download German sentences from Tatoeba.

    Returns path to a raw text file (one sentence per line).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "tatoeba_de.txt"

    if raw_path.exists():
        print(f"Tatoeba: already downloaded → {raw_path}", file=sys.stderr)
        return raw_path

    bz2_path = output_dir / "deu_sentences.tsv.bz2"

    print("Tatoeba: downloading German sentences...", file=sys.stderr)
    with requests.get(_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(bz2_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(
                        f"\r  {downloaded >> 20} / {total >> 20} MB ({pct}%)",
                        end="",
                        file=sys.stderr,
                    )
    print(file=sys.stderr)

    print("Tatoeba: decompressing and extracting sentences...", file=sys.stderr)
    count = 0
    with bz2.open(bz2_path, "rt", encoding="utf-8") as bzf, open(
        raw_path, "w", encoding="utf-8"
    ) as out:
        reader = csv.reader(bzf, delimiter="\t")
        for row in reader:
            # columns: sentence_id, lang, text
            if len(row) >= 3:
                text = row[2].strip()
                if text:
                    out.write(text + "\n")
                    count += 1

    bz2_path.unlink()
    print(f"Tatoeba: {count} sentences → {raw_path}", file=sys.stderr)
    return raw_path
