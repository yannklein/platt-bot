"""Fetch German monolingual sentences from OPUS OpenSubtitles."""

import gzip
import sys
from pathlib import Path

import requests

# Monolingual German raw text (one sentence per line, gzipped)
_URL = "https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/mono/de.txt.gz"


def fetch(output_dir: Path) -> Path:
    """Download OpenSubtitles German monolingual text.

    Returns path to the decompressed raw text file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "opensubtitles_de.txt"

    if raw_path.exists():
        print(f"OpenSubtitles: already downloaded → {raw_path}", file=sys.stderr)
        return raw_path

    gz_path = output_dir / "opensubtitles_de.txt.gz"

    print("OpenSubtitles: downloading German monolingual data...", file=sys.stderr)
    with requests.get(_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(gz_path, "wb") as f:
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

    print("OpenSubtitles: decompressing...", file=sys.stderr)
    with gzip.open(gz_path, "rt", encoding="utf-8") as gz, open(
        raw_path, "w", encoding="utf-8"
    ) as out:
        for line in gz:
            out.write(line)

    gz_path.unlink()
    print(f"OpenSubtitles: done → {raw_path}", file=sys.stderr)
    return raw_path
