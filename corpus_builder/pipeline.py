"""Filter, deduplicate, and merge raw source files into corpus/input.jsonl."""

import json
import re
import sys
from pathlib import Path

# Sentences outside this word-count range are dropped
DEFAULT_MIN_WORDS = 5
DEFAULT_MAX_WORDS = 30


def _is_usable(line: str, min_words: int, max_words: int) -> bool:
    """Quick heuristics to keep natural German sentences."""
    words = line.split()
    if not (min_words <= len(words) <= max_words):
        return False
    # Must start with an uppercase letter (German sentence convention)
    if not re.match(r"[A-ZÄÖÜ]", line):
        return False
    # Must end with sentence-ending punctuation
    if not re.search(r"[.!?]$", line):
        return False
    # Skip lines that are mostly non-letter (timestamps, markup, etc.)
    letters = sum(c.isalpha() for c in line)
    if letters < len(line) * 0.6:
        return False
    return True


def build_input(
    raw_files: list[Path],
    output_path: Path,
    *,
    min_words: int = DEFAULT_MIN_WORDS,
    max_words: int = DEFAULT_MAX_WORDS,
    max_sentences: int | None = None,
) -> int:
    """Read raw text files, filter + dedupe, write corpus/input.jsonl.

    Returns the number of sentences written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    sentence_id = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for raw_file in raw_files:
            source_name = raw_file.stem
            print(f"Processing {raw_file.name}...", file=sys.stderr)
            kept = 0

            with open(raw_file, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Normalise whitespace
                    line = " ".join(line.split())

                    if line in seen:
                        continue
                    if not _is_usable(line, min_words, max_words):
                        continue

                    seen.add(line)
                    sentence_id += 1
                    record = {
                        "id": sentence_id,
                        "source": line,
                        "origin": source_name,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept += 1

                    if max_sentences and sentence_id >= max_sentences:
                        print(
                            f"  reached --max-sentences {max_sentences}",
                            file=sys.stderr,
                        )
                        break

            print(f"  kept {kept} from {source_name}", file=sys.stderr)
            if max_sentences and sentence_id >= max_sentences:
                break

    print(f"Total: {sentence_id} sentences → {output_path}", file=sys.stderr)
    return sentence_id
