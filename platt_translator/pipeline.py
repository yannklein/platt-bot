"""Corpus processing pipeline: batch translate + validate with resume support."""

import json
import sys
import time
from pathlib import Path

from langchain_mistralai import ChatMistralAI

from .chains import build_translator_chain, build_validator_chain


def _load_input(path: Path) -> list[dict]:
    """Load input corpus from JSONL or plain-text file."""
    entries = []
    suffix = path.suffix.lower()

    with open(path, encoding="utf-8") as f:
        if suffix == ".jsonl":
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        elif suffix == ".txt":
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if line:
                    entries.append({"id": idx, "source": line})
        else:
            raise ValueError(f"Unsupported input format: {suffix} (use .jsonl or .txt)")

    return entries


def _load_done_ids(path: Path) -> set:
    """Read already-processed IDs from the output file for resume support."""
    done = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                done.add(record["id"])
    return done


def translate_one(
    text: str,
    *,
    translator_chain,
    validator_chain,
    max_retries: int = 2,
) -> dict:
    """Translate a single sentence and validate it.

    Returns a dict with keys: target, validation, retries.
    """
    best_target = None
    best_validation = None

    for attempt in range(1 + max_retries):
        target = translator_chain.invoke({"text": text}).strip()
        validation = validator_chain.invoke({"text": target}).strip().upper()

        # Normalise to one of the expected labels
        if "VALID" in validation and "INVALID" not in validation:
            validation = "VALID"
        elif "QUESTIONABLE" in validation:
            validation = "QUESTIONABLE"
        elif "INVALID" in validation:
            validation = "INVALID"

        if best_target is None or validation == "VALID":
            best_target = target
            best_validation = validation

        if validation == "VALID":
            return {"target": best_target, "validation": "VALID", "retries": attempt}

    return {"target": best_target, "validation": best_validation, "retries": max_retries}


def process_corpus(
    input_path: str | Path,
    output_path: str | Path,
    *,
    model: str = "mistral-large-latest",
    api_key: str | None = None,
    max_retries: int = 2,
    delay: float = 1.0,
) -> None:
    """Process an entire corpus: translate, validate, write results to JSONL.

    Supports resuming — already-processed IDs in the output file are skipped.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    llm_kwargs = {"model": model}
    if api_key:
        llm_kwargs["api_key"] = api_key
    llm = ChatMistralAI(**llm_kwargs)

    translator = build_translator_chain(llm)
    validator = build_validator_chain(llm)

    entries = _load_input(input_path)
    done_ids = _load_done_ids(output_path)

    pending = [e for e in entries if e["id"] not in done_ids]
    total = len(entries)
    skipped = total - len(pending)

    if skipped:
        print(f"Resuming: {skipped}/{total} already processed", file=sys.stderr)

    with open(output_path, "a", encoding="utf-8") as out:
        for i, entry in enumerate(pending, start=1):
            source = entry["source"]
            entry_id = entry["id"]
            print(
                f"[{skipped + i}/{total}] id={entry_id}: {source[:60]}...",
                file=sys.stderr,
            )

            result = translate_one(
                source,
                translator_chain=translator,
                validator_chain=validator,
                max_retries=max_retries,
            )

            record = {
                "id": entry_id,
                "source": source,
                "target": result["target"],
                "validation": result["validation"],
                "retries": result["retries"],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            if i < len(pending):
                time.sleep(delay)

    print(f"Done. Output written to {output_path}", file=sys.stderr)
