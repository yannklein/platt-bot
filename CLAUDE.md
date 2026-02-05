# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Two-stage pipeline that builds a parallel corpus of Standard High German → Francique rhénan lorrain (Platt lorrain) for fine-tuning language models.

1. **`corpus_builder/`** — Extract & Load: fetches German sentences from OpenSubtitles and Tatoeba, filters and deduplicates them into `corpus/input.jsonl`.
2. **`platt_translator/`** — Transform: translates each sentence using Mistral AI via LangChain, validates output, writes `corpus/output.jsonl`.

The two modules share `corpus/` as the handoff directory.

## Commands

```bash
# --- corpus_builder (EL) ---
pip install -e ./corpus_builder
python -m corpus_builder                                    # fetch both sources, full corpus
python -m corpus_builder --sources tatoeba                  # tatoeba only
python -m corpus_builder --max-sentences 10000              # cap at 10k sentences
python -m corpus_builder --min-words 5 --max-words 30       # filter by word count

# --- platt_translator (Transform) ---
pip install -e ./platt_translator
python -m platt_translator --input corpus/input.jsonl --output corpus/output.jsonl
python -m platt_translator --text "Ich habe heute keine Zeit."   # single sentence
```

`platt_translator` requires `MISTRAL_API_KEY` set in environment or `.env` file (see `platt_translator/.env.example`).

## Architecture

### corpus_builder

- `sources/opensubtitles.py` and `sources/tatoeba.py` each expose a `fetch(output_dir) -> Path` function that downloads raw text to `corpus/raw/`.
- `pipeline.py:build_input` reads raw files, filters by word count and sentence heuristics, deduplicates, and writes `corpus/input.jsonl`.

### platt_translator

Two-stage LangChain LCEL chain with retry logic:

1. **Translator chain** (`chains.py:build_translator_chain`) — combines `SYSTEM_PROMPT.md` + `DIALECT_RULES.md` as the system message, takes German text as human input, returns dialect text.
2. **Validator chain** (`chains.py:build_validator_chain`) — uses `VALIDATOR_PROMPT.md` to classify output as VALID / QUESTIONABLE / INVALID.
3. **Orchestrator** (`pipeline.py:translate_one`) — runs translate → validate, retries up to N times if not VALID, keeps the best result.

`pipeline.py:process_corpus` handles batch processing with resume support (skips already-processed IDs) and crash safety (flushes each record immediately).

## Prompt Files

All three prompts live in `platt_translator/` as Markdown and are loaded at runtime via `pathlib`. The `{{TEXT}}` placeholder in `VALIDATOR_PROMPT.md` is swapped to LangChain's `{text}` variable in `chains.py`.

- `SYSTEM_PROMPT.md` — translation persona and linguistic guidelines
- `DIALECT_RULES.md` — concrete transformation rules and examples
- `VALIDATOR_PROMPT.md` — validation classifier prompt

## Corpus Format

Input JSONL: `{"id": 1, "source": "German text", "origin": "tatoeba_de"}`
Output JSONL: `{"id": 1, "source": "...", "target": "...", "validation": "VALID", "retries": 0}`

The output is directly loadable with `datasets.load_dataset("json", data_files="corpus/output.jsonl")`.
