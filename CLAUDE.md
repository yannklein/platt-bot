# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Four-stage pipeline that builds a parallel corpus of Standard High German → Francique rhénan lorrain (Platt lorrain) and fine-tunes a language model on it.

1. **`corpus_builder/`** — Extract & Load: fetches German sentences from OpenSubtitles and Tatoeba, filters and deduplicates them into `corpus/input.jsonl`.
2. **`platt_translator/`** — Transform: translates each sentence using Mistral AI via LangChain, validates output, writes `corpus/output.jsonl`.
3. **`training/prepare_dataset.py`** — Prepare: converts the parallel corpus into chat training format (`training/platt_chat_train.jsonl`) using varied conversation templates.
4. **Model training** — Fine-tunes a model on the prepared dataset using a Google Colab notebook.

The first two modules share `corpus/` as the handoff directory.

The trained model is deployed as a Hugging Face Space: https://huggingface.co/spaces/yannklein/platt-bot

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

# --- prepare_dataset (Prepare training data) ---
python training/prepare_dataset.py                                          # defaults: corpus/output.jsonl → training/platt_chat_train.jsonl
python training/prepare_dataset.py -i corpus/output.jsonl -o training/platt_chat_train.jsonl
python training/prepare_dataset.py --examples-per-item 3                    # more examples per corpus item
python training/prepare_dataset.py --include-questionable                   # also use QUESTIONABLE validations

# --- Model training (Fine-tune) ---
# Run via Google Colab notebook:
# https://colab.research.google.com/drive/1DiJyeGUXA93rqu8G2Qo-Iz9HI9Y2IVPe?usp=sharing

# --- Deploy LoRA adapter to Hugging Face ---
unzip platt-lorrain-lora.zip -d platt-lorrain-lora
huggingface-cli login
huggingface-cli upload yannklein/platt-bot platt-lorrain-lora/
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

### prepare_dataset

`training/prepare_dataset.py` converts the parallel corpus (`corpus/output.jsonl`) into chat-format training data (`training/platt_chat_train.jsonl`). It filters for VALID items, then generates multiple training examples per corpus entry using varied conversation templates (direct conversation, translation requests, Platt-to-Platt). A system prompt establishing the Platt persona is included in every example.

## Model Training

Fine-tuning is done in a Google Colab notebook: https://colab.research.google.com/drive/1DiJyeGUXA93rqu8G2Qo-Iz9HI9Y2IVPe?usp=sharing

The notebook takes `training/platt_chat_train.jsonl` (produced by `prepare_dataset.py`) as input and produces a fine-tuned model.

## Deployment

The fine-tuned model is hosted as a Hugging Face Space: https://huggingface.co/spaces/yannklein/platt-bot
