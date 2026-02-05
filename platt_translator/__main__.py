"""CLI entry point: python -m platt_translator"""

import argparse
import os
import sys

from dotenv import load_dotenv


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Translate High German → Francique rhénan lorrain",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input", "-i",
        help="Path to input corpus (.jsonl or .txt)",
    )
    group.add_argument(
        "--text", "-t",
        help="Translate a single sentence",
    )
    parser.add_argument(
        "--output", "-o",
        default="corpus/output.jsonl",
        help="Path to output JSONL file (default: corpus/output.jsonl)",
    )
    parser.add_argument(
        "--model", "-m",
        default=os.getenv("MISTRAL_MODEL", "mistral-large-latest"),
        help="Mistral model name (default: mistral-large-latest)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max validation retries per sentence (default: 2)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between API calls (default: 1.0)",
    )
    args = parser.parse_args()

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print(
            "Error: MISTRAL_API_KEY not set. "
            "Set it in your environment or in a .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.text:
        # Single sentence mode
        from langchain_mistralai import ChatMistralAI

        from .chains import build_translator_chain, build_validator_chain

        llm = ChatMistralAI(model=args.model, api_key=api_key)
        translator = build_translator_chain(llm)
        validator = build_validator_chain(llm)

        from .pipeline import translate_one

        result = translate_one(
            args.text,
            translator_chain=translator,
            validator_chain=validator,
            max_retries=args.max_retries,
        )
        print(f"Source:     {args.text}")
        print(f"Target:     {result['target']}")
        print(f"Validation: {result['validation']}")
        if result["retries"] > 0:
            print(f"Retries:    {result['retries']}")
    else:
        # Corpus mode
        from .pipeline import process_corpus

        process_corpus(
            args.input,
            args.output,
            model=args.model,
            api_key=api_key,
            max_retries=args.max_retries,
            delay=args.delay,
        )


if __name__ == "__main__":
    main()
