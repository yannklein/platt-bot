#!/usr/bin/env python3
"""
Convert parallel corpus to chat training format for fine-tuning.

Creates varied conversation patterns to teach the model to:
1. Respond naturally in Platt
2. Translate from German to Platt when asked
3. Maintain Platt persona throughout conversations
"""

import json
import random
from pathlib import Path

# System prompt that establishes the Platt persona
SYSTEM_PROMPT = """Du bist ein freundlicher Assistent, der Platt Lorrain (Francique rhénan lorrain) spricht.
Du antwortest IMMER auf Platt, egal in welcher Sprache die Frage gestellt wird.
Platt ist ein deutscher Dialekt aus Lothringen, nahe am Hochdeutschen aber mit eigenen Regeln:
- "ich" bleibt "ich", aber Endungen werden weicher
- "das" wird zu "dat"
- "was" wird zu "wat"
- Viele Wörter enden auf -e statt -en
- Verkleinerungen enden auf -che (wie "Kanéngche")
Sei natürlich, freundlich und hilfsbereit - immer auf Platt!"""

# Templates for varied training examples
TEMPLATES = [
    # Direct conversation (user asks in German, assistant responds in Platt)
    {"user": "{source}", "assistant": "{target}"},

    # Translation request
    {"user": "Wie sagt man auf Platt: {source}", "assistant": "{target}"},
    {"user": "Übersetze auf Platt: {source}", "assistant": "{target}"},
    {"user": "Auf Platt bitte: {source}", "assistant": "{target}"},

    # User speaks Platt, assistant responds in Platt (teaches Platt-to-Platt)
    {"user": "{target}", "assistant": "Jo, {target_variation}"},
]


def load_corpus(path: Path) -> list[dict]:
    """Load the output.jsonl corpus."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def create_training_example(item: dict, template: dict) -> dict:
    """Create a single training example from a corpus item."""
    # Simple variation for Platt-to-Platt responses
    target_variation = item["target"]
    if target_variation.endswith("!"):
        target_variation = target_variation[:-1] + ", gell!"
    elif target_variation.endswith("."):
        target_variation = target_variation[:-1] + ", jo."

    user_msg = template["user"].format(
        source=item["source"],
        target=item["target"],
        target_variation=target_variation
    )
    assistant_msg = template["assistant"].format(
        source=item["source"],
        target=item["target"],
        target_variation=target_variation
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ]
    }


def prepare_dataset(
    input_path: Path,
    output_path: Path,
    examples_per_item: int = 2,
    include_questionable: bool = False
):
    """
    Convert corpus to chat training format.

    Args:
        input_path: Path to output.jsonl (the translated corpus)
        output_path: Path to write training data
        examples_per_item: How many training examples to create per corpus item
        include_questionable: Whether to include QUESTIONABLE validations
    """
    corpus = load_corpus(input_path)

    # Filter by validation status
    valid_items = [
        item for item in corpus
        if item["validation"] == "VALID" or
           (include_questionable and item["validation"] == "QUESTIONABLE")
    ]

    print(f"Loaded {len(corpus)} items, {len(valid_items)} valid")

    training_examples = []

    for item in valid_items:
        # Select random templates for this item
        selected_templates = random.sample(
            TEMPLATES,
            min(examples_per_item, len(TEMPLATES))
        )

        for template in selected_templates:
            example = create_training_example(item, template)
            training_examples.append(example)

    # Shuffle to mix different template types
    random.shuffle(training_examples)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for example in training_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Created {len(training_examples)} training examples")
    print(f"Written to {output_path}")

    # Show a sample
    print("\n--- Sample training example ---")
    sample = random.choice(training_examples)
    for msg in sample["messages"]:
        role = msg["role"].upper()
        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        print(f"{role}: {content}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare training dataset")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("corpus/output.jsonl"),
        help="Input corpus path"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("training/platt_chat_train.jsonl"),
        help="Output training data path"
    )
    parser.add_argument(
        "--examples-per-item", "-n",
        type=int,
        default=2,
        help="Training examples to generate per corpus item"
    )
    parser.add_argument(
        "--include-questionable",
        action="store_true",
        help="Include QUESTIONABLE validations"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()
    random.seed(args.seed)

    prepare_dataset(
        args.input,
        args.output,
        args.examples_per_item,
        args.include_questionable
    )
