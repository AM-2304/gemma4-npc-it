#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
PIPPA Placeholder Replacement — Step 02
Replaces {{char}}, {{user}}, and {{random_user_N}} placeholders with actual names.

Usage:
    python data/process_pippa/02_replace_placeholders.py \
        [--input data/raw/pippa_cleaned.jsonl] \
        [--output data/processed/pippa_placeholders_replaced.jsonl] \
        [--names data/process_pippa/first_names.json] \
        [--seed 42]
"""

import argparse
import json
import logging
import random
import re
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Replace PIPPA placeholders")
    parser.add_argument(
        "--input", type=str, default="data/raw/pippa_cleaned.jsonl",
        help="Input cleaned JSONL",
    )
    parser.add_argument(
        "--output", type=str, default="data/processed/pippa_placeholders_replaced.jsonl",
        help="Output JSONL with placeholders replaced",
    )
    parser.add_argument(
        "--names", type=str, default="data/process_pippa/first_names.json",
        help="Path to first_names.json",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def load_names(path: str) -> list[str]:
    """Load the list of first names from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        names = json.load(f)
    logger.info(f"Loaded {len(names)} first names from {path}")
    return names


def replace_in_text(
    text: str,
    bot_name: str,
    user_name: str,
    rng: random.Random,
    names: list[str],
    counters: dict,
) -> str:
    """Replace all {{char}} and {{user}} variants in a text string."""
    if text is None:
        return text

    original = text

    # Replace {{char}} with bot_name
    if "{{char}}" in text:
        count = text.count("{{char}}")
        text = text.replace("{{char}}", bot_name)
        counters["{{char}}"] += count

    # Replace {{user}} with user_name (case-sensitive match for exact)
    if "{{user}}" in text:
        count = text.count("{{user}}")
        text = text.replace("{{user}}", user_name)
        counters["{{user}}"] += count

    # Replace {{User}} with user_name
    if "{{User}}" in text:
        count = text.count("{{User}}")
        text = text.replace("{{User}}", user_name)
        counters["{{User}}"] += count

    # Replace {{random_user_N}} patterns (e.g., {{random_user_1}}, {{random_user_42}})
    random_user_pattern = re.compile(r'\{\{random_user_\d+\}\}')
    matches = random_user_pattern.findall(text)
    if matches:
        # Use a consistent mapping per conversation: same placeholder → same name
        for match in set(matches):
            replacement = rng.choice(names)
            text = text.replace(match, replacement)
            counters["{{random_user_N}}"] += text.count(replacement)

    return text


def replace_in_conversation(
    conversation: list[dict],
    bot_name: str,
    user_name: str,
    rng: random.Random,
    names: list[str],
    counters: dict,
) -> list[dict]:
    """Replace placeholders in all conversation turns."""
    result = []
    for turn in conversation:
        new_turn = dict(turn)
        if "message" in new_turn and isinstance(new_turn["message"], str):
            new_turn["message"] = replace_in_text(
                new_turn["message"], bot_name, user_name, rng, names, counters
            )
        result.append(new_turn)
    return result


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    names = load_names(args.names)

    # Counters for replacement statistics
    counters = {
        "{{char}}": 0,
        "{{user}}": 0,
        "{{User}}": 0,
        "{{random_user_N}}": 0,
    }
    field_counters = {
        "bot_description": 0,
        "bot_definitions": 0,
        "conversation": 0,
    }

    # Count input rows
    with open(input_path, "r", encoding="utf-8") as f:
        total_rows = sum(1 for _ in f)

    logger.info(f"Processing {total_rows} rows from {input_path}...")

    written = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, total=total_rows, desc="Replacing placeholders"):
            row = json.loads(line.strip())
            bot_name = row.get("bot_name", "Character")

            # Sample a deterministic user name for this row
            user_name = rng.choice(names)

            # Replace in bot_description
            if row.get("bot_description"):
                old = row["bot_description"]
                row["bot_description"] = replace_in_text(
                    old, bot_name, user_name, rng, names, counters
                )
                if row["bot_description"] != old:
                    field_counters["bot_description"] += 1

            # Replace in bot_definitions
            if row.get("bot_definitions"):
                old = row["bot_definitions"]
                row["bot_definitions"] = replace_in_text(
                    old, bot_name, user_name, rng, names, counters
                )
                if row["bot_definitions"] != old:
                    field_counters["bot_definitions"] += 1

            # Replace in conversation turns
            if row.get("conversation") and isinstance(row["conversation"], list):
                old_conv = json.dumps(row["conversation"])
                row["conversation"] = replace_in_conversation(
                    row["conversation"], bot_name, user_name, rng, names, counters
                )
                if json.dumps(row["conversation"]) != old_conv:
                    field_counters["conversation"] += 1

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    # --- Statistics ---
    logger.info("=" * 60)
    logger.info("Placeholder Replacement Statistics")
    logger.info("=" * 60)
    logger.info(f"Rows processed: {written}")
    logger.info("")
    logger.info("Replacement counts by placeholder type:")
    for placeholder, count in sorted(counters.items()):
        logger.info(f"  {placeholder:25s}: {count:6d} replacements")
    logger.info("")
    logger.info("Rows with replacements by field:")
    for field, count in sorted(field_counters.items()):
        logger.info(f"  {field:25s}: {count:6d} rows affected")
    logger.info(f"\n✅ Output saved to: {output_path}")


if __name__ == "__main__":
    main()
