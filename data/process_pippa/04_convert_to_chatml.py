#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Convert PIPPA to Gemma 4 Native Chat Format — Step 04

CRITICAL: Uses Gemma 4's native `system` role (NOT the Gemma 3n hack of
injecting system as first user message). This is the single most important
architectural upgrade over Gemma3NPC.

Output format:
    {"messages": [
        {"role": "system", "content": "<system_prompt>"},
        {"role": "user", "content": "..."},
        {"role": "model", "content": "..."},
        ...
    ]}

Usage:
    python data/process_pippa/04_convert_to_chatml.py \
        [--input data/processed/pippa_clean.jsonl] \
        [--output data/final/pippa_gemma4.jsonl]
"""

import argparse
import json
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Gemma 4 NPC System Prompt Template — uses native system role
SYSTEM_PROMPT_TEMPLATE = """You are roleplaying as {bot_name}. Stay in character at all times. Respond naturally, concisely, and in a way that advances the conversation. Avoid long monologues.

Character Name: {bot_name}
Character Category: {categories}
Character Description: {bot_description}
Character Background & Example Dialogue: {bot_definitions}

Immerse yourself completely in this character. Do not break the fourth wall."""


def build_system_prompt(row: dict) -> str:
    """Build the system prompt from PIPPA row fields."""
    bot_name = row.get("bot_name", "Character")
    categories = row.get("categories", "General")
    if isinstance(categories, list):
        categories = ", ".join(categories)
    bot_description = row.get("bot_description", "")
    bot_definitions = row.get("bot_definitions", "")

    return SYSTEM_PROMPT_TEMPLATE.format(
        bot_name=bot_name,
        categories=categories or "General",
        bot_description=bot_description or "No description provided.",
        bot_definitions=bot_definitions or "No additional background.",
    )


def convert_conversation(conversation: list[dict]) -> list[dict] | None:
    """
    Convert PIPPA conversation format to Gemma 4 chat turns.
    PIPPA format: [{"message": str, "is_human": bool}, ...]
    Gemma 4 format: [{"role": "user"|"model", "content": str}, ...]

    Returns None if conversation has invalid turn ordering.
    """
    if not conversation or len(conversation) < 2:
        return None

    turns = []
    prev_is_human = None

    for i, turn in enumerate(conversation):
        is_human = turn.get("is_human", True)
        message = turn.get("message", "").strip()

        if not message:
            continue

        # Detect double turns (same role twice in a row)
        if prev_is_human is not None and is_human == prev_is_human:
            # Merge with previous turn if same role
            if turns:
                turns[-1]["content"] += "\n" + message
                continue
            else:
                return None

        role = "user" if is_human else "model"
        turns.append({"role": role, "content": message})
        prev_is_human = is_human

    if not turns:
        return None

    # Ensure conversation starts with a valid pattern
    # Gemma 4 chat template requires the first message to be user (after system)
    if turns[0]["role"] == "model":
        turns.insert(0, {"role": "user", "content": "*Approaches*"})

    # Ensure conversation ends on a model turn
    if turns[-1]["role"] != "model":
        # Drop the trailing user turn
        turns = turns[:-1]

    if len(turns) < 2:
        return None

    return turns


def main():
    parser = argparse.ArgumentParser(description="Convert PIPPA to Gemma 4 chat format")
    parser.add_argument("--input", default="data/processed/pippa_clean.jsonl")
    parser.add_argument("--output", default="data/final/pippa_gemma4.jsonl")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        return

    with open(input_path, "r") as f:
        total = sum(1 for _ in f)

    written = 0
    skipped = 0
    skip_reasons = {"no_conversation": 0, "bad_ordering": 0, "too_short": 0}
    turn_counts = []

    logger.info(f"Converting {total} rows to Gemma 4 native chat format...")

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for idx, line in enumerate(tqdm(fin, total=total, desc="Converting")):
            row = json.loads(line.strip())

            # Build system prompt
            system_prompt = build_system_prompt(row)

            # Convert conversation
            conversation = row.get("conversation", [])
            if not conversation:
                skipped += 1
                skip_reasons["no_conversation"] += 1
                continue

            turns = convert_conversation(conversation)
            if turns is None:
                skipped += 1
                skip_reasons["bad_ordering"] += 1
                continue

            # Build final message list with system role
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(turns)

            # Validate: minimum 4 messages (system + at least 3 turns)
            if len(messages) < 4:
                skipped += 1
                skip_reasons["too_short"] += 1
                continue

            # Validate turn alternation after system
            valid = True
            for i in range(2, len(messages)):
                expected = "user" if messages[i - 1]["role"] == "model" else "model"
                if i == 1:
                    continue  # First turn after system can be either
                if messages[i]["role"] == messages[i - 1]["role"] and i > 1:
                    valid = False
                    break

            if not valid:
                skipped += 1
                skip_reasons["bad_ordering"] += 1
                continue

            output_row = {"messages": messages}
            fout.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            written += 1
            turn_counts.append(len(messages))

    # Statistics
    logger.info("=" * 60)
    logger.info("Gemma 4 Chat Conversion Statistics")
    logger.info("=" * 60)
    logger.info(f"Rows converted:     {written}")
    logger.info(f"Rows skipped:       {skipped}")
    for reason, count in skip_reasons.items():
        logger.info(f"  {reason:20s}: {count}")
    if turn_counts:
        avg = sum(turn_counts) / len(turn_counts)
        logger.info(f"Avg turns/convo:    {avg:.1f}")
        logger.info(f"Min turns:          {min(turn_counts)}")
        logger.info(f"Max turns:          {max(turn_counts)}")
    logger.info(f"\n✅ Output: {output_path}")


if __name__ == "__main__":
    main()
