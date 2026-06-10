#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
NPC Dialogue v2 — JSON Validation (Step 03)
Validates every generated conversation for schema compliance.

Usage:
    python data/generate_npc_dialogue/03_validate_json.py \
        [--input data/npc_dialogue_v2/raw_generated.jsonl] \
        [--valid data/npc_dialogue_v2/validated.jsonl] \
        [--invalid data/npc_dialogue_v2/invalid.jsonl]
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


def validate_entry(data: dict) -> tuple[bool, str]:
    """Validate a single conversation entry."""
    if not isinstance(data, dict):
        return False, "Not a dict"
    if "messages" not in data:
        return False, "Missing 'messages'"

    msgs = data["messages"]
    if not isinstance(msgs, list):
        return False, "'messages' not a list"
    if len(msgs) != 24:
        return False, f"Expected 24 messages, got {len(msgs)}"

    # First message must be system
    if msgs[0].get("role") != "system":
        return False, f"First role is '{msgs[0].get('role')}', not 'system'"

    # Check alternation
    expected = ["system"] + (["model", "user"] * 11) + ["model"]
    for i, (msg, exp_role) in enumerate(zip(msgs, expected)):
        role = msg.get("role", "")
        if role != exp_role:
            return False, f"Msg {i}: role '{role}' != expected '{exp_role}'"

        content = msg.get("content", "")
        if not content or len(content.strip()) < 5:
            return False, f"Msg {i}: content too short ({len(content.strip())} chars)"

    # Check for unresolved placeholders
    full = json.dumps(data)
    for placeholder in ["{{char}}", "{{user}}", "{{random_user"]:
        if placeholder in full:
            return False, f"Unresolved placeholder: {placeholder}"

    return True, "Valid"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/npc_dialogue_v2/raw_generated.jsonl")
    parser.add_argument("--valid", default="data/npc_dialogue_v2/validated.jsonl")
    parser.add_argument("--invalid", default="data/npc_dialogue_v2/invalid.jsonl")
    args = parser.parse_args()

    for p in [args.valid, args.invalid]:
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r") as f:
        total = sum(1 for _ in f)

    valid_count = 0
    invalid_count = 0
    reasons = {}

    with open(args.input, "r") as fin, \
         open(args.valid, "w") as fvalid, \
         open(args.invalid, "w") as finvalid:

        for line in tqdm(fin, total=total, desc="Validating"):
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                invalid_count += 1
                reasons["json_parse_error"] = reasons.get("json_parse_error", 0) + 1
                finvalid.write(line)
                continue

            is_valid, reason = validate_entry(data)
            if is_valid:
                # Strip metadata fields
                clean = {"messages": data["messages"]}
                fvalid.write(json.dumps(clean, ensure_ascii=False) + "\n")
                valid_count += 1
            else:
                data["_validation_error"] = reason
                finvalid.write(json.dumps(data, ensure_ascii=False) + "\n")
                invalid_count += 1
                reasons[reason.split(":")[0]] = reasons.get(reason.split(":")[0], 0) + 1

    logger.info("=" * 60)
    logger.info(f"Valid:   {valid_count} ({valid_count/total*100:.1f}%)")
    logger.info(f"Invalid: {invalid_count} ({invalid_count/total*100:.1f}%)")
    if reasons:
        logger.info("Failure reasons:")
        for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
            logger.info(f"  {r}: {c}")
    logger.info(f"✅ Valid: {args.valid}")
    logger.info(f"✅ Invalid: {args.invalid}")


if __name__ == "__main__":
    main()
