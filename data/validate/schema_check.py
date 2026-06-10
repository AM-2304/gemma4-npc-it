#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Schema Validation — Validates any JSONL file against Gemma 4 chat schema.
Exits with code 1 and detailed error report if ANY row fails.

Usage:
    python data/validate/schema_check.py data/final/pippa_gemma4.jsonl
    python data/validate/schema_check.py data/final/RolePlay-NPC-v2_train.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VALID_ROLES = {"system", "user", "model"}


def validate_row(row: dict, idx: int) -> list[str]:
    """Validate a single row. Returns list of error messages."""
    errors = []

    if not isinstance(row, dict):
        return [f"Row {idx}: Not a dict"]

    if "messages" not in row:
        return [f"Row {idx}: Missing 'messages' key"]

    msgs = row["messages"]
    if not isinstance(msgs, list):
        return [f"Row {idx}: 'messages' is not a list"]

    if len(msgs) < 4:
        errors.append(f"Row {idx}: Too few messages ({len(msgs)}, need >= 4)")

    if not msgs:
        return [f"Row {idx}: Empty messages list"]

    # First message must be system
    if msgs[0].get("role") != "system":
        errors.append(f"Row {idx}: First message role is '{msgs[0].get('role')}', expected 'system'")

    # Check all messages
    for i, msg in enumerate(msgs):
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Valid role check
        if role not in VALID_ROLES:
            errors.append(f"Row {idx}, msg {i}: Invalid role '{role}'")

        # Content check
        if content is None or (isinstance(content, str) and content.strip() == ""):
            errors.append(f"Row {idx}, msg {i}: Empty/null content")

    # Check alternation after system
    for i in range(2, len(msgs)):
        prev_role = msgs[i - 1].get("role", "")
        curr_role = msgs[i].get("role", "")
        if curr_role == prev_role:
            errors.append(f"Row {idx}, msg {i}: Duplicate role '{curr_role}' (prev was '{prev_role}')")

    # Must end on model turn
    if msgs and msgs[-1].get("role") != "model":
        errors.append(f"Row {idx}: Last message role is '{msgs[-1].get('role')}', expected 'model'")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate Gemma 4 chat schema")
    parser.add_argument("file", help="JSONL file to validate")
    parser.add_argument("--max-errors", type=int, default=50, help="Max errors to report")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        logger.error(f"File not found: {path}")
        sys.exit(1)

    all_errors = []
    total = 0
    failed_rows = 0

    with open(path, "r") as f:
        for idx, line in enumerate(f):
            total += 1
            try:
                row = json.loads(line.strip())
            except json.JSONDecodeError as e:
                all_errors.append(f"Row {idx}: JSON parse error — {e}")
                failed_rows += 1
                continue

            errors = validate_row(row, idx)
            if errors:
                all_errors.extend(errors)
                failed_rows += 1

    # Report
    logger.info("=" * 60)
    logger.info(f"Schema Validation: {path.name}")
    logger.info("=" * 60)
    logger.info(f"Total rows:  {total}")
    logger.info(f"Valid rows:  {total - failed_rows}")
    logger.info(f"Failed rows: {failed_rows}")

    if all_errors:
        logger.error(f"\n❌ {len(all_errors)} validation errors found:")
        for err in all_errors[:args.max_errors]:
            logger.error(f"  {err}")
        if len(all_errors) > args.max_errors:
            logger.error(f"  ... and {len(all_errors) - args.max_errors} more")
        sys.exit(1)
    else:
        logger.info(f"\n✅ All {total} rows pass schema validation")
        sys.exit(0)


if __name__ == "__main__":
    main()
