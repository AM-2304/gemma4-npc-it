#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Dataset Statistics Report — Prints comprehensive stats for any JSONL dataset.

Usage:
    python data/validate/stats.py [file1.jsonl] [file2.jsonl] ...
    python data/validate/stats.py  # defaults to all files in data/final/
"""

import argparse
import json
import logging
from pathlib import Path
from collections import Counter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def analyze_file(path: str):
    """Print statistics for a single JSONL file."""
    p = Path(path)
    if not p.exists():
        logger.warning(f"File not found: {path}")
        return

    rows = 0
    turn_counts = []
    role_counts = Counter()
    content_lengths = []
    system_lengths = []

    with open(path, "r") as f:
        for line in f:
            row = json.loads(line.strip())
            msgs = row.get("messages", [])
            rows += 1
            turn_counts.append(len(msgs))

            for msg in msgs:
                role = msg.get("role", "unknown")
                role_counts[role] += 1
                content = msg.get("content", "")
                content_lengths.append(len(content))
                if role == "system":
                    system_lengths.append(len(content))

    logger.info(f"\n{'='*60}")
    logger.info(f"📊 Statistics for: {p.name}")
    logger.info(f"{'='*60}")
    logger.info(f"Total rows:          {rows}")
    logger.info(f"File size:           {p.stat().st_size / 1024 / 1024:.1f} MB")

    if turn_counts:
        avg_turns = sum(turn_counts) / len(turn_counts)
        logger.info(f"Avg turns/convo:     {avg_turns:.1f}")
        logger.info(f"Min turns:           {min(turn_counts)}")
        logger.info(f"Max turns:           {max(turn_counts)}")

    logger.info(f"\nRole distribution:")
    for role, count in sorted(role_counts.items()):
        logger.info(f"  {role:10s}: {count:8d}")

    if content_lengths:
        avg_len = sum(content_lengths) / len(content_lengths)
        logger.info(f"\nAvg content length:  {avg_len:.0f} chars")

    if system_lengths:
        avg_sys = sum(system_lengths) / len(system_lengths)
        logger.info(f"Avg system prompt:   {avg_sys:.0f} chars")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="JSONL files to analyze")
    args = parser.parse_args()

    if args.files:
        for f in args.files:
            analyze_file(f)
    else:
        # Default: analyze all files in data/final/
        final_dir = Path("data/final")
        if final_dir.exists():
            for f in sorted(final_dir.glob("*.jsonl")):
                analyze_file(str(f))
        else:
            logger.info("No files specified and data/final/ not found")


if __name__ == "__main__":
    main()
