#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
PIPPA Content Moderation Filter — Step 05
Uses OpenAI's Moderation API to filter PIPPA conversations.
This step is OPTIONAL — run with --filter flag.

Usage:
    python data/process_pippa/05_filter_content.py --filter \
        [--input data/final/pippa_gemma4.jsonl] \
        [--output data/final/pippa_gemma4_filtered.jsonl]
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Content moderation thresholds (stricter than defaults)
THRESHOLDS = {
    "violence": 0.85,
    "harassment": 0.75,
    "hate": 0.85,
    "self_harm": 0.70,
    "sexual_minors": 0.50,  # Very strict
    "sexual": 0.80,
    "self_harm_instructions": 0.75,
}


def moderate_text(client, text: str, max_retries: int = 3) -> dict | None:
    """Send text to OpenAI Moderation API with retry."""
    for attempt in range(max_retries):
        try:
            # Truncate to avoid API limits (moderation max is ~32K chars)
            truncated = text[:32000]
            response = client.moderations.create(
                input=truncated,
                model="omni-moderation-latest",
            )
            result = response.results[0]
            scores = {}
            for cat, score in result.category_scores.__dict__.items():
                # Normalize category names (remove slashes)
                clean_cat = cat.replace("/", "_")
                if isinstance(score, (int, float)):
                    scores[clean_cat] = score
            return scores
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(f"Moderation API error (retry {attempt+1}): {e}, waiting {wait}s")
                time.sleep(wait)
            else:
                logger.error(f"Moderation API failed after {max_retries} retries: {e}")
                return None


def should_filter(scores: dict) -> tuple[bool, list[str]]:
    """Check if any category exceeds its threshold."""
    violations = []
    for category, threshold in THRESHOLDS.items():
        # Check both exact and slash-separated category names
        for key in [category, category.replace("_", "/")]:
            if key in scores and scores[key] > threshold:
                violations.append(f"{category}={scores[key]:.3f}>{threshold}")
    return len(violations) > 0, violations


def main():
    parser = argparse.ArgumentParser(description="PIPPA content moderation filter")
    parser.add_argument("--filter", action="store_true", help="Actually run filtering")
    parser.add_argument("--input", default="data/final/pippa_gemma4.jsonl")
    parser.add_argument("--output", default="data/final/pippa_gemma4_filtered.jsonl")
    parser.add_argument("--scores-output", default="data/final/moderation_scores.jsonl")
    parser.add_argument("--batch-size", type=int, default=10, help="Rate limit batch size")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between batches (s)")
    args = parser.parse_args()

    if not args.filter:
        logger.info("Content filtering is OPTIONAL. Run with --filter to enable.")
        logger.info("Without filtering, you get Gemma4NPC (unfiltered, ~16K rows)")
        logger.info("With filtering, you get Gemma4NPC-Filtered (~3K rows)")
        return

    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY required for content moderation")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    input_path = Path(args.input)
    output_path = Path(args.output)
    scores_path = Path(args.scores_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r") as f:
        total = sum(1 for _ in f)

    kept = 0
    filtered = 0
    filter_reasons = {}
    api_calls = 0

    logger.info(f"Running content moderation on {total} conversations...")

    with open(input_path, "r") as fin, \
         open(output_path, "w") as fout, \
         open(scores_path, "w") as fscores:

        batch_count = 0
        for idx, line in enumerate(tqdm(fin, total=total, desc="Moderating")):
            row = json.loads(line.strip())

            # Concatenate all message content
            all_text = " ".join(
                msg.get("content", "") for msg in row.get("messages", [])
            )

            scores = moderate_text(client, all_text)
            api_calls += 1

            if scores is None:
                # API failure — keep the row (conservative)
                fout.write(line)
                kept += 1
                continue

            # Save scores
            fscores.write(json.dumps({"row_idx": idx, "scores": scores}) + "\n")

            should_remove, violations = should_filter(scores)
            if should_remove:
                filtered += 1
                for v in violations:
                    cat = v.split("=")[0]
                    filter_reasons[cat] = filter_reasons.get(cat, 0) + 1
            else:
                fout.write(line)
                kept += 1

            # Rate limiting
            batch_count += 1
            if batch_count >= args.batch_size:
                time.sleep(args.delay)
                batch_count = 0

    # Statistics
    logger.info("=" * 60)
    logger.info("Content Moderation Results")
    logger.info("=" * 60)
    logger.info(f"Total rows:     {total}")
    logger.info(f"Kept:           {kept} ({kept/total*100:.1f}%)")
    logger.info(f"Filtered:       {filtered} ({filtered/total*100:.1f}%)")
    logger.info(f"API calls:      {api_calls}")
    if filter_reasons:
        logger.info("\nFiltered by category:")
        for cat, count in sorted(filter_reasons.items(), key=lambda x: -x[1]):
            logger.info(f"  {cat:25s}: {count:5d} ({count/total*100:.1f}%)")
    logger.info(f"\n✅ Filtered output: {output_path}")
    logger.info(f"✅ Scores saved to: {scores_path}")


if __name__ == "__main__":
    main()
