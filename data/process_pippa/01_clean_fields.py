#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
PIPPA Dataset Cleaning — Step 01
Loads PygmalionAI/PIPPA from HuggingFace, removes unused columns,
and saves the cleaned dataset to JSONL.

Usage:
    python data/process_pippa/01_clean_fields.py [--output data/raw/pippa_cleaned.jsonl]
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Clean PIPPA dataset fields")
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/pippa_cleaned.jsonl",
        help="Output path for cleaned JSONL",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    if not hf_token:
        logger.error("HF_TOKEN required. Set via --hf-token or .env file.")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load dataset ---
    logger.info("Loading PygmalionAI/PIPPA from HuggingFace Hub...")
    logger.info("Note: This dataset requires authentication (not-for-all-audiences=true)")

    try:
        from huggingface_hub import hf_hub_download
        logger.info("Downloading pippa_deduped.jsonl via HuggingFace Hub...")
        file_path = hf_hub_download(
            repo_id="PygmalionAI/PIPPA",
            filename="pippa_deduped.jsonl",
            repo_type="dataset",
            token=hf_token,
        )
        dataset = load_dataset(
            "json",
            data_files=file_path,
            split="train",
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error(
            "Make sure you've accepted the dataset terms at "
            "https://huggingface.co/datasets/PygmalionAI/PIPPA"
        )
        sys.exit(1)

    logger.info(f"Loaded {len(dataset)} rows from PIPPA")

    # --- Identify fields to keep and remove ---
    KEEP_FIELDS = {"bot_name", "bot_description", "bot_definitions", "categories", "conversation"}
    REMOVE_FIELDS = {"submission_timestamp", "bot_id"}

    all_fields = set(dataset.column_names)
    actual_remove = REMOVE_FIELDS.intersection(all_fields)
    extra_fields = all_fields - KEEP_FIELDS - REMOVE_FIELDS

    logger.info(f"Dataset columns: {sorted(all_fields)}")
    logger.info(f"Keeping: {sorted(KEEP_FIELDS.intersection(all_fields))}")
    logger.info(f"Removing: {sorted(actual_remove)}")
    if extra_fields:
        logger.warning(f"Unexpected extra fields (will be removed): {sorted(extra_fields)}")

    # --- Compute statistics before cleaning ---
    logger.info("Computing dataset statistics...")
    null_counts = {field: 0 for field in KEEP_FIELDS}
    total_conv_lengths = []

    for row in dataset:
        for field in KEEP_FIELDS:
            if field in row:
                val = row[field]
                if val is None or (isinstance(val, str) and val.strip() == ""):
                    null_counts[field] += 1
                elif field == "conversation" and isinstance(val, list):
                    total_conv_lengths.append(len(val))

    # --- Clean and save ---
    logger.info(f"Writing cleaned dataset to {output_path}...")
    written = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for row in tqdm(dataset, desc="Cleaning"):
            cleaned_row = {}
            for field in KEEP_FIELDS:
                if field in row:
                    cleaned_row[field] = row[field]
                else:
                    cleaned_row[field] = None

            f.write(json.dumps(cleaned_row, ensure_ascii=False) + "\n")
            written += 1

    # --- Print statistics ---
    logger.info("=" * 60)
    logger.info("PIPPA Cleaning Statistics")
    logger.info("=" * 60)
    logger.info(f"Total rows loaded:    {len(dataset)}")
    logger.info(f"Total rows written:   {written}")
    logger.info(f"Fields kept:          {sorted(KEEP_FIELDS)}")
    logger.info(f"Fields removed:       {sorted(actual_remove | extra_fields)}")
    logger.info("")
    logger.info("Null/empty rates per field:")
    for field, count in sorted(null_counts.items()):
        rate = count / len(dataset) * 100 if len(dataset) > 0 else 0
        logger.info(f"  {field:25s}: {count:5d} nulls ({rate:.1f}%)")

    if total_conv_lengths:
        avg_len = sum(total_conv_lengths) / len(total_conv_lengths)
        min_len = min(total_conv_lengths)
        max_len = max(total_conv_lengths)
        logger.info("")
        logger.info(f"Conversation lengths:")
        logger.info(f"  Average: {avg_len:.1f} turns")
        logger.info(f"  Min:     {min_len} turns")
        logger.info(f"  Max:     {max_len} turns")

    logger.info(f"\n✅ Output saved to: {output_path}")


if __name__ == "__main__":
    main()
