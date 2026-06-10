#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Dataset Combination — Merges PIPPA + NPC_dialogue_v2 into final training set.

Usage:
    python data/combine/combine_datasets.py \
        [--pippa data/final/pippa_gemma4.jsonl] \
        [--npc data/final/npc_dialogue_v2.jsonl] \
        [--output-dir data/final] \
        [--seed 42] \
        [--push-to-hub] [--hub-repo username/RolePlay-NPC-v2]
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            rows.append(row)
    return rows


def save_jsonl(rows: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Combine datasets for Gemma4NPC")
    parser.add_argument("--pippa", default="data/final/pippa_gemma4.jsonl")
    parser.add_argument("--npc", default="data/final/npc_dialogue_v2.jsonl")
    parser.add_argument("--output-dir", default="data/final")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.98)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    pippa_data = []
    npc_data = []

    if Path(args.pippa).exists():
        pippa_data = load_jsonl(args.pippa)
        logger.info(f"Loaded {len(pippa_data)} rows from PIPPA")
    else:
        logger.warning(f"PIPPA file not found: {args.pippa}")

    if Path(args.npc).exists():
        npc_data = load_jsonl(args.npc)
        logger.info(f"Loaded {len(npc_data)} rows from NPC_dialogue_v2")
    else:
        logger.warning(f"NPC dialogue file not found: {args.npc}")

    # Tag source for statistics
    for row in pippa_data:
        row["_source"] = "pippa"
    for row in npc_data:
        row["_source"] = "npc_dialogue_v2"

    # Combine and shuffle
    all_data = pippa_data + npc_data
    rng = random.Random(args.seed)
    rng.shuffle(all_data)

    total = len(all_data)
    logger.info(f"Total combined rows: {total}")

    # Split: 98% train, 1% val, 1% test
    train_end = int(total * args.train_ratio)
    val_end = train_end + int(total * 0.01)

    train_data = all_data[:train_end]
    val_data = all_data[train_end:val_end]
    test_data = all_data[val_end:]

    # Remove source tags before saving
    for split in [train_data, val_data, test_data]:
        for row in split:
            row.pop("_source", None)

    # Save splits
    train_path = output_dir / "RolePlay-NPC-v2_train.jsonl"
    val_path = output_dir / "RolePlay-NPC-v2_val.jsonl"
    test_path = output_dir / "RolePlay-NPC-v2_test.jsonl"

    save_jsonl(train_data, str(train_path))
    save_jsonl(val_data, str(val_path))
    save_jsonl(test_data, str(test_path))

    # Statistics
    pippa_count = len(pippa_data)
    npc_count = len(npc_data)
    turn_counts = []
    for row in all_data:
        msgs = row.get("messages", [])
        turn_counts.append(len(msgs))

    logger.info("=" * 60)
    logger.info("Combined Dataset Statistics")
    logger.info("=" * 60)
    logger.info(f"PIPPA rows:           {pippa_count}")
    logger.info(f"NPC_dialogue_v2 rows: {npc_count}")
    logger.info(f"Total:                {total}")
    logger.info(f"Train split:          {len(train_data)} ({len(train_data)/total*100:.1f}%)")
    logger.info(f"Val split:            {len(val_data)} ({len(val_data)/total*100:.1f}%)")
    logger.info(f"Test split:           {len(test_data)} ({len(test_data)/total*100:.1f}%)")
    if turn_counts:
        avg = sum(turn_counts) / len(turn_counts)
        logger.info(f"Avg turns/convo:      {avg:.1f}")
    logger.info(f"\n✅ Saved to {output_dir}/")

    # Optional: Push to Hub
    if args.push_to_hub:
        load_dotenv()
        from datasets import Dataset, DatasetDict
        from huggingface_hub import login

        token = os.environ.get("HF_TOKEN")
        if not token:
            logger.error("HF_TOKEN required for --push-to-hub")
            return
        login(token=token)

        repo = args.hub_repo or os.environ.get("HF_DATASET_REPO", "chimbiwide/RolePlay-NPC-v2")
        dd = DatasetDict({
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(val_data),
            "test": Dataset.from_list(test_data),
        })
        dd.push_to_hub(repo)
        logger.info(f"✅ Pushed to {repo}")


if __name__ == "__main__":
    main()
