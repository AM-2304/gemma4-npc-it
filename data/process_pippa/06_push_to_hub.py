#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Push Processed Datasets to HuggingFace Hub — Step 06

Pushes all four processed Gemma4NPC datasets:
1. pippa-gemma4         — Processed PIPPA in Gemma 4 chat format
2. pippa-gemma4-filtered — Content-filtered PIPPA
3. NPC-dialogue-v2      — Synthetically generated NPC conversations
4. RolePlay-NPC-v2      — Combined dataset (train/val/test splits)

Usage:
    python data/process_pippa/06_push_to_hub.py --username YOUR_HF_USERNAME
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import login

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file into list of dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line.strip()))
    return rows


def push_single_dataset(
    data_path: str,
    repo_name: str,
    description: str,
    tags: list[str],
    private: bool = False,
    nsfw_flag: bool = False,
):
    """Push a single JSONL file as a HF dataset."""
    path = Path(data_path)
    if not path.exists():
        logger.warning(f"File not found, skipping: {path}")
        return

    rows = load_jsonl(data_path)
    dataset = Dataset.from_list(rows)

    logger.info(f"Pushing {len(rows)} rows to {repo_name}...")
    dataset.push_to_hub(
        repo_name,
        private=private,
    )
    logger.info(f"✅ Pushed {repo_name} ({len(rows)} rows)")


def push_split_dataset(
    train_path: str,
    val_path: str,
    test_path: str,
    repo_name: str,
):
    """Push a dataset with train/val/test splits."""
    splits = {}
    for name, path in [("train", train_path), ("validation", val_path), ("test", test_path)]:
        p = Path(path)
        if p.exists():
            rows = load_jsonl(path)
            splits[name] = Dataset.from_list(rows)
            logger.info(f"  {name}: {len(rows)} rows")
        else:
            logger.warning(f"  {name} file not found: {path}")

    if not splits:
        logger.error(f"No splits found for {repo_name}")
        return

    dataset_dict = DatasetDict(splits)
    logger.info(f"Pushing {repo_name} with splits: {list(splits.keys())}...")
    dataset_dict.push_to_hub(repo_name)
    logger.info(f"✅ Pushed {repo_name}")


def main():
    parser = argparse.ArgumentParser(description="Push datasets to HuggingFace Hub")
    parser.add_argument("--username", required=True, help="HuggingFace username")
    parser.add_argument("--private", action="store_true", help="Push as private datasets")
    parser.add_argument(
        "--datasets", nargs="+",
        choices=["pippa", "pippa-filtered", "npc-dialogue", "combined", "all"],
        default=["all"],
    )
    args = parser.parse_args()

    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN required")
        sys.exit(1)
    login(token=token)

    datasets_to_push = set(args.datasets)
    if "all" in datasets_to_push:
        datasets_to_push = {"pippa", "pippa-filtered", "npc-dialogue", "combined"}

    username = args.username

    if "pippa" in datasets_to_push:
        push_single_dataset(
            "data/final/pippa_gemma4.jsonl",
            f"{username}/pippa-gemma4",
            "Processed PIPPA in Gemma 4 native chat format",
            ["roleplay", "gemma4", "npc", "pippa"],
            private=args.private,
            nsfw_flag=True,
        )

    if "pippa-filtered" in datasets_to_push:
        push_single_dataset(
            "data/final/pippa_gemma4_filtered.jsonl",
            f"{username}/pippa-gemma4-filtered",
            "Content-filtered PIPPA in Gemma 4 format",
            ["roleplay", "gemma4", "npc", "filtered"],
            private=args.private,
            nsfw_flag=True,
        )

    if "npc-dialogue" in datasets_to_push:
        push_single_dataset(
            "data/final/npc_dialogue_v2.jsonl",
            f"{username}/NPC-dialogue-v2",
            "Synthetic NPC dialogue v2 (24-turn conversations)",
            ["roleplay", "gemma4", "npc", "synthetic"],
            private=args.private,
        )

    if "combined" in datasets_to_push:
        push_split_dataset(
            "data/final/RolePlay-NPC-v2_train.jsonl",
            "data/final/RolePlay-NPC-v2_val.jsonl",
            "data/final/RolePlay-NPC-v2_test.jsonl",
            f"{username}/RolePlay-NPC-v2",
        )

    logger.info("\n🎉 All datasets pushed successfully!")


if __name__ == "__main__":
    main()
