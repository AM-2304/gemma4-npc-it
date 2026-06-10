#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
NPC Dialogue v2 — Finalize JSONL (Step 04)
Strips generation metadata and ensures clean {"messages": [...]} format.

Usage:
    python data/generate_npc_dialogue/04_convert_to_jsonl.py \
        [--input data/npc_dialogue_v2/validated.jsonl] \
        [--output data/final/npc_dialogue_v2.jsonl]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/npc_dialogue_v2/validated.jsonl")
    parser.add_argument("--output", default="data/final/npc_dialogue_v2.jsonl")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r") as f:
        total = sum(1 for _ in f)

    written = 0
    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for line in tqdm(fin, total=total, desc="Finalizing"):
            data = json.loads(line.strip())
            # Keep only messages — strip any metadata
            clean = {"messages": data["messages"]}
            fout.write(json.dumps(clean, ensure_ascii=False) + "\n")
            written += 1

    logger.info(f"✅ Finalized {written} conversations → {args.output}")


if __name__ == "__main__":
    main()
