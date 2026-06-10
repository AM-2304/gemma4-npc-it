#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
NPC Dialogue v2 — System Prompt Generation (Step 01)
Loads amaydle/npc-dialogue and constructs Gemma 4-native system prompts.

Usage:
    python data/generate_npc_dialogue/01_generate_system_prompts.py \
        [--output data/npc_dialogue_v2/system_prompts.jsonl]
"""

import argparse
import json
import logging
import os
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT_V2 = """You are {name}, a character living in a richly detailed world. Immerse yourself completely in this role.

Name: {name}
Background: {biography}
Personality Traits: Inferred from your background and example interactions
Example Interaction:
  Player: "{query}"
  {name}: "{response}" [Emotion: {emotion}]

Roleplay Rules:
- Stay completely in character. You do not know you are in a game.
- Speak in a way authentic to your background (dialect, vocabulary, world knowledge)
- React emotionally and authentically to what the player says
- Reference your background and world naturally in conversation
- Vary response length: sometimes 1-2 sentences, sometimes 4-5 when emotional or explaining
- Push the interaction forward: ask questions, reveal information, react
- Never say "I am an AI" or break the fourth wall
- You exist in your world, with your history, your concerns"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/npc_dialogue_v2/system_prompts.jsonl")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    load_dotenv()
    token = args.hf_token or os.environ.get("HF_TOKEN")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading amaydle/npc-dialogue from HuggingFace Hub...")
    dataset = load_dataset("amaydle/npc-dialogue", split="train", token=token)
    logger.info(f"Loaded {len(dataset)} NPC entries")

    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in tqdm(dataset, desc="Generating system prompts"):
            name = row.get("Name", "NPC")
            biography = row.get("Biography", "")
            query = row.get("Query", "")
            response = row.get("Response", "")
            emotion = row.get("Emotion", "neutral")

            system_prompt = SYSTEM_PROMPT_V2.format(
                name=name,
                biography=biography,
                query=query,
                response=response,
                emotion=emotion,
            )

            entry = {
                "npc_name": name,
                "biography": biography,
                "example_query": query,
                "example_response": response,
                "emotion": emotion,
                "system_prompt": system_prompt,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

    logger.info(f"✅ Generated {written} system prompts → {output_path}")


if __name__ == "__main__":
    main()
