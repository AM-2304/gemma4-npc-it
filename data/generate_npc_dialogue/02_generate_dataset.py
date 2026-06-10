#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
NPC Dialogue v2 — Dataset Generation (Step 02)
Uses Gemini 2.0/2.5 Flash to generate 24-turn NPC conversations.

Features:
- Exponential backoff retry (max 5 retries)
- JSON validation + repair prompts
- Incremental saving (crash-safe)
- Cost estimation

Usage:
    python data/generate_npc_dialogue/02_generate_dataset.py \
        [--input data/npc_dialogue_v2/system_prompts.jsonl] \
        [--output data/npc_dialogue_v2/raw_generated.jsonl] \
        [--model gemini-2.0-flash]
"""

import argparse
import json
import logging
import os
import re
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

GENERATION_PROMPT = """Generate a complete RPG NPC conversation dataset entry.

TASK: Create one realistic, engaging conversation between a player and the NPC defined in the system prompt.

NPC CHARACTER: The system_prompt field below defines this NPC's entire identity. Roleplay them consistently.

SYSTEM PROMPT:
{system_prompt}

CONVERSATION REQUIREMENTS:
- Exactly 24 messages alternating user/model (12 exchanges)
- Message 1 (system): The provided system_prompt
- Message 2 (model): NPC's opening greeting — natural, in-character, brief
- Messages 3-24: Natural back-and-forth (11 more exchanges)
- Player messages: 1-3 sentences. Can be curious, dismissive, friendly, confused, provocative
- NPC messages: Vary length 1-5 sentences. Short for reactions, longer for lore/quest info
- Include: emotional beats, personality reveals, optional mini-quest hook, world-building detail

INTERACTION TYPES — randomly pick one per conversation:
- Quest: NPC gives player a task, player asks questions, negotiates
- Trade: Player wants to buy/sell, NPC haggles, discusses wares
- Lore: Player asks about the world, NPC shares knowledge with personal spin
- Personal: Player discovers NPC's backstory, fears, dreams
- Conflict: Player says something NPC dislikes; NPC reacts but stays civil

QUALITY STANDARDS:
- NPC voice must match their background (a blacksmith speaks differently than a court mage)
- No purple prose; dialogue should feel spoken, not written
- NPC does NOT explain their personality; they demonstrate it through action
- Player should feel like someone is actually listening and responding to them specifically

OUTPUT: ONLY valid JSON. No markdown. No explanation. No trailing text.

JSON STRUCTURE:
{{
  "messages": [
    {{"role": "system", "content": "<the system_prompt>"}},
    {{"role": "model", "content": "<NPC opening>"}},
    {{"role": "user", "content": "<player>"}},
    {{"role": "model", "content": "<NPC>"}},
    ... (continue for 24 total messages)
  ]
}}"""

REPAIR_PROMPT = """The previous JSON was malformed. Output ONLY valid JSON with exactly 24 messages, starting with role: system. No markdown, no code fences, no explanation."""


def validate_conversation(data: dict, system_prompt: str) -> tuple[bool, str]:
    """Validate generated conversation JSON."""
    if not isinstance(data, dict):
        return False, "Not a dict"
    if "messages" not in data:
        return False, "Missing 'messages' key"

    msgs = data["messages"]
    if not isinstance(msgs, list):
        return False, "'messages' is not a list"
    if len(msgs) != 24:
        return False, f"Expected 24 messages, got {len(msgs)}"
    if msgs[0].get("role") != "system":
        return False, f"First message role is '{msgs[0].get('role')}', expected 'system'"

    # Check alternation starting from message 2
    expected_roles = ["model", "user"] * 11 + ["model"]  # 23 roles after system
    for i, (msg, expected) in enumerate(zip(msgs[1:], expected_roles), start=1):
        if msg.get("role") != expected:
            return False, f"Message {i+1} role is '{msg.get('role')}', expected '{expected}'"

    # Check content quality
    for i, msg in enumerate(msgs):
        content = msg.get("content", "")
        if not content or len(content.strip()) < 5:
            return False, f"Message {i+1} content too short: '{content[:20]}'"

    # Check for unresolved placeholders
    full_text = json.dumps(data)
    if "{{char}}" in full_text or "{{user}}" in full_text:
        return False, "Unresolved placeholders found"

    return True, "Valid"


def generate_one(client, model_name: str, system_prompt: str, max_retries: int = 5) -> dict | None:
    """Generate one conversation with retry and repair."""
    import google.generativeai as genai

    prompt = GENERATION_PROMPT.format(system_prompt=system_prompt)

    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt if attempt == 0 else prompt + "\n\n" + REPAIR_PROMPT,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.9,
                    max_output_tokens=8192,
                ),
            )

            text = response.text.strip()
            # Clean potential markdown fences
            text = re.sub(r'^```json\s*', '', text)
            text = re.sub(r'\s*```$', '', text)

            data = json.loads(text)
            is_valid, reason = validate_conversation(data, system_prompt)

            if is_valid:
                return data
            else:
                logger.warning(f"  Attempt {attempt+1}: Invalid — {reason}")
                if attempt < max_retries - 1:
                    prompt = GENERATION_PROMPT.format(system_prompt=system_prompt) + f"\n\nPREVIOUS ERROR: {reason}. Fix it."

        except json.JSONDecodeError as e:
            logger.warning(f"  Attempt {attempt+1}: JSON parse error — {e}")
        except Exception as e:
            logger.warning(f"  Attempt {attempt+1}: API error — {e}")
            wait = min(2 ** attempt, 30)
            time.sleep(wait)

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/npc_dialogue_v2/system_prompts.jsonl")
    parser.add_argument("--output", default="data/npc_dialogue_v2/raw_generated.jsonl")
    parser.add_argument("--model", default="gemini-2.0-flash")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (s)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY required")
        sys.exit(1)

    import google.generativeai as genai
    genai.configure(api_key=api_key)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load system prompts
    prompts = []
    with open(input_path, "r") as f:
        for line in f:
            prompts.append(json.loads(line.strip()))
    logger.info(f"Loaded {len(prompts)} system prompts")

    # Resume support
    existing = set()
    if args.resume and output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                row = json.loads(line.strip())
                existing.add(row.get("_source_idx", -1))
        logger.info(f"Resuming: {len(existing)} already generated")

    success = 0
    failed = 0
    total_retries = 0

    mode = "a" if args.resume else "w"
    with open(output_path, mode, encoding="utf-8") as fout:
        for idx, entry in enumerate(tqdm(prompts, desc="Generating")):
            if idx in existing:
                continue

            system_prompt = entry["system_prompt"]
            npc_name = entry.get("npc_name", "Unknown")

            result = generate_one(genai, args.model, system_prompt, args.max_retries)

            if result is not None:
                result["_source_idx"] = idx
                result["_npc_name"] = npc_name
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                success += 1
            else:
                failed += 1
                logger.error(f"Failed to generate for NPC: {npc_name} (idx={idx})")

            time.sleep(args.delay)

    logger.info("=" * 60)
    logger.info(f"Generation complete: {success} success, {failed} failed")
    logger.info(f"Success rate: {success/(success+failed)*100:.1f}%")
    logger.info(f"✅ Output: {output_path}")


if __name__ == "__main__":
    main()
