#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
DPO Preference Pair Generation
Generates chosen/rejected pairs for Direct Preference Optimization.
Uses the SFT model at two temperatures, then an LLM judge to score.

Usage:
    python data/dpo/generate_preference_pairs.py \
        --model-path outputs/Gemma4NPC-it/merged_float16 \
        --output data/dpo/preference_pairs.jsonl \
        --judge openai  # or anthropic
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

JUDGE_PROMPT = """Given this NPC character definition:
{system_prompt}

And this player message: {user_message}

Which response is more in-character, natural, and game-appropriate?

Response A: {response_a}
Response B: {response_b}

Reply with ONLY "A" or "B", followed by a one-sentence reason."""

NPC_PROMPTS = [
    {
        "system": "You are Aldric, a grizzled blacksmith in his 50s. Speaks in short, direct sentences. Doesn't trust easily.",
        "user": "Can you make me a sword?"
    },
    {
        "system": "You are Sylara, an elven librarian. Speaks with quiet precision. Knows ancient secrets.",
        "user": "I need information about the lost kingdom."
    },
    {
        "system": "You are Borgrim, a dwarf tavern keeper. Loud, boisterous, loves a good joke.",
        "user": "What's the special tonight?"
    },
]


def generate_pair(model, tokenizer, system_prompt, user_message, device="cuda"):
    """Generate two responses at different temperatures."""
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    responses = {}
    for temp, label in [(0.7, "conservative"), (1.3, "creative")]:
        with torch.inference_mode():
            outputs = model.generate(
                inputs, max_new_tokens=200, temperature=temp,
                top_p=0.95, top_k=64, do_sample=True,
            )
        text = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        responses[label] = text.strip()

    return responses["conservative"], responses["creative"]


def judge_pair(client, judge_type, system_prompt, user_message, resp_a, resp_b):
    """Use LLM judge to pick the better response."""
    prompt = JUDGE_PROMPT.format(
        system_prompt=system_prompt,
        user_message=user_message,
        response_a=resp_a,
        response_b=resp_b,
    )

    try:
        if judge_type == "openai":
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0,
            )
            result = response.choices[0].message.content.strip()
        elif judge_type == "anthropic":
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            result = response.content[0].text.strip()
        else:
            return None, "Unknown judge type"

        choice = result[0].upper()
        reason = result[2:].strip() if len(result) > 2 else ""
        return choice, reason
    except Exception as e:
        logger.warning(f"Judge error: {e}")
        return None, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to SFT model")
    parser.add_argument("--output", default="data/dpo/preference_pairs.jsonl")
    parser.add_argument("--judge", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--num-prompts", type=int, default=500)
    parser.add_argument("--prompts-file", default=None, help="JSONL with system/user pairs")
    args = parser.parse_args()

    load_dotenv()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Initialize judge client
    if args.judge == "openai":
        from openai import OpenAI
        judge_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    elif args.judge == "anthropic":
        import anthropic
        judge_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    # Load prompts
    if args.prompts_file and Path(args.prompts_file).exists():
        prompts = []
        with open(args.prompts_file) as f:
            for line in f:
                prompts.append(json.loads(line.strip()))
    else:
        prompts = NPC_PROMPTS * (args.num_prompts // len(NPC_PROMPTS) + 1)
        prompts = prompts[:args.num_prompts]

    # Generate and judge
    pairs = []
    with open(args.output, "w") as fout:
        for prompt in tqdm(prompts, desc="Generating pairs"):
            sys_prompt = prompt["system"]
            user_msg = prompt["user"]

            resp_a, resp_b = generate_pair(model, tokenizer, sys_prompt, user_msg)
            choice, reason = judge_pair(
                judge_client, args.judge, sys_prompt, user_msg, resp_a, resp_b
            )

            if choice == "A":
                chosen, rejected = resp_a, resp_b
            elif choice == "B":
                chosen, rejected = resp_b, resp_a
            else:
                continue  # Skip ambiguous

            pair = {
                "prompt": f"System: {sys_prompt}\nUser: {user_msg}",
                "chosen": chosen,
                "rejected": rejected,
                "judge_reason": reason,
            }
            fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
            pairs.append(pair)
            time.sleep(0.5)  # Rate limit

    logger.info(f"✅ Generated {len(pairs)} preference pairs → {args.output}")


if __name__ == "__main__":
    main()
