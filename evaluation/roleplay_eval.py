#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Roleplay Evaluation — Side-by-side comparison across models.
Runs all models through every scenario with a 6-turn conversation.

Usage:
    python evaluation/roleplay_eval.py \
        --models google/gemma-4-12B-it outputs/Gemma4NPC-it/merged_float16 \
        --output evaluation/results/roleplay_eval.json
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Standard evaluation prompts (3 exchanges per scenario)
EVAL_EXCHANGES = [
    "Hello there.",
    "Tell me about yourself.",
    "What do you think about the current state of things?",
]


def load_scenarios(scenarios_dir: str = "evaluation/scenarios") -> dict[str, str]:
    """Load all scenario system prompts."""
    scenarios = {}
    for f in sorted(Path(scenarios_dir).glob("*.txt")):
        scenarios[f.stem] = f.read_text().strip()
    return scenarios


def run_model_eval(model_path: str, scenarios: dict[str, str]) -> dict:
    """Run all scenarios through one model."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    logger.info(f"Loading: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    results = {}
    for scenario_name, system_prompt in scenarios.items():
        logger.info(f"  Scenario: {scenario_name}")
        history = []
        turns = []

        for user_msg in EVAL_EXCHANGES:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(history)
            messages.append({"role": "user", "content": user_msg})

            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
            ).to(model.device)

            start = time.time()
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs, max_new_tokens=256,
                    temperature=1.0, top_p=0.95, top_k=64, do_sample=True,
                )
            elapsed = time.time() - start

            input_len = inputs["input_ids"].shape[-1]
            response = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
            output_tokens = outputs.shape[-1] - input_len

            turns.append({
                "user": user_msg,
                "model": response,
                "tokens": output_tokens,
                "time": round(elapsed, 3),
            })

            history.append({"role": "user", "content": user_msg})
            history.append({"role": "model", "content": response})

        results[scenario_name] = turns

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="Model paths")
    parser.add_argument("--output", default=None)
    parser.add_argument("--scenarios-dir", default="evaluation/scenarios")
    args = parser.parse_args()

    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token)

    scenarios = load_scenarios(args.scenarios_dir)
    logger.info(f"Loaded {len(scenarios)} scenarios")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"evaluation/results/roleplay_eval_{timestamp}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for model_path in args.models:
        all_results[model_path] = run_model_eval(model_path, scenarios)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print comparison table
    print("\n" + "=" * 80)
    print("ROLEPLAY EVALUATION — SIDE-BY-SIDE COMPARISON")
    print("=" * 80)

    for scenario in scenarios:
        print(f"\n📜 Scenario: {scenario}")
        print("-" * 60)
        for user_msg_idx, user_msg in enumerate(EVAL_EXCHANGES):
            print(f"\n  Player: \"{user_msg}\"")
            for model_path in args.models:
                model_name = Path(model_path).name or model_path.split("/")[-1]
                turn = all_results[model_path][scenario][user_msg_idx]
                response = turn["model"][:150]
                print(f"  [{model_name}]: {response}...")

    logger.info(f"\n✅ Full results: {output_path}")


if __name__ == "__main__":
    main()
