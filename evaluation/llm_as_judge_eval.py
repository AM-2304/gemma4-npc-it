#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
LLM-as-Judge Evaluation — Uses GPT-4o to score roleplay quality.
Scores on 5 dimensions: consistency, naturalness, advancement, length, knowledge.

Usage:
    python evaluation/llm_as_judge_eval.py \
        --models google/gemma-4-12B-it outputs/Gemma4NPC-it/merged_float16 \
        --num-prompts 50
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are evaluating an NPC response in a video game.

NPC Character Definition:
{system_prompt}

Player said: {user_message}

NPC responded: {npc_response}

Score this response on each dimension from 1 (very poor) to 5 (excellent):
1. Character consistency (does it match the defined persona?)
2. Response naturalness (does it sound like natural spoken dialogue?)
3. Conversation advancement (does it move the interaction forward?)
4. Length appropriateness (right length for game dialogue?)
5. In-character knowledge (only references info the character would know?)

Output ONLY valid JSON:
{{"character_consistency": X, "naturalness": X, "advancement": X, "length": X, "knowledge": X, "brief_reason": "..."}}"""


def judge_response(client, system_prompt, user_message, npc_response):
    """Get GPT-4o judgment scores."""
    prompt = JUDGE_PROMPT.format(
        system_prompt=system_prompt[:500],
        user_message=user_message,
        npc_response=npc_response,
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.warning(f"Judge error: {e}")
        return None


def generate_response(model, processor, system_prompt, user_message):
    """Generate one NPC response."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=256,
            temperature=1.0, top_p=0.95, do_sample=True,
        )
    input_len = inputs["input_ids"].shape[-1]
    return processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--output", default="evaluation/results/llm_judge.json")
    args = parser.parse_args()

    load_dotenv()
    from openai import OpenAI
    judge_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token)

    # Load scenarios
    scenarios = {}
    for f in sorted(Path("evaluation/scenarios").glob("*.txt")):
        scenarios[f.stem] = f.read_text().strip()

    test_cases = []
    for name, prompt in scenarios.items():
        for user_msg in ["Hello there.", "Tell me about yourself.", "What do you think?"]:
            test_cases.append({"scenario": name, "system": prompt, "user": user_msg})
    test_cases = test_cases[:args.num_prompts]

    all_results = {}
    for model_path in args.models:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info(f"Evaluating: {model_path}")
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
        )
        model.eval()

        model_scores = []
        for tc in tqdm(test_cases, desc=Path(model_path).name):
            response = generate_response(model, processor, tc["system"], tc["user"])
            scores = judge_response(judge_client, tc["system"], tc["user"], response)
            if scores:
                model_scores.append(scores)
            time.sleep(0.3)

        del model
        torch.cuda.empty_cache()

        # Average scores
        dims = ["character_consistency", "naturalness", "advancement", "length", "knowledge"]
        avg = {}
        for d in dims:
            vals = [s.get(d, 0) for s in model_scores if s.get(d)]
            avg[d] = round(sum(vals) / len(vals), 2) if vals else 0
        avg["total"] = round(sum(avg.values()) / len(avg), 2)
        avg["n"] = len(model_scores)
        all_results[model_path] = avg

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print table
    print("\n" + "=" * 100)
    print("LLM-as-Judge Evaluation Results (GPT-4o)")
    print("=" * 100)
    header = f"{'Model':<35} {'Consist':>8} {'Natural':>8} {'Advance':>8} {'Length':>8} {'Knowledge':>10} {'TOTAL':>8}"
    print(header)
    print("-" * 100)
    for model_path, scores in all_results.items():
        name = Path(model_path).name or model_path.split("/")[-1]
        print(f"{name:<35} {scores.get('character_consistency',0):>8.2f} "
              f"{scores.get('naturalness',0):>8.2f} {scores.get('advancement',0):>8.2f} "
              f"{scores.get('length',0):>8.2f} {scores.get('knowledge',0):>10.2f} "
              f"{scores.get('total',0):>8.2f}")

    logger.info(f"\n✅ Results: {args.output}")


if __name__ == "__main__":
    main()
