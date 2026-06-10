#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Gemma4NPC — Batch Inference
Runs model inference on a batch of prompts for evaluation.

Usage:
    python inference/batch_inference.py \
        --model-path outputs/Gemma4NPC-it/merged_float16 \
        --prompts-file evaluation/test_prompts.jsonl \
        --output-file evaluation/results/batch_output.jsonl
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


def main():
    parser = argparse.ArgumentParser(description="Batch inference")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    logger.info(f"Loading model: {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    # Load prompts
    prompts = []
    with open(args.prompts_file) as f:
        for line in f:
            prompts.append(json.loads(line.strip()))
    logger.info(f"Loaded {len(prompts)} prompts")

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    results = []

    with open(args.output_file, "w") as fout:
        for prompt in tqdm(prompts, desc="Generating"):
            messages = prompt.get("messages", [
                {"role": "system", "content": prompt.get("system_prompt", "")},
                {"role": "user", "content": prompt.get("user_message", "Hello")},
            ])

            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
            ).to(model.device)

            start = time.time()
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=0.95, top_k=64,
                    do_sample=True,
                )
            elapsed = time.time() - start

            input_len = inputs["input_ids"].shape[-1]
            output_len = outputs.shape[-1] - input_len
            response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

            result = {
                "prompt": prompt,
                "response": response.strip(),
                "output_tokens": output_len,
                "time_seconds": round(elapsed, 3),
                "tokens_per_second": round(output_len / elapsed, 2) if elapsed > 0 else 0,
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            results.append(result)

    avg_tps = sum(r["tokens_per_second"] for r in results) / len(results) if results else 0
    avg_tokens = sum(r["output_tokens"] for r in results) / len(results) if results else 0
    logger.info(f"Avg tok/sec: {avg_tps:.1f} | Avg tokens: {avg_tokens:.0f}")
    logger.info(f"✅ Output: {args.output_file}")


if __name__ == "__main__":
    main()
