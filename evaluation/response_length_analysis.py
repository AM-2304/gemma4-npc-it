#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Response Length Analysis — Compares token length distributions across models.
Target: fine-tuned models should produce 30-60 token responses (not 100+ like base).

Usage:
    python evaluation/response_length_analysis.py \
        --models google/gemma-4-12B-it outputs/Gemma4NPC-it/merged_float16 \
        --test-file data/final/RolePlay-NPC-v2_test.jsonl \
        --num-samples 100
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


def measure_lengths(model_path: str, test_prompts: list, num_samples: int) -> list[int]:
    """Measure response token lengths for a model."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    lengths = []
    for prompt in tqdm(test_prompts[:num_samples], desc=Path(model_path).name):
        messages = prompt.get("messages", [])[:3]  # system + first exchange
        if len(messages) < 2:
            continue

        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs, max_new_tokens=256,
                temperature=1.0, top_p=0.95, top_k=64, do_sample=True,
            )
        output_len = outputs.shape[-1] - inputs["input_ids"].shape[-1]
        lengths.append(output_len)

    del model
    torch.cuda.empty_cache()
    return lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--test-file", default="data/final/RolePlay-NPC-v2_test.jsonl")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--output-plot", default="evaluation/plots/response_length_dist.png")
    args = parser.parse_args()

    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token)

    # Load test prompts
    test_prompts = []
    with open(args.test_file) as f:
        for line in f:
            test_prompts.append(json.loads(line.strip()))

    results = {}
    for model_path in args.models:
        logger.info(f"Measuring: {model_path}")
        lengths = measure_lengths(model_path, test_prompts, args.num_samples)
        results[model_path] = lengths

    # Print statistics
    print("\n" + "=" * 70)
    print("RESPONSE LENGTH ANALYSIS")
    print("=" * 70)
    print(f"{'Model':<40} {'Mean':>8} {'Std':>8} {'Min':>6} {'Max':>6}")
    print("-" * 70)
    for model_path, lengths in results.items():
        name = Path(model_path).name or model_path.split("/")[-1]
        if lengths:
            import statistics
            mean = statistics.mean(lengths)
            std = statistics.stdev(lengths) if len(lengths) > 1 else 0
            print(f"{name:<40} {mean:>8.1f} {std:>8.1f} {min(lengths):>6} {max(lengths):>6}")

    # Generate histogram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        Path(args.output_plot).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))

        for model_path, lengths in results.items():
            name = Path(model_path).name or model_path.split("/")[-1]
            ax.hist(lengths, bins=30, alpha=0.5, label=name)

        ax.set_xlabel("Response Length (tokens)")
        ax.set_ylabel("Count")
        ax.set_title("Gemma4NPC Response Length Distribution")
        ax.legend()
        plt.tight_layout()
        plt.savefig(args.output_plot, dpi=150)
        logger.info(f"✅ Plot saved: {args.output_plot}")
    except ImportError:
        logger.warning("matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
