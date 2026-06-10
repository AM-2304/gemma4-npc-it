#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Inference Speed Benchmark — Multi-device benchmark table.
Reproduces the speed comparison from the original Gemma3NPC project.

Usage:
    python evaluation/inference_speed_benchmark.py \
        --model-path path/to/gguf/or/safetensors \
        --model-type vllm-api \
        --api-url http://10.198.63.20:8000/v1 \
        --num-trials 10 \
        --output-json results/benchmark.json
"""

import argparse
import json
import logging
import os
import platform
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BENCHMARK_SYSTEM = """You are roleplaying as Rele, a mysterious forest spirit guide. Stay in character at all times. Respond naturally, concisely, and in a way that advances the conversation.

Character Name: Rele
Character Category: Fantasy, Nature Spirit
Character Description: Rele is an ancient forest spirit who appears as a shimmering, translucent humanoid figure made of interwoven leaves and light."""

BENCHMARK_USER = "Hello there. What brings you to this part of the forest?"


def detect_hardware() -> dict:
    """Auto-detect hardware info."""
    info = {
        "cpu": platform.processor() or "Unknown",
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        else:
            info["gpu"] = "None"
            info["vram_gb"] = 0
    except ImportError:
        info["gpu"] = "Unknown"
        info["vram_gb"] = 0

    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        info["ram_gb"] = 0

    return info


def benchmark_gguf(model_path: str, num_trials: int) -> list[dict]:
    """Benchmark GGUF model."""
    from llama_cpp import Llama

    llm = Llama(
        model_path=model_path, n_gpu_layers=-1, n_ctx=4096,
        chat_format="gemma", verbose=False,
    )

    messages = [
        {"role": "system", "content": BENCHMARK_SYSTEM},
        {"role": "user", "content": BENCHMARK_USER},
    ]

    trials = []
    # Warmup
    logger.info("Running warmup...")
    llm.create_chat_completion(messages=messages, max_tokens=100, temperature=1.0, stop=["<turn|>"])

    for i in range(num_trials):
        start = time.perf_counter()
        result = llm.create_chat_completion(
            messages=messages, max_tokens=256,
            temperature=1.0, top_p=0.95, top_k=64, stop=["<turn|>"],
        )
        elapsed = time.perf_counter() - start

        usage = result.get("usage", {})
        output_tokens = usage.get("completion_tokens", 0)
        tok_sec = output_tokens / elapsed if elapsed > 0 else 0

        trials.append({
            "trial": i + 1,
            "output_tokens": output_tokens,
            "time_seconds": round(elapsed, 3),
            "tokens_per_second": round(tok_sec, 2),
        })

    return trials


def benchmark_transformers(model_path: str, num_trials: int) -> list[dict]:
    """Benchmark transformers model."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    messages = [
        {"role": "system", "content": BENCHMARK_SYSTEM},
        {"role": "user", "content": BENCHMARK_USER},
    ]

    trials = []
    # Warmup
    logger.info("Running warmup...")
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(model.device)
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=50)

    for i in range(num_trials):
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(model.device)

        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, max_new_tokens=256,
                temperature=1.0, top_p=0.95, top_k=64, do_sample=True,
            )
        elapsed = time.perf_counter() - start

        input_len = inputs["input_ids"].shape[-1]
        output_tokens = outputs.shape[-1] - input_len
        tok_sec = output_tokens / elapsed if elapsed > 0 else 0

        trials.append({
            "trial": i + 1,
            "output_tokens": int(output_tokens),
            "time_seconds": round(elapsed, 3),
            "tokens_per_second": round(tok_sec, 2),
        })

    return trials

def benchmark_vllm_api(model_name: str, num_trials: int, api_url: str) -> list[dict]:
    """Benchmark vLLM API server."""
    from openai import OpenAI
    client = OpenAI(api_key="EMPTY", base_url=api_url)

    messages = [
        {"role": "system", "content": BENCHMARK_SYSTEM},
        {"role": "user", "content": BENCHMARK_USER},
    ]

    trials = []
    logger.info(f"Running warmup against {api_url}...")
    try:
        client.chat.completions.create(model=model_name, messages=messages, max_tokens=100)
    except Exception as e:
        logger.error(f"Failed to connect to vLLM server: {e}")
        return []

    for i in range(num_trials):
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=256,
            temperature=1.0,
            top_p=0.95,
            stop=["<turn|>"]
        )
        elapsed = time.perf_counter() - start

        output_tokens = response.usage.completion_tokens if response.usage else 0
        tok_sec = output_tokens / elapsed if elapsed > 0 else 0

        trials.append({
            "trial": i + 1,
            "output_tokens": output_tokens,
            "time_seconds": round(elapsed, 3),
            "tokens_per_second": round(tok_sec, 2),
        })

    return trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-type", choices=["gguf", "transformers", "vllm-api"], required=True)
    parser.add_argument("--api-url", default="http://localhost:8000/v1", help="vLLM API URL")
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    hardware = detect_hardware()
    logger.info(f"Hardware: {hardware}")

    if args.model_type == "gguf":
        trials = benchmark_gguf(args.model_path, args.num_trials)
    elif args.model_type == "vllm-api":
        trials = benchmark_vllm_api(args.model_path, args.num_trials, args.api_url)
    else:
        trials = benchmark_transformers(args.model_path, args.num_trials)

    # Statistics
    tps_values = [t["tokens_per_second"] for t in trials]
    import statistics
    mean_tps = statistics.mean(tps_values)
    std_tps = statistics.stdev(tps_values) if len(tps_values) > 1 else 0
    mean_tokens = statistics.mean([t["output_tokens"] for t in trials])
    ttft = trials[0]["time_seconds"] * (1 / trials[0]["output_tokens"]) if trials[0]["output_tokens"] > 0 else 0

    # Print formatted table
    model_name = Path(args.model_path).stem
    print(f"\n┌{'─'*62}┐")
    print(f"│ {model_name} Benchmark{' '*(52-len(model_name))}│")
    print(f"├{'─'*12}┬{'─'*10}┬{'─'*10}┬{'─'*9}┬{'─'*17}┤")
    print(f"│ {'Device':<10} │ {'VRAM':>8} │ {'tok/sec':>8} │ {'tokens':>7} │ {'Time-to-first':>15} │")
    print(f"├{'─'*12}┼{'─'*10}┼{'─'*10}┼{'─'*9}┼{'─'*17}┤")
    gpu = hardware.get("gpu", "CPU")[:10]
    vram = f"{hardware.get('vram_gb', 0)}GB"
    print(f"│ {gpu:<10} │ {vram:>8} │ {mean_tps:>8.2f} │ {mean_tokens:>7.0f} │ {ttft:>14.3f}s │")
    print(f"└{'─'*12}┴{'─'*10}┴{'─'*10}┴{'─'*9}┴{'─'*17}┘")
    print(f"Mean: {mean_tps:.2f} ± {std_tps:.2f} tok/sec across {args.num_trials} trials")

    # Save results
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        result = {
            "model": args.model_path,
            "model_type": args.model_type,
            "hardware": hardware,
            "timestamp": datetime.now().isoformat(),
            "num_trials": args.num_trials,
            "mean_tps": round(mean_tps, 2),
            "std_tps": round(std_tps, 2),
            "ttft_seconds": round(ttft, 3),
            "trials": trials,
        }
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"✅ Results: {args.output_json}")


if __name__ == "__main__":
    main()
