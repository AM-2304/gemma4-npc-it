#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Apple Silicon (M1/M2/M3/M4) inference for Gemma4NPC using MLX.
MLX uses unified memory (RAM = VRAM), so no VRAM restrictions.
Gemma4NPC-it in 4-bit MLX: ~7-8GB unified memory required.

Install: pip install -r requirements-mlx.txt
Usage:   python inference/mlx_inference.py --model-path path/to/mlx/model
"""

import argparse
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_mlx_inference(model_path: str, system_prompt: str, messages: list,
                      max_tokens: int = 256, temp: float = 1.0) -> str:
    """Run inference using MLX on Apple Silicon."""
    from mlx_lm import load, generate

    model, tokenizer = load(model_path)
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    prompt = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=True,
    )
    response = generate(
        model, tokenizer, prompt=prompt,
        max_tokens=max_tokens, temp=temp, top_p=0.95,
        verbose=False,
    )
    return response


def interactive_chat(model_path: str, system_prompt: str, temperature: float = 1.0):
    """Interactive chat using MLX."""
    from mlx_lm import load, generate

    logger.info(f"Loading MLX model: {model_path}")
    model, tokenizer = load(model_path)
    history = []

    print("\n🍎 Gemma4NPC MLX Chat (Apple Silicon)")
    print("Type 'quit' to exit\n")

    while True:
        try:
            user_input = input("🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() == "quit":
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})
        messages = [{"role": "system", "content": system_prompt}] + history[-10:]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        response = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=256, temp=temperature, top_p=0.95,
            verbose=False,
        )
        print(f"🤖 NPC: {response}\n")
        history.append({"role": "model", "content": response})


def main():
    parser = argparse.ArgumentParser(description="MLX inference for Apple Silicon")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--system-prompt", default="You are a helpful NPC. Stay in character.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--one-shot", default=None)
    args = parser.parse_args()

    if args.one_shot:
        response = run_mlx_inference(
            args.model_path, args.system_prompt,
            [{"role": "user", "content": args.one_shot}],
            temp=args.temperature,
        )
        print(response)
    else:
        interactive_chat(args.model_path, args.system_prompt, args.temperature)


if __name__ == "__main__":
    main()
