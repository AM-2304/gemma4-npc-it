#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Gemma4NPC — Streaming Inference for Game Engine Integration
Designed to be spawned as a subprocess from a game engine.

Protocol:
  INPUT (stdin):  JSON line: {"system": "...", "history": [...], "message": "..."}
  OUTPUT (stdout): Token-by-token streaming, completed with <END> on its own line

Usage:
    python inference/streaming_inference.py --model-path models/Gemma4NPC-it-Q4_K_M.gguf
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,  # Keep quiet — stdout is for tokens only
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,  # Logs go to stderr, tokens go to stdout
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Streaming inference for game engines")
    parser.add_argument("--model-path", required=True, help="GGUF model path")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    from llama_cpp import Llama

    logger.info(f"Loading model: {args.model_path}")
    llm = Llama(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        chat_format="gemma",
        verbose=False,
    )
    logger.info("Model loaded. Waiting for input on stdin...")

    # Signal ready
    print("READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            print("ERROR: Invalid JSON", flush=True)
            print("<END>", flush=True)
            continue

        system_prompt = request.get("system", "You are a helpful NPC.")
        history = request.get("history", [])
        user_message = request.get("message", "")

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history[-10:])  # Sliding window
        messages.append({"role": "user", "content": user_message})

        try:
            for chunk in llm.create_chat_completion(
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=0.95,
                top_k=64,
                stop=["<turn|>"],
                stream=True,
            ):
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    print(content, end="", flush=True)
        except Exception as e:
            logger.error(f"Generation error: {e}")
            print(f"\nERROR: {e}", flush=True)

        print("\n<END>", flush=True)


if __name__ == "__main__":
    main()
