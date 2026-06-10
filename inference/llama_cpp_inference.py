#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Gemma4NPC — llama.cpp Inference (GGUF models)
For consumer GPU deployment using llama-cpp-python.

Usage:
    python inference/llama_cpp_inference.py --model-path models/Gemma4NPC-it-Q4_K_M.gguf
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_gguf_model(model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1):
    """Load a GGUF model with llama-cpp-python."""
    from llama_cpp import Llama

    logger.info(f"Loading GGUF: {model_path}")
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        chat_format="gemma",
        verbose=False,
    )
    logger.info("GGUF model loaded")
    return llm


def generate_response(
    llm,
    system_prompt: str,
    conversation_history: list[dict],
    user_message: str,
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 64,
    stream: bool = False,
) -> str:
    """Generate a response using llama-cpp-python."""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    if stream:
        response_text = ""
        for chunk in llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=["<turn|>"],
            stream=True,
        ):
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                response_text += content
                print(content, end="", flush=True)
        print()
        return response_text.strip()
    else:
        result = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=["<turn|>"],
        )
        return result["choices"][0]["message"]["content"].strip()


def add_turn(history: list[dict], role: str, content: str, max_turns: int = 10) -> list[dict]:
    """Sliding window history management."""
    history.append({"role": role, "content": content})
    if len(history) > max_turns:
        history = history[-max_turns:]
    return history


def interactive_chat(model_path: str, system_prompt: str, temperature: float = 1.0):
    """Interactive REPL for GGUF models."""
    llm = load_gguf_model(model_path)
    history = []

    print("\n" + "=" * 60)
    print("🎮 Gemma4NPC GGUF Chat")
    print(f"Model: {model_path}")
    print("Type 'quit' to exit, 'reset' to clear")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() == "quit":
            break
        elif user_input.lower() == "reset":
            history = []
            print("🔄 History cleared")
            continue
        elif not user_input:
            continue

        print("\n🤖 NPC: ", end="", flush=True)
        response = generate_response(
            llm, system_prompt, history, user_input,
            temperature=temperature, stream=True,
        )

        history = add_turn(history, "user", user_input)
        history = add_turn(history, "assistant", response)


def main():
    parser = argparse.ArgumentParser(description="Gemma4NPC GGUF Inference")
    parser.add_argument("--model-path", required=True, help="Path to GGUF file")
    parser.add_argument("--system-prompt-file", default=None)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--one-shot", default=None)
    args = parser.parse_args()

    if args.system_prompt_file:
        with open(args.system_prompt_file) as f:
            system_prompt = f.read().strip()
    elif args.system_prompt:
        system_prompt = args.system_prompt
    else:
        system_prompt = "You are a helpful NPC. Stay in character. Respond concisely."

    if args.one_shot:
        llm = load_gguf_model(args.model_path, n_ctx=args.n_ctx)
        response = generate_response(llm, system_prompt, [], args.one_shot,
                                     temperature=args.temperature)
        print(response)
    else:
        interactive_chat(args.model_path, system_prompt, args.temperature)


if __name__ == "__main__":
    main()
