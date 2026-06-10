#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Gemma4NPC — HuggingFace Transformers Inference
Supports text-only, image+text, and audio+text multimodal input.

IMPORTANT: Uses AutoModelForImageTextToText + AutoProcessor (NOT AutoModelForCausalLM).
Gemma 4 12B is encoder-free — all modalities feed directly into the unified decoder.

Usage:
    python inference/transformers_inference.py --model-path google/gemma-4-12B-it
    python inference/transformers_inference.py --model-path outputs/Gemma4NPC-it/merged_float16 \
        --system-prompt-file evaluation/scenarios/medieval_innkeeper.txt
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def strip_thinking_block(response: str) -> str:
    """Remove <think>...</think> blocks from model output before storing in history."""
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    return cleaned if cleaned else response


def load_model(model_path: str, device_map: str = "auto"):
    """Load Gemma 4 model and processor."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    logger.info(f"Loading model from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    model.eval()
    logger.info(f"Model loaded. Device: {model.device}")
    return model, processor


def generate_response(
    model,
    processor,
    system_prompt: str,
    conversation_history: list[dict],
    user_message: str,
    image_path: str | None = None,
    audio_path: str | None = None,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 64,
    thinking_mode: bool = False,
) -> str:
    """
    Generate a response from the Gemma4NPC model.

    Args:
        model: The loaded model
        processor: The AutoProcessor
        system_prompt: NPC character definition
        conversation_history: Previous turns [{"role": "user"|"model", "content": ...}]
        user_message: Current user input
        image_path: Optional path to image for multimodal input
        audio_path: Optional path to audio file for multimodal input
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling
        top_k: Top-k sampling
        thinking_mode: Enable Gemma 4 thinking mode (adds latency, not for NPC use)

    Returns:
        Generated response text
    """
    # Build messages list
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)

    # Build current user message content
    user_content = []

    if image_path:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        user_content.append({"type": "image", "image": img})  # Image BEFORE text (Gemma 4 rule)

    if audio_path:
        import soundfile as sf
        audio_array, sample_rate = sf.read(audio_path)
        user_content.append({"type": "audio", "audio": audio_array})

    if user_content:
        user_content.append({"type": "text", "text": user_message})
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": user_message})

    # Apply chat template
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    # Generate
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
        )

    # Decode only new tokens
    input_len = inputs["input_ids"].shape[-1]
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    # Strip thinking blocks if present
    response = strip_thinking_block(response)

    return response.strip()


def add_turn(history: list[dict], role: str, content: str, max_turns: int = 10) -> list[dict]:
    """Add a turn to history with sliding window."""
    history.append({"role": role, "content": content})
    if len(history) > max_turns:
        history = history[-max_turns:]
    return history


def interactive_chat(model_path: str, system_prompt: str, temperature: float = 1.0):
    """Interactive REPL for testing NPC conversations."""
    model, processor = load_model(model_path)
    history = []

    print("\n" + "=" * 60)
    print("🎮 Gemma4NPC Interactive Chat")
    print(f"Model: {model_path}")
    print(f"Temperature: {temperature}")
    print("Type 'quit' to exit, 'reset' to clear history")
    print("=" * 60)
    print(f"\n📜 System Prompt:\n{system_prompt[:200]}...\n")

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

        response = generate_response(
            model, processor,
            system_prompt=system_prompt,
            conversation_history=history,
            user_message=user_input,
            temperature=temperature,
        )

        print(f"\n🤖 NPC: {response}")

        # Update history (strip thinking before storing)
        history = add_turn(history, "user", user_input)
        history = add_turn(history, "model", strip_thinking_block(response))


def main():
    parser = argparse.ArgumentParser(description="Gemma4NPC Transformers Inference")
    parser.add_argument("--model-path", required=True, help="Model path or HF ID")
    parser.add_argument("--system-prompt-file", default=None, help="File with system prompt")
    parser.add_argument("--system-prompt", default=None, help="Direct system prompt string")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--image", default=None, help="Image path for multimodal test")
    parser.add_argument("--audio", default=None, help="Audio path for multimodal test")
    parser.add_argument("--one-shot", default=None, help="Single message (non-interactive)")
    args = parser.parse_args()

    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token)

    # Get system prompt
    if args.system_prompt_file:
        with open(args.system_prompt_file) as f:
            system_prompt = f.read().strip()
    elif args.system_prompt:
        system_prompt = args.system_prompt
    else:
        system_prompt = (
            "You are a helpful NPC in a fantasy RPG. Stay in character. "
            "Respond concisely and naturally."
        )

    if args.one_shot:
        model, processor = load_model(args.model_path)
        response = generate_response(
            model, processor,
            system_prompt=system_prompt,
            conversation_history=[],
            user_message=args.one_shot,
            image_path=args.image,
            audio_path=args.audio,
            temperature=args.temperature,
        )
        print(response)
    else:
        interactive_chat(args.model_path, system_prompt, args.temperature)


if __name__ == "__main__":
    main()
