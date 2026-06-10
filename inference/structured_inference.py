#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Gemma4NPC — Structured JSON Output Inference
Prompts the model to output structured JSON for game engine consumption.

Output format:
{
  "dialogue": "What the NPC says",
  "emotion": "neutral|happy|angry|sad|fearful|disgusted|surprised",
  "action": "none|turn_away|hand_item|point_direction|bow|wave",
  "quest_triggered": null or "quest_id"
}

Usage:
    python inference/structured_inference.py \
        --model-path models/Gemma4NPC-it-Q4_K_M.gguf \
        --system-prompt "You are Aldric, a grizzled blacksmith..."
"""

import argparse
import json
import logging
import re
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

STRUCTURED_SYSTEM_ADDENDUM = """
IMPORTANT OUTPUT FORMAT: You must respond in this exact JSON structure:
{
  "dialogue": "What you say to the player",
  "emotion": "one of: neutral, happy, angry, sad, fearful, disgusted, surprised",
  "action": "optional: one of: none, turn_away, hand_item, point_direction, bow, wave",
  "quest_triggered": "optional: quest ID string if you're giving a quest, else null"
}
Respond ONLY with valid JSON. No other text."""

VALID_EMOTIONS = {"neutral", "happy", "angry", "sad", "fearful", "disgusted", "surprised"}
VALID_ACTIONS = {"none", "turn_away", "hand_item", "point_direction", "bow", "wave"}


def parse_structured_response(raw_output: str) -> dict:
    """Parse model's JSON response, with fallback for malformed output."""
    try:
        cleaned = re.sub(r'```json|```', '', raw_output).strip()
        data = json.loads(cleaned)
        # Validate fields
        data["emotion"] = data.get("emotion", "neutral")
        if data["emotion"] not in VALID_EMOTIONS:
            data["emotion"] = "neutral"
        data["action"] = data.get("action", "none")
        if data["action"] not in VALID_ACTIONS:
            data["action"] = "none"
        data["quest_triggered"] = data.get("quest_triggered", None)
        return data
    except (json.JSONDecodeError, KeyError, TypeError):
        return {
            "dialogue": raw_output.strip(),
            "emotion": "neutral",
            "action": "none",
            "quest_triggered": None,
        }


def main():
    parser = argparse.ArgumentParser(description="Structured JSON inference")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--system-prompt", default="You are a helpful NPC.")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    from llama_cpp import Llama

    llm = Llama(
        model_path=args.model_path,
        n_gpu_layers=-1, n_ctx=4096,
        chat_format="gemma", verbose=False,
    )

    full_system = args.system_prompt + "\n" + STRUCTURED_SYSTEM_ADDENDUM
    history = []

    print("🎮 Structured NPC Chat (JSON output mode)")
    print("Type 'quit' to exit\n")

    while True:
        try:
            user_input = input("Player: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() == "quit":
            break

        messages = [{"role": "system", "content": full_system}]
        messages.extend(history[-10:])
        messages.append({"role": "user", "content": user_input})

        result = llm.create_chat_completion(
            messages=messages,
            max_tokens=300,
            temperature=args.temperature,
            stop=["<turn|>"],
        )

        raw = result["choices"][0]["message"]["content"]
        structured = parse_structured_response(raw)

        print(f"\n📦 Structured Output:")
        print(json.dumps(structured, indent=2))
        print(f"\n💬 {structured['dialogue']}")
        print(f"😊 Emotion: {structured['emotion']}")
        print(f"🎬 Action: {structured['action']}")
        if structured["quest_triggered"]:
            print(f"⚔️ Quest: {structured['quest_triggered']}")
        print()

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": structured["dialogue"]})


if __name__ == "__main__":
    main()
