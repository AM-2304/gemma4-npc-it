#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Goblin Merchant Backend — Demonstrates structured JSON output controlling game state.

Usage:
    python demos/goblin_merchant/npc_backend.py --model-path models/Gemma4NPC-it-Q4_K_M.gguf
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Gribble, a notoriously greedy, eccentric, and somewhat paranoid goblin merchant in a fantasy bazaar. 
You are currently selling a "Glowing Amulet of Unfathomable Power" (which might just be a rock painted with glowing mushrooms).
The player wants to buy it. You initially want 500 gold for it. The player only has 50 gold.

The player might try to haggle, barter with strange items in their inventory, flatter you, or threaten you.
You must react in character! You love shiny things, hate being insulted, and are terrified of city guards.

CRITICAL INSTRUCTION: You MUST output your response EXACTLY as a valid JSON object. Do not wrap it in markdown code blocks. Do not add any conversational text outside the JSON.
The JSON must have this exact schema:
{
  "dialogue": "Your in-character spoken dialogue here. Be expressive and goblin-like!",
  "current_price": <integer, your current asking price. Update this based on the negotiation.>,
  "merchant_mood": "<one of: greedy, angry, terrified, amused, insulted, intrigued>",
  "deal_accepted": <boolean, set to true ONLY if you agree to sell it for the player's current offer>
}"""


class NPCBackend:
    def __init__(self, model_path: str):
        try:
            from llama_cpp import Llama
        except ImportError:
            logger.error("llama-cpp-python not installed. Run: pip install llama-cpp-python")
            exit(1)

        logger.info(f"Loading NPC model: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=4096,
            chat_format="gemma",
            verbose=False,
        )
        self.conversation_history = []
        logger.info("NPC model loaded — Gribble is ready to haggle! 💰")

    def get_response(self, player_message: str) -> dict:
        """Generate NPC response and parse it as JSON to drive game state."""
        # We inject the schema reminder into the final user message to ensure compliance
        schema_reminder = "\n\n(Remember: Respond ONLY with the valid JSON object containing dialogue, current_price, merchant_mood, and deal_accepted)"
        
        # Sliding window history
        history = self.conversation_history[-10:]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": player_message + schema_reminder},
        ]

        result = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=250,
            temperature=0.7, # Slightly lower temp for better JSON adherence
            top_p=0.95,
            stop=["<turn|>"],
            response_format={"type": "json_object"}, # Force JSON mode if supported by backend
        )
        
        raw_response = result["choices"][0]["message"]["content"].strip()
        
        # Strip markdown if the model hallucinated it
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:]
        if raw_response.startswith("```"):
            raw_response = raw_response[3:]
        if raw_response.endswith("```"):
            raw_response = raw_response[:-3]
            
        raw_response = raw_response.strip()

        try:
            parsed_data = json.loads(raw_response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from model: {raw_response}")
            # Fallback state if model fails to output JSON
            parsed_data = {
                "dialogue": "*Gribble mutters incoherently and ignores you.* (Error: Model failed to output JSON)",
                "current_price": 500,
                "merchant_mood": "confused",
                "deal_accepted": False
            }

        # Save to history (save the raw JSON string so the model remembers its state)
        self.conversation_history.append({"role": "user", "content": player_message})
        self.conversation_history.append({"role": "assistant", "content": json.dumps(parsed_data)})

        return parsed_data

def main():
    """Interactive CLI test mode for the NPC backend."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to GGUF model")
    args = parser.parse_args()

    backend = NPCBackend(args.model_path)

    print("\n💰 You approach Gribble's stall. He's selling a Glowing Amulet for 500 gold.")
    print("Type messages to haggle with Gribble (type 'quit' to exit)\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() == "quit":
            break
        if not user_input:
            continue

        response_data = backend.get_response(user_input)
        
        print(f"\n[Mood: {response_data.get('merchant_mood', 'greedy').upper()}]")
        print(f"[Price: {response_data.get('current_price', 500)} Gold]")
        print(f"Gribble: \"{response_data.get('dialogue', '')}\"")
        
        if response_data.get('deal_accepted', False):
            print("\n🎉 DEAL ACCEPTED! You bought the amulet!")
            break


if __name__ == "__main__":
    main()
