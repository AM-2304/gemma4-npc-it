#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
vLLM Inference — High-throughput serving for production.
vLLM supports continuous batching and PagedAttention for maximum throughput.

Usage:
    # 1. Start vLLM server
    python -m vllm.entrypoints.openai.api_server \
        --model outputs/Gemma4NPC-it/merged_float16 \
        --served-model-name gemma4npc-it \
        --dtype bfloat16 \
        --max-model-len 8192

    # 2. Run this test script
    python inference/vllm_inference.py
"""

import logging
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    # vLLM provides an OpenAI-compatible API out of the box
    client = OpenAI(
        api_key="EMPTY",  # vLLM doesn't require an API key by default
        base_url="http://localhost:8000/v1",
    )

    system_prompt = (
        "You are Aldric, a grizzled blacksmith in his 50s. "
        "Speaks in short, direct sentences. Doesn't trust easily."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "I need a sword repaired."},
    ]

    logger.info("Sending request to vLLM server...")
    
    try:
        response = client.chat.completions.create(
            model="gemma4npc-it",
            messages=messages,
            temperature=1.0,
            top_p=0.95,
            max_tokens=150,
            stop=["<turn|>"],  # Gemma 4 EOS token
        )
        
        reply = response.choices[0].message.content
        logger.info(f"\nAldric: {reply}")
        
    except Exception as e:
        logger.error(f"Failed to connect to vLLM: {e}")
        logger.error("Make sure you started the vLLM server first! (See file header)")


if __name__ == "__main__":
    main()
