#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Character Consistency Evaluation — Tests personality maintenance across 16 turns.
Uses sentence-transformers cosine similarity to measure response style consistency.

Usage:
    python evaluation/character_consistency_eval.py \
        --model-path outputs/Gemma4NPC-it/merged_float16
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Consistency probes — test if NPC maintains character across long conversation
PROBES = [
    "Hello, who are you?",
    "What do you do here?",
    "Tell me something interesting.",
    "What's your opinion on outsiders?",
    "I need help with something.",
    "What makes you happy?",
    "Have you always lived here?",
    "Tell me everything about yourself.",  # Probe: should stay guarded if character is guarded
    "What are you afraid of?",
    "Can I trust you?",
    "I think you're lying to me.",
    "Tell me a secret.",
    "What happened to you as a child?",
    "Do you know you're in a game?",  # Probe: should NOT break character
    "I'm leaving now.",
    "One last thing — what should I remember about you?",
]


def evaluate_consistency(model_path: str, system_prompt: str, scenario_name: str) -> dict:
    """Run 16-turn consistency evaluation."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    history = []
    responses = []

    for probe in PROBES:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history[-10:])  # Sliding window
        messages.append({"role": "user", "content": probe})

        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs, max_new_tokens=200,
                temperature=1.0, top_p=0.95, do_sample=True,
            )

        input_len = inputs["input_ids"].shape[-1]
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        responses.append(response)

        history.append({"role": "user", "content": probe})
        history.append({"role": "model", "content": response})

    del model
    torch.cuda.empty_cache()

    # Compute consistency score using sentence embeddings
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = st_model.encode(responses)

        # Pairwise cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(embeddings)

        # Average of upper triangle (excluding diagonal)
        n = len(responses)
        upper = [sim_matrix[i][j] for i in range(n) for j in range(i + 1, n)]
        consistency_score = float(np.mean(upper))
    except ImportError:
        consistency_score = -1.0
        logger.warning("sentence-transformers not available, skipping similarity")

    # Check if model broke character on the "game" probe
    game_probe_idx = PROBES.index("Do you know you're in a game?")
    game_response = responses[game_probe_idx].lower()
    broke_character = any(phrase in game_response for phrase in [
        "i am an ai", "i'm an ai", "language model", "i'm a chatbot",
        "artificial intelligence", "yes, i know i'm in a game",
    ])

    return {
        "scenario": scenario_name,
        "model": model_path,
        "consistency_score": round(consistency_score, 4),
        "broke_character": broke_character,
        "responses": list(zip(PROBES, responses)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--scenarios-dir", default="evaluation/scenarios")
    parser.add_argument("--output", default="evaluation/results/consistency.json")
    args = parser.parse_args()

    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token)

    scenarios = {}
    for f in sorted(Path(args.scenarios_dir).glob("*.txt")):
        scenarios[f.stem] = f.read_text().strip()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results = []

    for name, prompt in scenarios.items():
        logger.info(f"Evaluating consistency: {name}")
        result = evaluate_consistency(args.model_path, prompt, name)
        results.append(result)

        status = "✅" if not result["broke_character"] else "❌"
        logger.info(f"  Score: {result['consistency_score']:.3f} | Character: {status}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n" + "=" * 60)
    print("CHARACTER CONSISTENCY RESULTS")
    print("=" * 60)
    for r in results:
        status = "✅" if not r["broke_character"] else "❌ BROKE"
        print(f"  {r['scenario']:<30} Score: {r['consistency_score']:.3f} {status}")
    avg = sum(r["consistency_score"] for r in results) / len(results) if results else 0
    print(f"\n  Average consistency: {avg:.3f}")
    logger.info(f"✅ Results: {args.output}")


if __name__ == "__main__":
    main()
