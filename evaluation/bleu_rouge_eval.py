#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
BLEU/ROUGE Evaluation — Automated metrics against reference responses.

Usage:
    python evaluation/bleu_rouge_eval.py \
        --model-path outputs/Gemma4NPC-it/merged_float16 \
        --test-file data/final/RolePlay-NPC-v2_test.jsonl \
        --num-samples 100
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--test-file", default="data/final/RolePlay-NPC-v2_test.jsonl")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--output", default="evaluation/results/bleu_rouge.json")
    args = parser.parse_args()

    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    # Load test data
    test_data = []
    with open(args.test_file) as f:
        for line in f:
            test_data.append(json.loads(line.strip()))

    predictions = []
    references = []

    for row in tqdm(test_data[:args.num_samples], desc="Generating"):
        msgs = row["messages"]
        # Find first model response as reference
        ref_idx = None
        for i, m in enumerate(msgs):
            if m["role"] == "model" and i > 0:
                ref_idx = i
                break
        if ref_idx is None:
            continue

        # Generate with context up to the reference
        context = msgs[:ref_idx]
        if not context:
            continue

        inputs = processor.apply_chat_template(
            context, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs, max_new_tokens=256,
                temperature=1.0, top_p=0.95, do_sample=True,
            )

        input_len = inputs["input_ids"].shape[-1]
        pred = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        ref = msgs[ref_idx]["content"]

        predictions.append(pred)
        references.append(ref)

    del model
    torch.cuda.empty_cache()

    # Compute BLEU
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1
    refs_tokenized = [[r.split()] for r in references]
    preds_tokenized = [p.split() for p in predictions]
    bleu = corpus_bleu(refs_tokenized, preds_tokenized, smoothing_function=smooth)

    # Compute ROUGE
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)

    rouge_avg = {k: sum(v) / len(v) if v else 0 for k, v in rouge_scores.items()}

    results = {
        "model": args.model_path,
        "num_samples": len(predictions),
        "bleu": round(bleu, 4),
        "rouge1": round(rouge_avg["rouge1"], 4),
        "rouge2": round(rouge_avg["rouge2"], 4),
        "rougeL": round(rouge_avg["rougeL"], 4),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBLEU: {results['bleu']:.4f}")
    print(f"ROUGE-1: {results['rouge1']:.4f}")
    print(f"ROUGE-2: {results['rouge2']:.4f}")
    print(f"ROUGE-L: {results['rougeL']:.4f}")
    logger.info(f"✅ Results: {args.output}")


if __name__ == "__main__":
    main()
