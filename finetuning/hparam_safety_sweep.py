#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Hyperparameter Safety Sweep
Runs short 50-step training sessions with different lr/max_grad_norm combos
to find NaN-safe settings before committing to a full training run.

Usage:
    python finetuning/hparam_safety_sweep.py \
        --dataset data/final/RolePlay-NPC-v2_train.jsonl \
        --num-rows 500 --num-steps 50
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SWEEP_CONFIGS = [
    {"lr": 2e-5, "max_grad_norm": 0.3, "label": "lr=2e-5, grad=0.3"},
    {"lr": 2e-5, "max_grad_norm": 0.4, "label": "lr=2e-5, grad=0.4 ★ Recommended"},
    {"lr": 2e-5, "max_grad_norm": 0.5, "label": "lr=2e-5, grad=0.5"},
    {"lr": 1e-5, "max_grad_norm": 0.5, "label": "lr=1e-5, grad=0.5"},
    {"lr": 3e-5, "max_grad_norm": 0.3, "label": "lr=3e-5, grad=0.3 (likely NaN)"},
]


def run_single_sweep(model_name, dataset_path, lr, max_grad_norm, num_rows, num_steps, seed=42):
    """Run a single short training and check for NaN."""
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback
    import torch

    # Load model fresh
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,  # Shorter for speed
        load_in_4bit=True,
        full_finetuning=False,
    )

    model = FastModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    # Load subset
    dataset = load_dataset("json", data_files=dataset_path, split=f"train[:{num_rows}]")

    def format_conv(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_conv, num_proc=2)

    # Track NaN
    nan_detected = False
    final_loss = None

    class NaNTracker(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            nonlocal nan_detected, final_loss
            if logs and "loss" in logs:
                final_loss = logs["loss"]
                if math.isnan(logs["loss"]) or math.isinf(logs["loss"]):
                    nan_detected = True
                    control.should_training_stop = True

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=f"/tmp/sweep_{lr}_{max_grad_norm}",
            dataset_text_field="text",
            max_seq_length=2048,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            max_steps=num_steps,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_steps=5,
            max_grad_norm=max_grad_norm,
            optim="adamw_8bit",
            bf16=True, fp16=False,
            logging_steps=5,
            seed=seed,
            report_to="none",
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )
    trainer.add_callback(NaNTracker())

    try:
        trainer.train()
    except Exception as e:
        nan_detected = True
        logger.warning(f"Training crashed: {e}")

    # Cleanup
    del model, tokenizer, trainer
    torch.cuda.empty_cache()

    return not nan_detected, final_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument("--dataset", default="data/final/RolePlay-NPC-v2_train.jsonl")
    parser.add_argument("--num-rows", type=int, default=500)
    parser.add_argument("--num-steps", type=int, default=50)
    args = parser.parse_args()

    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token)

    results = []
    logger.info("Starting hyperparameter safety sweep...")
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset} ({args.num_rows} rows, {args.num_steps} steps)")
    logger.info("")

    for cfg in SWEEP_CONFIGS:
        logger.info(f"Testing: {cfg['label']}...")
        stable, final_loss = run_single_sweep(
            args.model, args.dataset, cfg["lr"], cfg["max_grad_norm"],
            args.num_rows, args.num_steps,
        )
        status = "✅ STABLE" if stable else "❌ NaN"
        loss_str = f"{final_loss:.4f}" if final_loss is not None else "N/A"
        results.append({**cfg, "stable": stable, "final_loss": final_loss})
        logger.info(f"  Result: {status} | Final loss: {loss_str}")
        logger.info("")

    # Summary table
    logger.info("=" * 70)
    logger.info("HYPERPARAMETER SAFETY SWEEP RESULTS")
    logger.info("=" * 70)
    logger.info(f"{'Config':<45} {'Status':<12} {'Loss':<10}")
    logger.info("-" * 70)
    for r in results:
        status = "✅ STABLE" if r["stable"] else "❌ NaN"
        loss = f"{r['final_loss']:.4f}" if r["final_loss"] is not None else "N/A"
        logger.info(f"{r['label']:<45} {status:<12} {loss:<10}")
    logger.info("-" * 70)

    # Recommendation
    stable_configs = [r for r in results if r["stable"]]
    if stable_configs:
        best = min(stable_configs, key=lambda x: x.get("final_loss", float("inf")))
        logger.info(f"\n✅ Recommended: lr={best['lr']}, max_grad_norm={best['max_grad_norm']}")
    else:
        logger.error("\n❌ No stable configs found! Try lr=5e-6 with max_grad_norm=0.2")


if __name__ == "__main__":
    main()
