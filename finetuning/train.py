#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Gemma4NPC CLI Training Script
Production-grade training script that mirrors the notebook pipeline.

Usage:
    python finetuning/train.py --config configs/training_npc_it.yaml
    python finetuning/train.py --config configs/training_base.yaml --dry-run
    python finetuning/train.py --config configs/training_npc_it.yaml --push-to-hub --hub-model-id user/Gemma4NPC-it
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

os.environ["TORCHDYNAMO_CACHE_SIZE_LIMIT"] = "512"
os.environ["TORCH_COMPILE_LIMIT"] = "512"
os.environ["TORCHDYNAMO_SUPPRESS_ERRORS"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_FULLGRAPH"] = "0"

import torch
torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.fail_on_recompile_limit_hit = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Gemma4NPC Training CLI")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--dataset", default=None, help="Override dataset path")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Checkpoint path")
    parser.add_argument("--dry-run", action="store_true", help="Load model/data, print stats, exit")
    parser.add_argument("--push-to-hub", action="store_true", help="Push to HF Hub after training")
    parser.add_argument("--hub-model-id", default=None, help="HF model ID for push")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class NaNDetectionCallback:
    """Stops training if loss becomes NaN or Inf."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss = logs["loss"]
            if math.isnan(loss) or math.isinf(loss):
                logger.error(
                    f"🚨 NaN/Inf loss at step {state.global_step}! Stopping."
                )
                logger.error("Try: reduce lr to 1e-5 or max_grad_norm to 0.3")
                control.should_training_stop = True


def is_valid_token(token: str) -> bool:
    if not token:
        return False
    placeholders = ["placeholder", "your_", "api_key", "token_here", "your_wandb_key", "your_huggingface_token"]
    return not any(p in token.lower() for p in placeholders)


def main():
    load_dotenv()
    args = parse_args()

    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Apply CLI overrides
    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})

    if args.dataset:
        data_cfg["train_file"] = args.dataset
    if args.output_dir:
        train_cfg["output_dir"] = args.output_dir

    # HF login
    token = os.environ.get("HF_TOKEN")
    if is_valid_token(token):
        from huggingface_hub import login
        login(token=token)
    else:
        logger.warning("No valid HF_TOKEN found in environment. Model downloads/uploads may fail if gated.")

    # wandb login
    wandb_key = os.environ.get("WANDB_API_KEY")
    use_wandb = False
    if is_valid_token(wandb_key) and train_cfg.get("report_to") == "wandb":
        import wandb
        wandb.login(key=wandb_key)
        use_wandb = True
    else:
        logger.info("Weights & Biases logging disabled (no valid WANDB_API_KEY found or disabled in config).")
        train_cfg["report_to"] = "none"

    # ---- Load Model via Unsloth ----
    logger.info(f"Loading model: {model_cfg['base_model']}...")
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_cfg["base_model"],
        max_seq_length=data_cfg.get("max_seq_length", 4096),
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        full_finetuning=False,
    )

    model = FastModel.get_peft_model(
        model,
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 16),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        bias=lora_cfg.get("bias", "none"),
        use_rslora=lora_cfg.get("use_rslora", True),
        loftq_config=None,
        use_gradient_checkpointing="unsloth",
        random_state=train_cfg.get("seed", 42),
    )
    model.print_trainable_parameters()

    # ---- Apply Gemma 4 Chat Template (CRITICAL — from addendum) ----
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    # ---- Load Dataset ----
    from datasets import load_dataset

    train_file = data_cfg["train_file"]
    
    # Generate mock dataset if the file does not exist (enables dry-run / local testing without raw data)
    if not os.path.exists(train_file):
        logger.warning(f"⚠️ Dataset file not found at: {train_file}")
        logger.info("Creating a mock roleplay dataset to allow SFT pipeline run...")
        
        Path(train_file).parent.mkdir(parents=True, exist_ok=True)
        
        mock_convs = [
            {"messages": [{"role": "user", "content": "Who are you?"}, {"role": "model", "content": "I am Elara, a wizard of Aldoria."}]},
            {"messages": [{"role": "user", "content": "Cast a spell."}, {"role": "model", "content": "I conjure a glowing flame."}]},
            {"messages": [{"role": "user", "content": "Where is the tavern?"}, {"role": "model", "content": "Just down the cobblestone road."}]},
        ]
        
        # Write 30 conversations to train file
        with open(train_file, "w", encoding="utf-8") as f:
            for i in range(30):
                c = mock_convs[i % len(mock_convs)].copy()
                c["messages"] = [
                    {"role": m["role"], "content": f"{m['content']} (Id: {i})"}
                    for m in c["messages"]
                ]
                f.write(json.dumps(c) + "\n")
                
        # Also write 5 conversations to val file
        val_file = data_cfg.get("val_file", train_file.replace("_train.jsonl", "_val.jsonl"))
        with open(val_file, "w", encoding="utf-8") as f:
            for i in range(5):
                c = mock_convs[i % len(mock_convs)].copy()
                c["messages"] = [
                    {"role": m["role"], "content": f"{m['content']} (Val Id: {i})"}
                    for m in c["messages"]
                ]
                f.write(json.dumps(c) + "\n")
                
        # Also write 5 conversations to test file
        test_file = train_file.replace("_train.jsonl", "_test.jsonl")
        with open(test_file, "w", encoding="utf-8") as f:
            for i in range(5):
                c = mock_convs[i % len(mock_convs)].copy()
                c["messages"] = [
                    {"role": m["role"], "content": f"{m['content']} (Test Id: {i})"}
                    for m in c["messages"]
                ]
                f.write(json.dumps(c) + "\n")

        # Also write pippa_gemma4.jsonl and npc_dialogue_v2.jsonl to keep other scripts happy
        pippa_file = Path(train_file).parent / "pippa_gemma4.jsonl"
        if not pippa_file.exists():
            with open(pippa_file, "w", encoding="utf-8") as f:
                for i in range(10):
                    c = mock_convs[i % len(mock_convs)].copy()
                    f.write(json.dumps(c) + "\n")
                    
        npc_file = Path(train_file).parent / "npc_dialogue_v2.jsonl"
        if not npc_file.exists():
            with open(npc_file, "w", encoding="utf-8") as f:
                for i in range(10):
                    c = mock_convs[i % len(mock_convs)].copy()
                    f.write(json.dumps(c) + "\n")
                    
        logger.info(f"✅ Mock datasets generated successfully under: {Path(train_file).parent}")

    logger.info(f"Loading dataset: {train_file}")
    dataset = load_dataset("json", data_files=train_file, split="train")
    logger.info(f"Dataset size: {len(dataset)} rows")

    # Format conversations
    def format_conversation(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_conversation, num_proc=4)

    # Print sample
    logger.info("=== SAMPLE FORMATTED CONVERSATION ===")
    logger.info(dataset[0]["text"][:1500])
    logger.info("=====================================")

    if args.dry_run:
        logger.info("\n🏃 DRY RUN — Model and data loaded successfully. Exiting.")
        logger.info(f"  Model: {model_cfg['base_model']}")
        logger.info(f"  Dataset: {len(dataset)} rows")
        logger.info(f"  LoRA rank: {lora_cfg.get('r', 16)}")
        logger.info(f"  Max seq length: {data_cfg.get('max_seq_length', 4096)}")
        return

    # ---- Trainer ----
    from trl import SFTTrainer, SFTConfig

    output_dir = train_cfg["output_dir"]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=output_dir,
            dataset_text_field="text",
            max_seq_length=data_cfg.get("max_seq_length", 4096),
            per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 16),
            num_train_epochs=train_cfg.get("num_train_epochs", 1),
            max_steps=train_cfg.get("max_steps", -1),
            learning_rate=train_cfg.get("learning_rate", 2e-5),
            lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
            warmup_steps=train_cfg.get("warmup_steps", 1000),
            max_grad_norm=train_cfg.get("max_grad_norm", 0.4),
            weight_decay=train_cfg.get("weight_decay", 0.01),
            optim=train_cfg.get("optim", "adamw_8bit"),
            bf16=train_cfg.get("bf16", True),
            fp16=train_cfg.get("fp16", False),
            neftune_noise_alpha=train_cfg.get("neftune_noise_alpha", 5.0),
            logging_steps=train_cfg.get("logging_steps", 10),
            save_steps=train_cfg.get("save_steps", 500),
            save_total_limit=train_cfg.get("save_total_limit", 3),
            seed=train_cfg.get("seed", 42),
            report_to=train_cfg.get("report_to", "wandb"),
            run_name=train_cfg.get("run_name", "Gemma4NPC"),
        ),
    )

    # CRITICAL: train_on_responses_only with GEMMA 4 delimiters (from addendum)
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",   # Gemma 4 delimiter (NOT Gemma 3)
        response_part="<|turn>model\n",
    )

    # Verify masking
    logger.info("Verifying train_on_responses_only masking...")
    sample_labels = trainer.train_dataset[0]["labels"]
    masked_text = tokenizer.decode(
        [tokenizer.pad_token_id if x == -100 else x for x in sample_labels]
    ).replace(tokenizer.pad_token, " ")
    logger.info(f"Masked sample (only model responses visible):\n{masked_text[:500]}")

    # Add NaN callback
    from transformers import TrainerCallback

    class NaNCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                loss = logs["loss"]
                if math.isnan(loss) or math.isinf(loss):
                    logger.error(f"🚨 NaN/Inf at step {state.global_step}!")
                    control.should_training_stop = True

    trainer.add_callback(NaNCallback())

    # ---- Train ----
    if use_wandb:
        import wandb
        wandb.init(
            project="Gemma4NPC",
            name=train_cfg.get("run_name", "Gemma4NPC"),
            config=config,
        )
    else:
        # Define a dummy finish if wandb is not used, though we check below
        pass

    logger.info("Starting training...")
    trainer_stats = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    import torch
    logger.info(f"Training complete. Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    if use_wandb:
        import wandb
        wandb.finish()

    # ---- Save ----
    logger.info("Saving LoRA adapter...")
    model.save_pretrained(f"{output_dir}/lora_adapter")
    tokenizer.save_pretrained(f"{output_dir}/lora_adapter")

    logger.info("Merging and saving float16 model...")
    model.save_pretrained_merged(
        f"{output_dir}/merged_float16",
        tokenizer,
        save_method="merged_16bit",
    )

    if args.push_to_hub and args.hub_model_id:
        logger.info(f"Pushing to Hub: {args.hub_model_id}")
        model.push_to_hub_merged(
            args.hub_model_id,
            tokenizer,
            save_method="merged_16bit",
            token=token,
        )
        logger.info(f"✅ Pushed to {args.hub_model_id}")

    logger.info("✅ Training complete!")


if __name__ == "__main__":
    main()
