#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Gemma4NPC — Direct Preference Optimization (DPO) Training Script
Standard HuggingFace Implementation (Bypassing Unsloth Vision Bugs)
"""

import os
import argparse
import logging
import json
import torch
from pathlib import Path

# Monkey-patch TRANSFORMERS_CACHE for llm-blender compatibility
import transformers.utils.hub
if not hasattr(transformers.utils.hub, "TRANSFORMERS_CACHE"):
    transformers.utils.hub.TRANSFORMERS_CACHE = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-model", type=str, default="outputs/Gemma4NPC-it/merged_float16", help="Path to SFT model")
    parser.add_argument("--dpo-data", type=str, default="data/dpo/preference_pairs.jsonl", help="Path to DPO preference dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/Gemma4NPC-it-DPO", help="Output directory for DPO")
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"🚀 Starting DPO alignment run.")
    
    # 1. Load the SFT merged model using standard transformers + bitsandbytes
    logger.info(f"Loading SFT model from {args.sft_model}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model,
        quantization_config=bnb_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model)
    
    # Apply Chat Template if missing
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\\n' + message['content'] + '<end_of_turn>\\n' }}{% elif message['role'] == 'model' or message['role'] == 'assistant' %}{{ '<start_of_turn>model\\n' + message['content'] + '<end_of_turn>\\n' }}{% endif %}{% end if %}{% endfor %}"
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Wrapper to satisfy TRL 0.10.0+ for models that claim to be multimodal but are used for text-only DPO
    from transformers import PreTrainedTokenizerBase
    class DummyProcessor(PreTrainedTokenizerBase):
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
        def __call__(self, text=None, images=None, **kwargs):
            res = self.tokenizer(text=text, **kwargs)
            # Add batch dimension to match standard Processor behavior
            if isinstance(res.get("input_ids"), list) and (not res["input_ids"] or isinstance(res["input_ids"][0], int)):
                res["input_ids"] = [res["input_ids"]]
                if "attention_mask" in res:
                    res["attention_mask"] = [res["attention_mask"]]
            # Inject dummy pixel values to satisfy TRL's multimodal expectations
            res["pixel_values"] = [None]
            res["pixel_attention_mask"] = [None]
            return res
        def save_pretrained(self, *args, **kwargs):
            return self.tokenizer.save_pretrained(*args, **kwargs)
        def __getattr__(self, name):
            return getattr(self.tokenizer, name)
        def __repr__(self):
            return repr(self.tokenizer)
            
    processing_class_wrapper = DummyProcessor(tokenizer)
        
    # 2. Add LoRA Adapters for DPO
    lora_config = LoraConfig(
        r=8, 
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=True # SOTA scaling
    )
    model = get_peft_model(model, lora_config)
    
    # 3. Load Dataset
    logger.info(f"Loading DPO preference dataset from {args.dpo_data}...")
    dataset = load_dataset("json", data_files=args.dpo_data, split="train")

    def format_dpo(example):
        return {
            "prompt": [{"role": "user", "content": example["prompt"]}],
            "chosen": [{"role": "assistant", "content": example["chosen"]}],
            "rejected": [{"role": "assistant", "content": example["rejected"]}]
        }
    dataset = dataset.map(format_dpo)

    # Add dummy images column to bypass TRL KeyError
    def add_dummy_images(example):
        example["images"] = None
        return example
    dataset = dataset.map(add_dummy_images)

    # 4. DPO Trainer Setup
    logger.info("Initializing DPOTrainer...")
    
    # Monkey-patch warnings_issued for TRL 0.10.0+ compatibility with bleeding edge transformers
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    if hasattr(model, "base_model") and not hasattr(model.base_model, "warnings_issued"):
        model.base_model.warnings_issued = {}
        
    trainer = DPOTrainer(
        model=model,
        ref_model=None, 
        processing_class=processing_class_wrapper,
        train_dataset=dataset,
        args=DPOConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=5e-7, 
            num_train_epochs=1,
            beta=0.1, 
            bf16=True,
            logging_steps=5,
            save_steps=50,
            report_to="none" 
        ),
    )
    
    # 5. Train
    logger.info("Starting DPO training loop (resuming from checkpoint if available)...")
    
    # Check if a checkpoint exists
    checkpoint_dir = args.output_dir
    resume = False
    if os.path.exists(checkpoint_dir):
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint")]
        if len(checkpoints) > 0:
            # Sort by checkpoint number
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
            latest_ckpt = os.path.join(checkpoint_dir, checkpoints[-1])
            # Verify the checkpoint is fully written
            if os.path.exists(os.path.join(latest_ckpt, "trainer_state.json")):
                resume = True
            else:
                logger.warning(f"Found corrupted/incomplete checkpoint {latest_ckpt}, deleting it and ignoring resume.")
                import shutil
                shutil.rmtree(latest_ckpt, ignore_errors=True)
                resume = False

    trainer.train(resume_from_checkpoint=resume)
    
    # 6. Save and Merge
    logger.info(f"Training complete. Saving merged weights to {args.output_dir}/merged_float16...")
    # TRL DPOTrainer + PEFT requires merging
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    from peft import AutoPeftModelForCausalLM
    model = AutoPeftModelForCausalLM.from_pretrained(args.output_dir, torch_dtype=torch.float16)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(f"{args.output_dir}/merged_float16", safe_serialization=True)
    tokenizer.save_pretrained(f"{args.output_dir}/merged_float16")
    
    logger.info("🎉 DPO successfully finished!")

if __name__ == "__main__":
    main()
