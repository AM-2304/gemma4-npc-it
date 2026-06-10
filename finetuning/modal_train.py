#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Modal Training Wrapper — Run SFT/DPO training on an A100 GPU in the cloud.

Usage:
    # 1. Setup modal token locally:
    #    pip install modal
    #    modal setup
    
    # 2. Run training in the cloud:
    #    modal run finetuning/modal_train.py
"""

import modal
import os

# Define the container image with CUDA-compatible ML libraries
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake", "libcurl4-openssl-dev")
    .pip_install(
        "torch>=2.4.0",
        "accelerate>=0.34.0",
        "datasets>=2.21.0",
        "peft>=0.12.0",
        "trl>=0.10.0",
        "bitsandbytes>=0.43.0",
        "sentencepiece",
        "protobuf",
        "pyyaml",
        "python-dotenv",
        "wandb",
        "gguf>=0.6.0",
        "openai",
        "tqdm",
    )
    .run_commands(
        # Install Unsloth inside the A100 CUDA container
        "pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"",
        # Upgrade transformers to development source to support gemma4_unified model type
        "pip install --upgrade git+https://github.com/huggingface/transformers.git",
        "pip install --no-deps xformers"
    )
    .add_local_dir(
        ".",
        remote_path="/root/gemma4npc",
        ignore=["venv", "models", "outputs", ".git"]
    )
)

app = modal.App("gemma4npc-sft")

# Persistent volume to store training checkpoints and merged models
volume = modal.Volume.from_name("gemma4npc-outputs", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",  # Requires an A100 (40GB or 80GB) to train the 12B model
    timeout=86400,  # Max 24 hours runtime
    secrets=[
        modal.Secret.from_dotenv()
    ],
    volumes={"/root/gemma4npc/outputs": volume}
)
def run_sft():
    import subprocess
    import sys
    import os
    
    print("🚀 Container started. Beginning SFT training on A100 GPU...")
    
    # Pass environment variables to bypass compilation cache limit crashes
    env = os.environ.copy()
    env["TORCHDYNAMO_CACHE_SIZE_LIMIT"] = "512"
    env["TORCH_COMPILE_LIMIT"] = "512"
    env["TORCHDYNAMO_SUPPRESS_ERRORS"] = "1"
    env["UNSLOTH_COMPILE_DISABLE"] = "1"
    env["UNSLOTH_FULLGRAPH"] = "0"
    
    # Build training command
    cmd = [
        sys.executable,
        "/root/gemma4npc/finetuning/train.py",
        "--config",
        "/root/gemma4npc/configs/training_npc_it.yaml"
    ]
    
    # Forward HuggingFace model repo for auto-upload if present
    model_repo = os.environ.get("HF_MODEL_REPO")
    if model_repo:
        print(f"Adding push-to-hub destination: {model_repo}")
        cmd.extend(["--push-to-hub", "--hub-model-id", model_repo])
        
    # Run SFT script inside the remote container
    result = subprocess.run(
        cmd,
        cwd="/root/gemma4npc",
        env=env,
    )
    
    if result.returncode != 0:
        print("❌ Training crashed with error code", result.returncode)
        sys.exit(result.returncode)
        
    print("🎉 SFT Training Complete!")


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_dotenv()
    ],
    volumes={"/root/gemma4npc/outputs": volume}
)
def push_from_volume():
    from huggingface_hub import HfApi
    import os
    
    token = os.environ.get("HF_TOKEN")
    model_repo = os.environ.get("HF_MODEL_REPO")
    if not token or not model_repo:
        print("❌ HF_TOKEN and HF_MODEL_REPO must be set!")
        return
        
    print(f"🚀 Pushing model from volume to Hugging Face Hub: {model_repo}...")
    api = HfApi(token=token)
    
    # Create repo if not exists
    try:
        api.create_repo(repo_id=model_repo, exist_ok=True)
    except Exception as e:
        print(f"⚠️ Failed to create repo (it might already exist or permission issue): {e}")
        
    # Upload folder
    folder_path = "/root/gemma4npc/outputs/Gemma4NPC-it/merged_float16"
    if not os.path.exists(folder_path):
        print(f"❌ Error: {folder_path} does not exist in the persistent volume yet! Did the training run complete successfully?")
        return
        
    api.upload_folder(
        folder_path=folder_path,
        repo_id=model_repo,
        repo_type="model"
    )
    print("🎉 Successfully uploaded model weights to Hugging Face Hub!")


@app.function(
    image=image,
    timeout=14400,  # Max 4 hours runtime for quantization
    volumes={"/root/gemma4npc/outputs": volume}
)
def run_quantization():
    import subprocess
    import sys
    import os
    
    print("🚀 Starting Remote GGUF Quantization...")
    # Run the quantization script inside the Modal container
    cmd = [
        sys.executable,
        "/root/gemma4npc/quantization/export_to_gguf.py",
        "--model-path", "/root/gemma4npc/outputs/Gemma4NPC-it/merged_float16",
        "--output-dir", "/root/gemma4npc/outputs/Gemma4NPC-it/gguf",
        "--quants", "Q4_K_M", "Q8_0",
        "--skip-verify"  # Skip inference verification locally since it requires llama-cpp-python
    ]
    
    result = subprocess.run(
        cmd,
        cwd="/root/gemma4npc",
    )
    
    if result.returncode != 0:
        print("❌ Quantization crashed with error code", result.returncode)
        sys.exit(result.returncode)
        
    print("🎉 Remote GGUF Quantization Complete!")
    print("You can download the quantizations from your volume using:")
    print("  modal volume get gemma4npc-outputs /Gemma4NPC-it/gguf/Gemma4NPC-it-Q4_K_M.gguf ./models/")


@app.function(
    image=image,
    gpu="A100",
    timeout=14400,
    secrets=[
        modal.Secret.from_dotenv()
    ],
    volumes={"/root/gemma4npc/outputs": volume}
)
def run_evaluation():
    import subprocess
    import sys
    
    print("🚀 Starting Remote Evaluation (LLM-as-a-Judge)...")
    cmd = [
        sys.executable,
        "/root/gemma4npc/evaluation/llm_as_judge_eval.py",
        "--models", "/root/gemma4npc/outputs/Gemma4NPC-it/merged_float16",
        "--num-prompts", "25",
        "--output", "/root/gemma4npc/outputs/Gemma4NPC-it/evaluation_results.json"
    ]
    
    result = subprocess.run(
        cmd,
        cwd="/root/gemma4npc",
    )
    
    if result.returncode != 0:
        print("❌ Evaluation crashed with error code", result.returncode)
        sys.exit(result.returncode)
        
    print("🎉 Remote Evaluation Complete!")
    print("Results saved to your volume. Download with:")
    print("  modal volume get gemma4npc-outputs /Gemma4NPC-it/evaluation_results.json .")


@app.local_entrypoint()
def main():
    run_sft.remote()
