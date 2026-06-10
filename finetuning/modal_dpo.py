#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Modal DPO Wrapper — Run Direct Preference Optimization on an A100 GPU in the cloud.

Usage:
    modal run finetuning/modal_dpo.py
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake", "libcurl4-openssl-dev")
    .pip_install(
        "torch>=2.4.0",
        "accelerate>=0.34.0",
        "datasets>=2.21.0",
        "peft>=0.12.0",
        "trl>=0.10.0",
        "mergekit",
        "llm-blender",
        "weave",
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
        "pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"",
        "pip install --upgrade git+https://github.com/huggingface/transformers.git",
        "pip install --no-deps xformers"
    )
    .add_local_dir(
        ".",
        remote_path="/root/gemma4npc",
        ignore=["venv", "models", "outputs", ".git", "llama.cpp", "llama.cpp/.git"]
    )
)

app = modal.App("gemma4npc-dpo")

# Crucial: We map the exact same volume used by modal_train.py
# This ensures train_dpo.py can instantly read the SFT weights that were just produced.
volume = modal.Volume.from_name("gemma4npc-outputs", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100",  # Requires an A100 to handle DPO for the 12B model
    timeout=86400,
    secrets=[
        modal.Secret.from_dotenv()
    ],
    volumes={"/root/gemma4npc/outputs": volume}
)
def run_dpo():
    import subprocess
    import sys
    import os
    
    print("🚀 Container started. Beginning DPO alignment on A100 GPU...")
    
    # Environment variables to bypass compilation limits
    env = os.environ.copy()
    env["TORCHDYNAMO_CACHE_SIZE_LIMIT"] = "512"
    env["TORCH_COMPILE_LIMIT"] = "512"
    env["TORCHDYNAMO_SUPPRESS_ERRORS"] = "1"
    env["UNSLOTH_COMPILE_DISABLE"] = "1"
    env["UNSLOTH_FULLGRAPH"] = "0"
    
    # Run the DPO training script
    cmd = [
        sys.executable,
        "/root/gemma4npc/finetuning/train_dpo.py"
    ]
    
    result = subprocess.run(
        cmd,
        cwd="/root/gemma4npc",
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    
    if result.returncode == 0:
        print("✅ DPO completed successfully! Model merged to volume.")
        # Commit the volume so changes are persisted back to Modal storage
        volume.commit()
    else:
        print("❌ DPO failed.")
        sys.exit(result.returncode)
