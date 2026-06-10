#!/usr/bin/env python3
import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake", "libcurl4-openssl-dev", "libssl-dev", "curl")
    .pip_install(
        "torch>=2.4.0",
        "accelerate>=0.34.0",
        "datasets>=2.21.0",
        "peft>=0.12.0",
        "trl>=0.10.0",
        "sentencepiece",
        "protobuf",
        "gguf>=0.6.0"
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

app = modal.App("gemma4npc-quantize")
volume = modal.Volume.from_name("gemma4npc-outputs")

@app.function(
    image=image,
    gpu="A100",  # We need VRAM to load the 24GB model for quantization
    timeout=86400,
    secrets=[modal.Secret.from_dotenv()],
    volumes={"/root/gemma4npc/outputs": volume}
)
def run_quantization():
    import subprocess
    import sys
    import shutil
    
    print("🚀 Starting Cloud Quantization...")
    
    # Write a complete preprocessor_config.json with vision fields required by Unsloth QAT
    import json
    config_data = {
        "image_size": [896, 896],
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711]
    }
    dest_config = "/root/gemma4npc/outputs/Gemma4NPC-it-DPO/merged_float16/preprocessor_config.json"
    with open(dest_config, "w") as f:
        json.dump(config_data, f)
    print("Created preprocessor_config.json with image_mean")
    
    cmd = [
        sys.executable,
        "/root/gemma4npc/quantization/export_qat.py",
        "--model-path", "/root/gemma4npc/outputs/Gemma4NPC-it-DPO/merged_float16",
        "--output-dir", "/root/gemma4npc/outputs/Gemma4NPC-it-DPO-GGUF"
    ]
    
    result = subprocess.run(
        cmd,
        cwd="/root/gemma4npc",
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    
    if result.returncode == 0:
        print("✅ Quantization completed! File written to volume.")
        volume.commit()
    else:
        print("❌ Quantization failed.")
        sys.exit(result.returncode)

if __name__ == "__main__":
    import sys
    sys.argv.append("run_quantization")
    # This script is meant to be run via `modal run quantization/modal_quantize.py`
