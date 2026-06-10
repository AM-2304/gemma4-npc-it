#!/usr/bin/env python3
import modal
import os
import argparse

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub")
)

app = modal.App("gemma4npc-hf-push")
volume = modal.Volume.from_name("gemma4npc-outputs")

@app.function(
    image=image,
    secrets=[modal.Secret.from_dotenv()],
    volumes={"/root/gemma4npc/outputs": volume},
    timeout=86400,
)
def push_to_hf(repo_id: str):
    from huggingface_hub import HfApi
    import os
    
    api = HfApi()
    
    print(f"Creating/Verifying repository {repo_id}...")
    api.create_repo(repo_id=repo_id, exist_ok=True, private=True)
    
    # 1. Upload the Unquantized merged_float16
    float16_path = "/root/gemma4npc/outputs/Gemma4NPC-it-DPO/merged_float16"
    if os.path.exists(float16_path):
        print(f"Uploading Unquantized Float16 DPO model from {float16_path} to {repo_id}...")
        api.upload_folder(
            folder_path=float16_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload Unquantized DPO Float16 weights"
        )
        print("✅ Float16 upload complete!")
    else:
        print(f"❌ Could not find Float16 weights at {float16_path}")
        
    # 2. Upload the GGUF file
    gguf_dir = "/root/gemma4npc/outputs/Gemma4NPC-it-DPO/merged_float16_gguf"
    if os.path.exists(gguf_dir):
        # Find the .gguf file inside
        gguf_files = [f for f in os.listdir(gguf_dir) if f.endswith(".gguf")]
        for gguf_file in gguf_files:
            file_path = os.path.join(gguf_dir, gguf_file)
            print(f"Uploading Quantized GGUF model from {file_path} to {repo_id}...")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=gguf_file,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload GGUF Quantized model: {gguf_file}"
            )
        print("✅ GGUF upload complete!")
    else:
        print(f"❌ Could not find GGUF directory at {gguf_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="HF Repo ID (e.g. YourUsername/Gemma4NPC-it)")
    args = parser.parse_args()
    
    with app.run():
        push_to_hf.remote(args.repo_id)
