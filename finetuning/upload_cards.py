import os
from huggingface_hub import HfApi

def upload_cards():
    api = HfApi()
    
    # 1. Update Model Repo (spy5er/Gemma4NPC-it)
    model_repo = "spy5er/Gemma4NPC-it"
    print(f"Uploading MODEL_CARD.md to {model_repo} as README.md...")
    api.upload_file(
        path_or_fileobj="MODEL_CARD.md",
        path_in_repo="README.md",
        repo_id=model_repo,
        repo_type="model",
        commit_message="Update model card with Quest capabilities"
    )
    print("✅ Model card updated successfully!")
    
    # 2. Update Dataset Repo (spy5er/Gemma4NPC-Quest-Dataset)
    dataset_repo = "spy5er/Gemma4NPC-Quest-Dataset"
    print(f"\nCreating dataset repository {dataset_repo}...")
    api.create_repo(repo_id=dataset_repo, repo_type="dataset", exist_ok=True, private=False)
    
    print(f"Uploading DATASET_CARD.md to {dataset_repo} as README.md...")
    api.upload_file(
        path_or_fileobj="DATASET_CARD.md",
        path_in_repo="README.md",
        repo_id=dataset_repo,
        repo_type="dataset",
        commit_message="Add dataset card"
    )
    
    print(f"Uploading augmented Quest dataset to {dataset_repo}...")
    api.upload_file(
        path_or_fileobj="data/augmented/npc_quest_sft_fixed.jsonl",
        path_in_repo="npc_quest_sft.jsonl",
        repo_id=dataset_repo,
        repo_type="dataset",
        commit_message="Upload SFT dataset"
    )
    print("✅ Dataset and card updated successfully!")

if __name__ == "__main__":
    upload_cards()
