import subprocess
import time
import sys

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, executable='/bin/bash')

def main():
    print("Starting automated pipeline...")
    
    # 1. Run/Resume Training
    print("Triggering SFT Training on Modal (resuming from checkpoint-200)...")
    run_cmd("./venv/bin/modal run finetuning/modal_train.py")
    
    print("Training step finished! Waiting 10 seconds to ensure logs and volumes are synced...")
    time.sleep(10)
    
    # 2. Run Evaluation
    print("Triggering LLM-as-a-judge Evaluation on Modal...")
    run_cmd("./venv/bin/modal run finetuning/modal_train.py::run_evaluation")
    
    # 3. Run Quantization
    print("Triggering GGUF Quantization on Modal...")
    run_cmd("./venv/bin/modal run finetuning/modal_train.py::run_quantization")
    
    # 4. Download Results
    print("Downloading Evaluation Results...")
    run_cmd("./venv/bin/modal volume get gemma4npc-outputs /Gemma4NPC-it/evaluation_results.json .")
    
    print("Downloading Quantized Model...")
    run_cmd("./venv/bin/modal volume get gemma4npc-outputs /Gemma4NPC-it/gguf/Gemma4NPC-it-Q4_K_M.gguf ./models/")
    
    print("All automated pipeline steps completed successfully!")

if __name__ == "__main__":
    main()
