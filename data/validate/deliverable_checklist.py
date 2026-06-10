#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Deliverable Checklist — Verifies that all expected outputs, scripts, and docs
are present and correctly structured before finalizing the project.

Usage:
    python data/validate/deliverable_checklist.py
"""

import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(message)s"
)
logger = logging.getLogger(__name__)

EXPECTED_FILES = [
    # Root
    "README.md",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-mlx.txt",
    "pyproject.toml",
    "Makefile",
    
    # Data Processing
    "data/process_pippa/01_clean_fields.py",
    "data/process_pippa/02_replace_placeholders.py",
    "data/process_pippa/03_replace_weird_names.py",
    "data/process_pippa/04_convert_to_chatml.py",
    "data/process_pippa/05_filter_content.py",
    "data/process_pippa/06_push_to_hub.py",
    "data/process_pippa/first_names.json",
    
    # NPC Generation
    "data/generate_npc_dialogue/01_generate_system_prompts.py",
    "data/generate_npc_dialogue/02_generate_dataset.py",
    "data/generate_npc_dialogue/03_validate_json.py",
    "data/generate_npc_dialogue/04_convert_to_jsonl.py",
    "data/generate_npc_dialogue/generation_prompt_template.txt",
    
    # Combine & Validate
    "data/combine/combine_datasets.py",
    "data/validate/schema_check.py",
    "data/validate/stats.py",
    
    # DPO
    "data/dpo/generate_preference_pairs.py",
    "data/dpo/preference_schema.md",
    
    # Configs
    "configs/training_base.yaml",
    "configs/training_npc_it.yaml",
    "configs/training_filtered.yaml",
    "configs/dpo_config.yaml",
    "configs/inference_defaults.yaml",
    
    # Finetuning
    "finetuning/train.py",
    "finetuning/hparam_safety_sweep.py",
    "finetuning/Gemma4NPC_Base.ipynb",
    "finetuning/Gemma4NPC_IT.ipynb",
    "finetuning/Gemma4NPC_Filtered.ipynb",
    "finetuning/Gemma4NPC_DPO.ipynb",
    
    # Inference
    "inference/transformers_inference.py",
    "inference/llama_cpp_inference.py",
    "inference/batch_inference.py",
    "inference/streaming_inference.py",
    "inference/structured_inference.py",
    "inference/vllm_inference.py",
    "inference/mlx_inference.py",
    "inference/ollama_integration.md",
    
    # Evaluation
    "evaluation/roleplay_eval.py",
    "evaluation/response_length_analysis.py",
    "evaluation/character_consistency_eval.py",
    "evaluation/bleu_rouge_eval.py",
    "evaluation/llm_as_judge_eval.py",
    "evaluation/inference_speed_benchmark.py",
    "evaluation/scenarios/rele_system_prompt.txt",
    
    # Quantization
    "quantization/export_to_gguf.py",
    "quantization/export_to_awq.py",
    "quantization/export_qat.py",
    "quantization/run_gguf_my_repo.md",
    "quantization/Quantizations--Discarded-Attempts/README.md",
    
    # Demos
    "demos/goblin_merchant/game.py",
    "demos/goblin_merchant/npc_backend.py",
    "demos/gemma4npc_space/app.py",
    "demos/gemma4npc_space/requirements.txt",
    "demos/gradio_chat/app.py",
    
    # Serving
    "serving/openai_compatible_wrapper.py",
    "serving/docker/Dockerfile",
    "serving/docker/docker-compose.yml",
    "serving/game_engine_plugins/godot_plugin_stub/GemmaNPCClient.gd",
    "serving/game_engine_plugins/unity_plugin_stub/GemmaNPCClient.cs",
    "serving/game_engine_plugins/unreal_plugin_stub/GemmaNPCClient.cpp",
    
    # Docs
    "docs/DATASET_DESIGN.md",
    "docs/INFERENCE_GUIDE.md",
    "docs/EVALUATION.md",
]


def main():
    root = Path(__file__).resolve().parent.parent.parent
    logger.info(f"Checking deliverables in {root}...\n")
    
    missing = []
    found = 0
    
    for relative_path in EXPECTED_FILES:
        p = root / relative_path
        if p.exists():
            found += 1
        else:
            missing.append(relative_path)
            
    total = len(EXPECTED_FILES)
    logger.info(f"✅ Found {found}/{total} files.")
    
    if missing:
        logger.warning(f"❌ Missing {len(missing)} files:")
        for m in missing:
            logger.warning(f"  - {m}")
        exit(1)
    else:
        logger.info("\n🎉 All deliverables are present! Project is complete.")
        exit(0)


if __name__ == "__main__":
    main()
