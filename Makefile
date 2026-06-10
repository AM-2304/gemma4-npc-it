# Gemma4NPC — Convenience Commands
# SPDX-License-Identifier: Apache-2.0

.PHONY: help setup data-pippa data-npc data-combine data-validate train-it train-base train-filtered sweep eval-all demo-game demo-gradio serve lint clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

# === Setup ===
setup: ## Install all dependencies
	pip install -r requirements.txt

setup-dev: ## Install dev dependencies
	pip install -r requirements-dev.txt

setup-mlx: ## Install Apple Silicon MLX dependencies
	pip install -r requirements-mlx.txt

# === Data Processing ===
data-pippa: ## Run full PIPPA processing pipeline
	python data/process_pippa/01_clean_fields.py
	python data/process_pippa/02_replace_placeholders.py
	python data/process_pippa/03_replace_weird_names.py
	python data/process_pippa/04_convert_to_chatml.py
	@echo "✅ PIPPA processing complete"

data-pippa-filtered: data-pippa ## Run PIPPA + content filtering
	python data/process_pippa/05_filter_content.py --filter
	@echo "✅ PIPPA filtered processing complete"

data-npc: ## Generate NPC_dialogue_v2 dataset
	python data/generate_npc_dialogue/01_generate_system_prompts.py
	python data/generate_npc_dialogue/02_generate_dataset.py
	python data/generate_npc_dialogue/03_validate_json.py
	python data/generate_npc_dialogue/04_convert_to_jsonl.py
	@echo "✅ NPC dialogue v2 generation complete"

data-combine: ## Combine all datasets into final training set
	python data/combine/combine_datasets.py

data-validate: ## Validate all final datasets against schema
	python data/validate/schema_check.py data/final/pippa_gemma4.jsonl
	python data/validate/schema_check.py data/final/npc_dialogue_v2.jsonl
	python data/validate/schema_check.py data/final/RolePlay-NPC-v2_train.jsonl
	@echo "✅ All datasets pass schema validation"

data-stats: ## Print dataset statistics
	python data/validate/stats.py

data-push: ## Push all datasets to HuggingFace Hub
	python data/process_pippa/06_push_to_hub.py

data-all: data-pippa data-npc data-combine data-validate ## Run entire data pipeline
	@echo "✅ Full data pipeline complete"

# === Training ===
train-it: ## Train Gemma4NPC-it (flagship model)
	python finetuning/train.py --config configs/training_npc_it.yaml

train-base: ## Train Gemma4NPC (PIPPA only)
	python finetuning/train.py --config configs/training_base.yaml

train-filtered: ## Train Gemma4NPC-Filtered
	python finetuning/train.py --config configs/training_filtered.yaml

sweep: ## Run hyperparameter safety sweep
	python finetuning/hparam_safety_sweep.py

# === Evaluation ===
eval-all: ## Run all evaluation scripts
	python evaluation/roleplay_eval.py
	python evaluation/response_length_analysis.py
	python evaluation/character_consistency_eval.py
	python evaluation/inference_speed_benchmark.py
	@echo "✅ All evaluations complete"

eval-judge: ## Run LLM-as-judge evaluation
	python evaluation/llm_as_judge_eval.py

# === Quantization ===
quant-gguf: ## Export to GGUF format
	python quantization/export_to_gguf.py --model-path outputs/Gemma4NPC-it/merged_float16

quant-qat: ## Export QAT variant
	python quantization/export_qat.py --model-path outputs/Gemma4NPC-it/merged_float16

# === Demos ===
demo-game: ## Launch Goblin Merchant game
	python demos/goblin_merchant/game.py

demo-gradio: ## Launch Gradio chat demo
	python demos/gradio_chat/app.py

# === Serving ===
serve: ## Start OpenAI-compatible API server
	python serving/openai_compatible_wrapper.py --port 8080

serve-docker: ## Build and start Docker container
	cd serving/docker && docker-compose up --build

# === Code Quality ===
lint: ## Run linting and formatting checks
	ruff check .
	black --check .

format: ## Auto-format code
	ruff check --fix .
	black .

# === Cleanup ===
clean: ## Remove generated artifacts (not models)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
	rm -f score.txt
	@echo "✅ Cleaned up"
