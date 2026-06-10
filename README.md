# 🎮 Gemma4NPC

A production-grade NPC roleplay fine-tuning framework targeting `google/gemma-4-12B-it`.

This repository provides a complete, end-to-end pipeline for processing roleplay data, fine-tuning Gemma 4, evaluating character consistency, quantizing models for consumer hardware, and deploying them to game engines like Unity, Unreal, and Godot.

## Features

- **Robust Data Pipeline**: Cleans and converts the PIPPA dataset, and generates synthetic 24-turn NPC conversations via Gemini.
- **Gemma 4 Architecture Ready**: Properly implements the new Gemma 4 unified multimodal architecture and native chat delimiters (`<|turn>user\n`, `<|turn>model\n`).
- **Safety First**: Implements strict NaN-prevention training configs (`max_grad_norm=0.4`, `bfloat16`, `adamw_8bit`).
- **Comprehensive Evaluation**: Includes LLM-as-a-judge, response length analysis, and 16-turn character consistency probes.
- **Game Engine Ready**: Provides GGUF/llama.cpp inference, an OpenAI-compatible REST server, and plugin stubs for major engines.
- **Structured Game State Control**: Built-in support for generating JSON outputs that actively control game variables (e.g., NPC mood, prices, events).
- **Multimodal Support**: Includes Gradio demos demonstrating how to pass images to the model so the NPC can "see" the game state.

## Quick Start

### 1. Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
*(For Apple Silicon, use `requirements-mlx.txt`)*

### 2. Prepare Data
```bash
make data-all
```
This runs the full dataset cleaning, generation, and validation pipeline.

### 3. Train
If running locally (requires 24GB+ VRAM):
```bash
make train-npc
```
For Colab/RunPod, use the provided notebooks in the `finetuning/` directory.

### 4. Serve
```bash
python serving/openai_compatible_wrapper.py --model-path models/Gemma4NPC-it-Q4_K_M.gguf
```

## Documentation

- **[Dataset Design](docs/DATASET_DESIGN.md)**: How we build and format the data.
- **[Inference & Serving Guide](docs/INFERENCE_GUIDE.md)**: How to integrate the model into a game.
- **[Evaluation Methodology](docs/EVALUATION.md)**: How we score NPC performance.
- **[Quantization Notes](quantization/Quantizations--Discarded-Attempts/README.md)**: Why we chose GGUF over AWQ/GPTQ.

## Project Structure

- `configs/` - YAML configuration files for training and inference
- `data/` - Pipeline scripts for data processing and DPO generation
- `demos/` - Example implementations (Goblin Merchant Haggling UI, Gradio chat)
- `docs/` - Project documentation
- `evaluation/` - Automated metrics, LLM-as-judge, and scenarios
- `finetuning/` - Training scripts and Colab notebooks
- `inference/` - Scripts for running the model via transformers, llama.cpp, vLLM, and MLX
- `quantization/` - Scripts for exporting to GGUF, QAT, and AWQ
- `serving/` - Docker deployment and game engine plugin stubs

## License
Apache 2.0
