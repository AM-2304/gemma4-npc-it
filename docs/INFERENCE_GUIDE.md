# Inference & Serving Guide

Gemma4NPC can be deployed in multiple ways depending on your hardware and game engine requirements.

## 1. Game Engine Deployment (GGUF / Llama.cpp)
For running locally alongside a game, **GGUF** is the only viable option.

**Hardware Requirements:**
*   Q4_K_M: ~7.5GB VRAM (RTX 3060/3070)
*   Q8_0: ~13GB VRAM (RTX 3090/4090)

**Usage:**
Run the OpenAI-compatible wrapper:
```bash
python serving/openai_compatible_wrapper.py --model-path models/Gemma4NPC-it-Q4_K_M.gguf
```
Then, point your Unity/Unreal/Godot OpenAI client to `http://localhost:8080/v1`.

## 2. Cloud Production Deployment (vLLM / Docker)
For a centralized server that handles NPC requests for many players, use **vLLM** for maximum throughput via continuous batching.

**Usage:**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model outputs/Gemma4NPC-it/merged_float16 \
    --served-model-name gemma4npc-it \
    --dtype bfloat16 \
    --max-model-len 8192
```

## 3. Apple Silicon (MLX)
If you're developing on a Mac, use MLX. Apple Silicon's unified memory means you aren't restricted by VRAM.

**Usage:**
```bash
python inference/mlx_inference.py --model-path path/to/mlx/model
```

## Prompting Tips
*   **Keep it short:** Game NPCs should speak in 1-3 sentences. Tell them this in the system prompt.
*   **Give them state:** Pass game state (player inventory, gold, NPC mood) invisibly at the end of the system prompt (see the `goblin_merchant/npc_backend.py` demo).
*   **Thinking Mode:** Gemma 4 has a `<think>` token. Disable this for NPCs unless you have massive compute, as the latency is too high for real-time game interaction.
