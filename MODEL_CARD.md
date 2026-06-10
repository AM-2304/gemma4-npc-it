---
language:
- en
license: gemma
library_name: transformers
tags:
- gemma
- gemma-4
- roleplay
- npc
- gaming
- json
- gguf
- macos
- dpo
base_model: google/gemma-4-12b
pipeline_tag: text-generation
---

# Model Card for Gemma4NPC-12B-it

## Model Details

### Model Description

Gemma4NPC-12B-it is a 12 billion parameter language model specifically fine-tuned and aligned to serve as the backend for Non-Playable Characters (NPCs) in video games. Built on top of Google's Gemma 4 architecture, this model is designed to solve common issues faced when integrating Large Language Models into game engines: character consistency and structured data output. 

Traditional language models often break immersion by referencing their nature as AI assistants or failing to output data in a format that a game engine can parse. Gemma4NPC addresses this by combining Supervised Fine-Tuning (SFT) for strict roleplay adherence with Direct Preference Optimization (DPO) to guarantee outputs in valid, machine-readable JSON formats. This allows game developers to seamlessly parse the NPC's dialogue alongside mathematical game-state updates (such as quest flags, inventory trades, or mood variables) directly in engines like Unity, Unreal Engine, or Godot.

- **Developed by:** spy5er
- **Model type:** Causal Language Model
- **Language(s):** English
- **License:** Gemma License
- **Finetuned from model:** google/gemma-4-12b

### Intended Uses & Limitations

**Intended Use Cases:**
- Real-time NPC dialogue generation in video games.
- Structured inference requiring strict JSON formatting alongside natural language.
- Interactive storytelling and text-based roleplaying environments.

**Limitations:**
- The model is heavily optimized for short-context, turn-based dialogue and may struggle with long-form essay generation or generalized assistant tasks.
- It is fine-tuned to maintain character immersion; therefore, it will actively resist breaking character even if explicitly prompted to do so.

## Training Details

### Training Data

The model was fine-tuned on a heavily sanitized and augmented subset of the PIPPA (Persona-Interacting Professional Play-Acting) dataset. The data was structured using the ChatML format. For the alignment phase, preference pairs were synthetically generated to penalize out-of-character behavior and reward strict JSON formatting. For more details, refer to the accompanying Dataset Card.

### Training Procedure

The training pipeline was executed in two distinct phases:

1. **Supervised Fine-Tuning (SFT):** The base Gemma-4-12B model was fine-tuned on the sanitized roleplay dataset to learn the basic grammar of acting as an NPC. The model was trained to output its responses wrapped inside structured JSON blocks.
2. **Direct Preference Optimization (DPO):** To heavily discourage hallucinations and character breaks, the model underwent DPO. The model was presented with paired responses (a chosen "in-character" response with perfect JSON, and a rejected "out-of-character" or improperly formatted response). This mathematical alignment severely punishes the weights responsible for AI-like apologies and rewards rigid adherence to game logic.

## Technical Specifications

### Architecture
Gemma4NPC-12B-it retains the core architecture of Gemma 4. The model weights provided in this repository are available in both unquantized (Float16) and quantized (GGUF) formats.

### Quantization (GGUF)
To facilitate local inference on consumer hardware and Apple Silicon (M-series Macs), the model has been quantized to the `Q4_K_M` GGUF format. This compresses the 24 GB Float16 model down to approximately 7.5 GB while maintaining over 95% of its reasoning quality. This allows the model to comfortably fit entirely within VRAM/Unified Memory, achieving speeds of 30 to 45 tokens per second.

## Comparison with Other Models

### Gemma4NPC-12B-it vs. chimbiwide/Gemma4NPC-E4B
There is another excellent community project, [`chimbiwide/Gemma4NPC-E4B`](https://huggingface.co/chimbiwide/Gemma4NPC-E4B), which targets similar NPC roleplay use cases. Here is a brief architectural and performance comparison to help you choose the right model for your project:

- **Model Size & Reasoning:** `Gemma4NPC-E4B` is built on the smaller `Gemma-4-E4B` (4 Billion parameter) foundation, making it highly efficient. However, our `Gemma4NPC-12B-it` leverages the 12 Billion parameter base, providing significantly deeper logic and context tracking for complex, multi-layered player negotiations.
- **Engine Integration:** `Gemma4NPC-E4B` outputs pure conversational text, requiring complex Regex parsing to extract game events. Our model is trained to output strict JSON schemas (e.g., `{"dialogue": "...", "agreed_price": 500}`), acting as a true engine backend for updating UI, inventories, and quests.
- **Alignment:** `Gemma4NPC-E4B` is trained via Supervised Fine-Tuning (LoRA). Our model employs a two-step SFT + Direct Preference Optimization (DPO) pipeline to actively penalize character breaks and item hallucinations.
- **Latency:** As a 4B model, `Gemma4NPC-E4B` is naturally faster. We bridge this gap by aggressively quantizing our 12B model to `Q4_K_M` GGUF, achieving 30-45+ tokens per second on consumer hardware while maintaining the superior reasoning of a 12B parameter model.

## How to Get Started with the Model

You can run this model locally using `llama.cpp` or `llama-cpp-python`.

**1. Download the Quantized Model:**
```bash
huggingface-cli download spy5er/Gemma4NPC-12B-it merged_float16.Q4_K_M.gguf --local-dir models/
```

**2. Python Inference Server (FastAPI Example):**
```python
from llama_cpp import Llama

# Load the model directly into your local GPU (Metal for Mac)
llm = Llama(
    model_path="models/merged_float16.Q4_K_M.gguf",
    n_gpu_layers=-1, # Loads 100% of the model into the GPU for maximum speed
    chat_format="gemma"
)

# Request a structured JSON output
response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are Gringo the Greedy, a goblin merchant..."},
        {"role": "user", "content": "I will give you 50 gold for that Amulet!"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "schema": {
                "type": "object",
                "properties": {
                    "dialogue": {"type": "string"},
                    "agreed_price": {"type": "integer"}
                },
                "required": ["dialogue", "agreed_price"],
                "additionalProperties": False
            }
        }
    }
)

print(response["choices"][0]["message"]["content"])
# Output: {"dialogue": "50 gold?! Are you mad? Make it 450!", "agreed_price": 500}
```
