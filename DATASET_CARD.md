---
license: cc-by-nc-sa-4.0
task_categories:
- text-generation
- reinforcement-learning
language:
- en
tags:
- npc
- roleplay
- dpo
- gaming
- pippa
- json
pretty_name: Gemma4NPC Preference Dataset
size_categories:
- 10K<n<100K
---

# Dataset Card for Gemma4NPC Preference Dataset

## Dataset Description

The Gemma4NPC Preference Dataset is a specialized text-generation and reinforcement learning dataset designed to train Large Language Models (LLMs) for use as Non-Playable Characters (NPCs) in video games. 

Integrating LLMs into game engines requires models that can seamlessly blend creative roleplay with strict formatting requirements. This dataset addresses two primary training objectives:
1. **Character Consistency:** Teaching the model to maintain strict adherence to a persona (e.g., a medieval merchant, a sci-fi robot) without hallucinating items or breaking immersion to act as an AI assistant.
2. **Structured Outputs:** Training the model to output its natural language dialogue alongside machine-readable JSON data. This enables game engines (like Unity or Unreal Engine) to dynamically parse game-state variables (such as inventory updates or quest flags) from the model's response.

- **Curated by:** spy5er
- **Language(s):** English
- **License:** CC-BY-NC-SA-4.0

## Dataset Structure

The repository contains two distinct data structures corresponding to the two phases of alignment training.

### 1. Supervised Fine-Tuning (SFT) Subset (`npc_sft_chatml.jsonl`)
This subset is used to teach the model the fundamental grammar of acting as an NPC and formatting its output into JSON. The data consists of multi-turn roleplay conversations curated from the PIPPA dataset, formatted into the ChatML standard, with the assistant's responses wrapped entirely within JSON objects.

**Example Entry:**
```json
{
  "messages": [
    {"role": "system", "content": "You are ARIA-7, a glitching sci-fi robot."},
    {"role": "user", "content": "Are you functional?"},
    {"role": "assistant", "content": "{\n  \"dialogue\": \"Functional? I— yes. Mostly. Core systems at 73%... *static hiss*...\"\n}"}
  ]
}
```

### 2. Direct Preference Optimization (DPO) Subset (`npc_dpo_pairs.jsonl`)
This subset is designed for the alignment phase. It contains paired responses to identical prompts, allowing the mathematical optimization algorithms (DPO) to heavily penalize out-of-character behavior and reward strict adherence to the JSON schema and the defined persona.

**Example Entry:**
```json
{
  "prompt": "You are Gringo the Goblin selling an Amulet for 500 gold. The user says: 'I will give you 50 gold!'",
  "chosen": "{\n  \"dialogue\": \"50 gold?! Are you trying to insult me? 450, not a copper less!\",\n  \"agreed_price\": 500\n}",
  "rejected": "I am an AI and cannot accept gold. However, I can lower the price to 50 for you."
}
```

## Dataset Creation

### Source Data
The baseline conversational data was derived from highly-rated, multi-turn interactions within the open-source PIPPA (Persona-Interacting Professional Play-Acting) dataset.

### Data Processing and Curation
1. **Sanitization:** The raw text was heavily filtered to remove corrupted formatting, excessive markdown, emojis, and out-of-character (OOC) system instructions.
2. **JSON Augmentation:** Automated scripts were used to wrap the raw assistant dialogue into the target `{"dialogue": "..."}` schema blocks.
3. **Negative Pair Generation (DPO):** To create the `rejected` examples for the DPO pairs, a baseline LLM was intentionally prompted to break character, ignore JSON constraints, or hallucinate inventory items. These were mapped against the sanitized, high-quality responses (the `chosen` examples).

## How to Use This Dataset

This dataset is pre-formatted for direct integration with popular alignment libraries such as `TRL` (Transformer Reinforcement Learning) and `Unsloth`.

```python
from datasets import load_dataset
from trl import DPOTrainer

# Load the preference dataset
dataset = load_dataset("spy5er/Gemma4NPC-Preference", data_files="npc_dpo_pairs.jsonl")

# Initialize the DPO trainer
trainer = DPOTrainer(
    model=model,
    train_dataset=dataset["train"],
    # Additional configurations...
)
```
