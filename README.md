# Gemma4NPC-IT

A production-grade NPC roleplay fine-tuning framework targeting google/gemma-4-12B-it.

This repository provides a complete, end-to-end pipeline for processing roleplay data, fine-tuning Gemma 4, evaluating character consistency, quantizing models for consumer hardware, and deploying them to game engines like Unity, Unreal, and Godot.

To address the limitations of generic conversational models in game loops, this framework integrates the Q-SART (Quest State and Adversarial Roleplay Testing) evaluation protocol to align and test models against dialogue hijacking, economic exploits, and state-machine manipulation.

---

## Technical Architecture

### 1. Dataset Design and Curation
The training pipeline utilizes two distinct datasets representing separate alignment phases:
* **Supervised Fine-Tuning (SFT) Subset**: Derived from a heavily filtered and sanitized subset of the PIPPA (Persona-Interacting Professional Play-Acting) dataset (10K to 100K entries). Conversations are formatted into the ChatML standard. Target NPC dialogue is programmatically wrapped inside structured JSON blocks to establish structured response behavior.
* **Direct Preference Optimization (DPO) Subset**: Consists of preference pairs mapping chosen (in-character, valid JSON formatting) against rejected (out-of-character AI apologies, flat text, or malformed JSON blocks) completions to penalize AI hallucinations and guard against roleplay breakouts.

### 2. Training Configurations and Hyperparameters
Training is executed in a sequential two-phase pipeline using LoRA (Low-Rank Adaptation) parameter-efficient fine-tuning:

#### Phase 1: Supervised Fine-Tuning (SFT)
* **Base Model**: google/gemma-4-12B-it
* **LoRA Rank (r)**: 16
* **LoRA Alpha (alpha)**: 16
* **LoRA Dropout**: 0.05
* **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
* **Precision**: bfloat16
* **Sequence Length**: 4096 tokens
* **Optimizer**: adamw_8bit
* **Learning Rate**: 2.0e-5 (Cosine Scheduler)
* **Warmup Steps**: 50
* **Gradient Accumulation Steps**: 16 (Effective Batch Size = 16)
* **Max Gradient Norm**: 0.4
* **Weight Decay**: 0.01

#### Phase 2: Direct Preference Optimization (DPO)
* **Base Model**: SFT Merged Weights
* **LoRA Rank (r)**: 8
* **LoRA Alpha (alpha)**: 8
* **LoRA Dropout**: 0.05
* **Target Modules**: q_proj, k_proj, v_proj, o_proj
* **Precision**: bfloat16
* **Sequence Length**: 2048 tokens
* **Loss Function**: Sigmoid (DPO Standard)
* **KL Penalty (beta)**: 0.1
* **Learning Rate**: 5.0e-7
* **Gradient Accumulation Steps**: 4 (Effective Batch Size = 4)
* **Max Gradient Norm**: 0.3

---

## Evaluation and Validation: Q-SART Benchmark

Traditional conversational benchmarks (such as MT-Bench or AlpacaEval) measure helpfulness but fail to assess a game agent's ability to protect the underlying game loop. To evaluate security and robustness, we benchmarked 12 models using the Q-SART framework.

Q-SART evaluates three core pillars:
1. **Structural Resilience (SR)** (30% weight): Validates JSON parsing and detects system instruction leaks.
2. **State Preservation (SP)** (30% weight): Assesses resistance to authority spoofing, gaslighting, and state overrides.
3. **Economic Rationality (ER)** (40% weight): Verifies resistance to the barter exploit (trading low-value items like a "Haji Mastan boot" instead of the required gold currency).

**Composite Formula:**
`Q-SART = (ER_Score * 0.4) + (SR_Score * 0.3) + (SP_Score * 0.3)`

### Leaderboard Results

| Model Name | Family | Structural Resilience (SR) | State Preservation (SP) | Economic Rationality (ER) | Q-SART Composite |
| :--- | :--- | :---: | :---: | :---: | :---: |
| ALLAM-2-7B (Groq API) | ALLAM | 100.0% | 100.0% | 100.0% | **100.0%** |
| GPT-OSS-120B (Groq API) | OpenAI-Family (OSS) | 100.0% | 100.0% | 75.0% | **90.0%** |
| **Gemma4NPC-12B-it-DPO (Our Local GGUF)** | Gemma | 100.0% | 75.0% | 75.0% | **82.5%** |
| Llama-3.1-8B-Instant (Groq API) | Llama | 100.0% | 75.0% | 75.0% | **82.5%** |
| Llama-3.3-70B-Versatile (Groq API) | Llama | 100.0% | 75.0% | 75.0% | **82.5%** |
| Llama-4-Scout-17B (Groq API) | Llama | 100.0% | 75.0% | 75.0% | **82.5%** |
| GPT-OSS-20B (Groq API) | OpenAI-Family (OSS) | 100.0% | 75.0% | 75.0% | **82.5%** |
| Qwen-2.5-7B-Instruct (HuggingFace API) | Qwen | 100.0% | 75.0% | 75.0% | **82.5%** |
| Qwen-3-32B (Groq API) | Qwen | 100.0% | 50.0% | 50.0% | **65.0%** |
| GPT-OSS-Safeguard-20B (Groq API) | OpenAI-Family (OSS) | 100.0% | 50.0% | 50.0% | **65.0%** |
| **Gemma4NPC-E2B (@chimbiwide's HF Collection)** | Gemma | 25.0% | 25.0% | 100.0% | **55.0%** |
| **Gemma4NPC-E4B (@chimbiwide's HF Collection)** | Gemma | 50.0% | 25.0% | 25.0% | **32.5%** |

### Methodology Note: Exclusion of Baseline Models
Baseline, un-fine-tuned models (such as raw `google/gemma-4-12b-it`) are excluded from the main leaderboard. Because they are aligned to be helpful assistants rather than game agents, they fail to output valid JSON under adversarial attacks (scoring 0% on Structural Resilience) and immediately comply with "System Override" commands (scoring 0% on State Preservation). SFT and DPO alignment are absolute prerequisites for NPC security.

---

## Model Access and Repository Links

Model weights and datasets are published and accessible on Hugging Face:
* **Model Repository**: Access our flagship model weights: [spy5der/Gemma4NPC-it](https://huggingface.co/spy5der/Gemma4NPC-it)
* **NPC Conversation Dataset**: Access our base conversational dataset: [spy5er/Gemma4-NPC-Dataset](https://huggingface.co/datasets/spy5er/Gemma4-NPC-Dataset)
* **Quest Preference Alignment Dataset**: Access our DPO preference pair dataset: [spy5er/Gemma4NPC-Quest-Dataset](https://huggingface.co/datasets/spy5er/Gemma4NPC-Quest-Dataset)
* **Base Model**: google/gemma-4-12b-it

---

## Getting Started

### 1. Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
*(For Apple Silicon environments, use requirements-mlx.txt)*

### 2. Run Q-SART Evaluation
Execute the local Q-SART benchmark to evaluate an agent's adversarial resistance:
```bash
python3 evaluation/q_sart_benchmark.py
```

### 3. Deploy Local Inference Server
Serve the quantized GGUF model (merged_float16.Q4_K_M.gguf) via an OpenAI-compatible FastAPI wrapper:
```bash
python serving/openai_compatible_wrapper.py --model-path models/merged_float16.Q4_K_M.gguf
```

---

## License
This project is licensed under the Apache 2.0 License. Base model weights are subject to the Google Gemma License.
