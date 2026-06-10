# Evaluation Framework

Evaluating roleplay models is notoriously difficult. Standard benchmarks (MMLU, HumanEval) don't measure personality consistency or naturalness.

We use a multi-pronged evaluation approach:

## 1. Character Consistency
`evaluation/character_consistency_eval.py`
We hit the model with 16 diverse probes ("Who are you?", "Do you know you're in a game?", "What's your secret?"). We then use sentence embeddings to ensure the character's *style* doesn't drift over the 16 turns. We also explicitly check if the model breaks the 4th wall.

## 2. LLM-as-a-Judge
`evaluation/llm_as_judge_eval.py`
We use GPT-4o to score responses on a 1-5 scale across five dimensions:
1.  **Consistency**: Adherence to the system prompt.
2.  **Naturalness**: Does it sound spoken or written?
3.  **Advancement**: Does it move the interaction forward?
4.  **Length**: Is it appropriately short for a game?
5.  **Knowledge**: Does it avoid hallucinating modern concepts?

## 3. Response Length Analysis
`evaluation/response_length_analysis.py`
Base models tend to write paragraphs. Fine-tuned NPC models should write 30-60 tokens. This script generates a histogram of response lengths to ensure the model isn't "over-talking."

## 4. Inference Speed
`evaluation/inference_speed_benchmark.py`
A slow NPC is a dead NPC. We benchmark time-to-first-token and tokens-per-second across different quantization formats. Target: >15 tok/sec.
