# DPO Preference Pair Schema

Each row in `preference_pairs.jsonl` must have:

```json
{
  "prompt": "System: <system_prompt>\nUser: <user_message>",
  "chosen": "<better response as judged by LLM>",
  "rejected": "<worse response as judged by LLM>",
  "judge_reason": "<one-sentence reason from GPT-4o/Claude>"
}
```

## Generation Process
1. Load the SFT fine-tuned model
2. For each NPC scenario, generate Response A (temp=0.7) and Response B (temp=1.3)
3. Send both to GPT-4o/Claude with a blind comparison prompt
4. Label the preferred response as "chosen", the other as "rejected"

## Quality Notes
- Minimum 500 pairs recommended for DPO
- Ideal: 1000+ pairs across diverse NPC archetypes
- Judge agreement rate should be >80% (if lower, the model responses may be too similar)
