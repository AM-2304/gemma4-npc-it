# Dataset Design

Gemma4NPC requires high-quality, perfectly-formatted conversational data.

## 1. PIPPA Processing
The raw PIPPA dataset is notoriously messy. Our pipeline (`data/process_pippa/`) does the following:
1. Replaces generic placeholders (`{{char}}`, `{{user}}`) with culturally diverse names from `first_names.json`.
2. Resolves weird, broken placeholders.
3. Converts the "system prompt inside the first user message" hack into Gemma 4's native `<|turn>system\n` role.
4. Drops malformed or non-alternating rows.

## 2. Synthetic NPC Dialogue Generation
We augment PIPPA with synthetically generated data to teach the model how to be a *game* NPC, not just a chatbot.
1. We use the `amaydle/npc-dialogue` dataset's NPC definitions as seeds.
2. We ask Gemini 2.0 Flash to generate 24-turn conversations based on those seeds.
3. We strictly validate that every generated conversation has exactly 24 messages, starts with a system prompt, and alternates correctly.

## 3. Formatting Rules (CRITICAL)
Gemma 4 requires absolute strictness in its chat format:
1. The first message **must** be `{"role": "system", "content": "..."}`.
2. After the system prompt, roles **must** alternate `user`, `model`, `user`, `model`.
3. You **cannot** have two `user` messages in a row.
4. You **cannot** have a conversation end on a `user` turn during training.

Any deviation will break the `train_on_responses_only` mask and cause NaN loss. Use `data/validate/schema_check.py` to ensure your data is clean before training!
