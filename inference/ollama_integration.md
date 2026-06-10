# Ollama Deployment Guide for Gemma4NPC
# SPDX-License-Identifier: Apache-2.0

## Quick Start

### 1. Install Ollama
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Create a Modelfile

Create a file called `Modelfile`:
```
FROM ./Gemma4NPC-it-Q4_K_M.gguf

PARAMETER temperature 1.0
PARAMETER top_p 0.95
PARAMETER top_k 64
PARAMETER stop "<turn|>"

SYSTEM """You are a helpful NPC in a fantasy RPG. Stay in character at all times. Respond naturally and concisely."""
```

### 3. Create the Ollama Model
```bash
ollama create gemma4npc -f Modelfile
```

### 4. Run
```bash
ollama run gemma4npc
```

### 5. Use via API
```bash
curl http://localhost:11434/api/chat -d '{
  "model": "gemma4npc",
  "messages": [
    {"role": "system", "content": "You are Aldric, a grizzled blacksmith."},
    {"role": "user", "content": "Can you fix my sword?"}
  ]
}'
```

## Integration with Game Engines
Ollama exposes an OpenAI-compatible API at `http://localhost:11434/v1/chat/completions`.
Use the standard OpenAI client libraries from any language.

## Hardware Requirements
- Q4_K_M: ~8GB RAM (CPU) or ~8GB VRAM (GPU)
- Q8_0: ~14GB RAM/VRAM
