# Using gguf-my-repo HuggingFace Space

If you don't have the hardware or toolchain to quantize locally, you can use the
`ggml-org/gguf-my-repo` HuggingFace Space to quantize any HuggingFace model.

## Step-by-Step Instructions

### Step 1: Upload Your Model to HuggingFace Hub
Make sure your merged float16 model is uploaded:
```bash
python finetuning/train.py --config configs/training_npc_it.yaml --push-to-hub --hub-model-id YOUR_USERNAME/Gemma4NPC-it
```

### Step 2: Navigate to the Space
Go to: https://huggingface.co/spaces/ggml-org/gguf-my-repo

### Step 3: Login with HuggingFace
Click "Sign in with Hugging Face" in the top right.

### Step 4: Enter Model Details
- **Model ID**: Enter `YOUR_USERNAME/Gemma4NPC-it`
- **Quantization type**: Select from dropdown:
  - `Q4_K_M` — Best for consumer GPUs (RTX 3060/3070, ~7.5GB)
  - `Q8_0` — Best quality for high-end GPUs (RTX 3090/4090, ~13GB)
  - `Q5_K_M` — Good balance between Q4 and Q8

### Step 5: Start Quantization
Click "Submit". The Space will:
1. Download your model
2. Convert to GGUF format
3. Quantize to your chosen format
4. Create a new repo: `YOUR_USERNAME/Gemma4NPC-it-GGUF`
5. Upload the quantized model

### Step 6: Verify
The quantized model will appear in your HuggingFace profile.
Download and test locally:
```bash
# Via huggingface-cli
huggingface-cli download YOUR_USERNAME/Gemma4NPC-it-GGUF --include "*.gguf" --local-dir ./models

# Test with llama-cpp-python
python inference/llama_cpp_inference.py --model-path models/Gemma4NPC-it-Q4_K_M.gguf
```

## Notes
- Processing time: ~15-30 minutes depending on queue
- The Space uses the latest llama.cpp for maximum compatibility
- Multiple quantization types can be generated in one run
- The output repo includes a `README.md` with model card
