# Quantization Discarded Attempts

This folder contains notebooks for quantization methods we attempted but ultimately could not use
for Gemma4NPC. We publish these for transparency and to save the community from hitting the same walls.

## Summary

| Method | Status | Reason | Recommendation |
|--------|--------|--------|----------------|
| AWQ | ❌ Failed | autoawq lacks support for encoder-free unified architecture | Use GGUF Q4_K_M |
| GPTQ | ❌ Failed | Requires ~25GB VRAM for calibration, exceeds A100 40GB usable budget | Use A100 80GB or GGUF |
| ExecuTorch (mobile) | ❌ Failed | 12B model too large for mobile RAM; requires E4B for mobile | Fine-tune Gemma 4 E4B separately |

## Detailed Failure Analysis

### AWQ (AutoAWQ)
- **Attempted**: autoawq v0.2.6
- **Error**: Model architecture `Gemma4ForConditionalGeneration` not supported
- **Root cause**: AutoAWQ's architecture detection doesn't handle Gemma 4's encoder-free unified model
- **Status as of writing**: No fix available; check autoawq GitHub for updates
- **Workaround**: Use GGUF Q4_K_M (equivalent quality, better compatibility)

### GPTQ (GPTQModel)
- **Attempted**: GPTQModel (ModelCloud's maintained fork, since auto-gptq is deprecated)
- **Error**: OOM during calibration — the full float16 model (~25GB) must be loaded
- **Root cause**: GPTQ calibration requires full-precision model + calibration data in VRAM simultaneously
- **Minimum requirement**: A100 80GB or 2× A100 40GB with model parallelism
- **Workaround**: If you have access to A100 80GB, GPTQ may work; otherwise use GGUF

### ExecuTorch (Mobile Deployment)
- **Attempted**: executorch conversion for iOS/Android deployment
- **Error**: Model exceeds mobile device RAM limits (iOS: 6-8GB, Android: 4-8GB)
- **Root cause**: Gemma 4 12B at 11.95B parameters is simply too large for mobile, even quantized
- **Recommendation**: For mobile NPC deployment, fine-tune Gemma 4 E4B (4.5B) instead
- **Note**: E4B has official LiteRT support for on-device inference

## Lessons Learned

1. **GGUF is king for compatibility**: llama.cpp supports virtually every architecture, making GGUF the safest choice
2. **Don't assume quantization library support**: New model architectures often lack support in AWQ/GPTQ for months
3. **Mobile deployment requires smaller models**: The E4B/E2B variants exist specifically for this use case
4. **Always test quantization BEFORE a full training run**: Discovering your target format isn't supported after 15 hours of training is painful
