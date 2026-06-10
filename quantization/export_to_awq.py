#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Export to AWQ — AutoAWQ quantization pipeline.

WARNING: As of mid-2025, AWQ support for Gemma 4 12B's encoder-free unified
architecture may be incomplete in autoawq. This script includes proper error
handling and falls back to GGUF guidance if AWQ fails.

Usage:
    python quantization/export_to_awq.py \
        --model-path outputs/Gemma4NPC-it/merged_float16 \
        --output-dir outputs/Gemma4NPC-it/awq
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export to AWQ")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--w-bit", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()

    output_dir = args.output_dir or f"{args.model_path}-AWQ"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

        logger.info(f"Loading model: {args.model_path}")
        model = AutoAWQForCausalLM.from_pretrained(
            args.model_path, safetensors=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        quant_config = {
            "zero_point": True,
            "q_group_size": args.group_size,
            "w_bit": args.w_bit,
            "version": "GEMM",
        }

        logger.info(f"Quantizing with config: {quant_config}")
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data="pileval",  # calibration dataset
        )

        logger.info(f"Saving to {output_dir}")
        model.save_quantized(output_dir)
        tokenizer.save_pretrained(output_dir)

        logger.info(f"✅ AWQ quantization complete: {output_dir}")

    except ImportError:
        logger.error("autoawq not installed. Install with: pip install autoawq")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ AWQ quantization failed: {e}")
        logger.error("")
        logger.error("This is expected — AWQ support for Gemma 4 12B's encoder-free")
        logger.error("unified architecture may not yet be available in autoawq.")
        logger.error("")
        logger.error("RECOMMENDED ALTERNATIVES:")
        logger.error("  1. Use GGUF Q4_K_M (best compatibility): python quantization/export_to_gguf.py")
        logger.error("  2. Use gguf-my-repo HF Space: see quantization/run_gguf_my_repo.md")
        logger.error("  3. Check autoawq GitHub for Gemma 4 support updates")
        logger.error("")
        logger.error("See: quantization/Quantizations--Discarded-Attempts/AWQ_attempt.ipynb")
        sys.exit(1)


if __name__ == "__main__":
    main()
