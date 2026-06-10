#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Export to GGUF — Converts merged float16 model to GGUF quantizations.
Requires llama.cpp to be cloned locally.

Usage:
    python quantization/export_to_gguf.py \
        --model-path outputs/Gemma4NPC-it/merged_float16 \
        --output-dir outputs/Gemma4NPC-it/gguf \
        --quants Q4_K_M Q8_0
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def clone_llama_cpp(target_dir: str = "llama.cpp"):
    """Clone llama.cpp if not present and build."""
    if not Path(target_dir).exists():
        logger.info("Cloning llama.cpp...")
        subprocess.run(
            ["git", "clone", "https://github.com/ggerganov/llama.cpp.git", target_dir],
            check=True,
        )
        
    quantize_bin = Path(target_dir) / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        logger.info("Building llama.cpp with CMake...")
        subprocess.run(["cmake", "-B", "build"], cwd=target_dir, check=True)
        subprocess.run(["cmake", "--build", "build", "--config", "Release", "-j"], cwd=target_dir, check=True)


def convert_to_gguf(model_path: str, output_path: str, llama_cpp_dir: str = "llama.cpp"):
    """Convert HF model to GGUF F16."""
    convert_script = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        logger.error(f"Convert script not found: {convert_script}")
        sys.exit(1)

    logger.info(f"Converting {model_path} to GGUF F16...")
    subprocess.run(
        [sys.executable, str(convert_script), model_path, "--outfile", output_path,
         "--outtype", "f16"],
        check=True,
    )
    size_gb = Path(output_path).stat().st_size / 1e9
    logger.info(f"F16 GGUF: {output_path} ({size_gb:.1f} GB)")


def quantize_gguf(input_gguf: str, output_gguf: str, quant_type: str,
                  llama_cpp_dir: str = "llama.cpp"):
    """Quantize GGUF to target format."""
    quantize_bin = Path(llama_cpp_dir) / "quantize"
    if not quantize_bin.exists():
        # Try build/bin location
        quantize_bin = Path(llama_cpp_dir) / "build" / "bin" / "llama-quantize"

    logger.info(f"Quantizing to {quant_type}...")
    subprocess.run(
        [str(quantize_bin), input_gguf, output_gguf, quant_type],
        check=True,
    )
    size_gb = Path(output_gguf).stat().st_size / 1e9
    logger.info(f"{quant_type} GGUF: {output_gguf} ({size_gb:.1f} GB)")


def verify_gguf(gguf_path: str):
    """Verify GGUF by loading and running a test prompt."""
    try:
        from llama_cpp import Llama
        logger.info(f"Verifying: {gguf_path}")
        llm = Llama(model_path=gguf_path, n_gpu_layers=0, n_ctx=512, verbose=False)
        result = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful NPC."},
                {"role": "user", "content": "Hello!"},
            ],
            max_tokens=50, temperature=1.0, stop=["<turn|>"],
        )
        response = result["choices"][0]["message"]["content"]
        logger.info(f"  ✅ Verified: \"{response[:80]}...\"")
        del llm
        return True
    except Exception as e:
        logger.error(f"  ❌ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export to GGUF")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--quants", nargs="+", default=["Q4_K_M", "Q8_0"])
    parser.add_argument("--llama-cpp-dir", default="llama.cpp")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir or f"{args.model_path}/gguf")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = Path(args.model_path).name
    f16_path = str(output_dir / f"{model_name}-F16.gguf")

    # Step 1: Clone/check llama.cpp
    clone_llama_cpp(args.llama_cpp_dir)

    # Step 2: Convert to F16 GGUF
    convert_to_gguf(args.model_path, f16_path, args.llama_cpp_dir)

    # Step 3: Quantize
    results = {}
    for quant in args.quants:
        quant_path = str(output_dir / f"{model_name}-{quant}.gguf")
        quantize_gguf(f16_path, quant_path, quant, args.llama_cpp_dir)
        results[quant] = quant_path

    # Step 4: Verify
    if not args.skip_verify:
        for quant, path in results.items():
            verify_gguf(path)

    # Summary
    print("\n" + "=" * 60)
    print("GGUF EXPORT SUMMARY")
    print("=" * 60)
    for quant, path in results.items():
        size = Path(path).stat().st_size / 1e9
        print(f"  {quant:<12}: {path} ({size:.1f} GB)")
    print("\nExpected sizes for Gemma 4 12B:")
    print("  Q4_K_M: ~7.5 GB — RTX 3060/3070 compatible")
    print("  Q8_0:   ~13 GB  — RTX 3090/4090 recommended")
    print("  F16:    ~24 GB  — A100 or dual GPU required")


if __name__ == "__main__":
    main()
