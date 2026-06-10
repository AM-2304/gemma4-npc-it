#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Local Gradio Chat — General NPC roleplay interface.

Usage:
    python demos/gradio_chat/app.py --model-path outputs/Gemma4NPC-it/merged_float16
"""

import argparse
import re

import gradio as gr
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def strip_thinking(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip() or text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    def generate(message, history, system_prompt, temperature, max_tokens):
        if not system_prompt.strip():
            system_prompt = "You are a helpful NPC."

        messages = [{"role": "system", "content": system_prompt}]
        for human, assistant in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "model", "content": assistant})
        messages.append({"role": "user", "content": message})

        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs, max_new_tokens=int(max_tokens),
                temperature=temperature, top_p=0.95, do_sample=True
            )

        input_len = inputs["input_ids"].shape[-1]
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        return strip_thinking(response).strip()

    with gr.Blocks(title="Local Gemma4NPC Chat") as demo:
        gr.Markdown("# Local Gemma4NPC Chat")
        
        with gr.Row():
            with gr.Column(scale=1):
                sys_prompt = gr.Textbox(
                    label="System Prompt (Define NPC)",
                    value="You are Bertram, a stout medieval innkeeper.",
                    lines=5
                )
                temp = gr.Slider(0.1, 1.5, value=1.0, label="Temperature")
                tokens = gr.Slider(50, 500, value=200, label="Max Tokens")
                
            with gr.Column(scale=2):
                gr.ChatInterface(
                    fn=generate,
                    additional_inputs=[sys_prompt, temp, tokens],
                    title=None
                )

    demo.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
