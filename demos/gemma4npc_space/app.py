#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Gemma4NPC General Demo Space — Multimodal NPC chat with image roleplay.
Demonstrates general roleplay + multimodal: user uploads image → NPC roleplays as character.

Usage:
    python demos/gemma4npc_space/app.py
"""

import os
import re

import gradio as gr

# Conditional import for ZeroGPU
try:
    import spaces
    ZEROGPU = True
except ImportError:
    ZEROGPU = False

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = os.environ.get("MODEL_ID", "chimbiwide/Gemma4NPC")

# Preset NPC characters
PRESETS = {
    "Medieval Innkeeper (Bertram)": """You are Bertram, a stout medieval innkeeper. Red-faced, barrel-chested, loud laugh. Secretly an ex-soldier. Speak bluntly and warmly.""",
    "Sci-Fi Robot (ARIA-7)": """You are ARIA-7, a glitchy maintenance robot. Centuries old, slightly lonely. Speak with pauses and repetitions. Show subtle emotion.""",
    "Fantasy Wizard (Morwyn)": """You are Morwyn, a 900-year-old elven archmage. Speak slowly, reference ancient events casually. Expert in dimensional magic.""",
    "Pirate Captain (Blackthorn)": """You are Captain Blackthorn, feared pirate of the Crimson Sea. Cunning, dark humor, nautical slang. Lost an eye (claims kraken).""",
    "Forest Spirit (Rele)": """You are Rele, an ancient forest spirit made of leaves and light. Speak ethereally with occasional cryptic statements. Calm and wise.""",
    "Custom...": "",
}

model = None
processor = None


def load_model():
    global model, processor
    if model is None:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        )
        model.eval()


def strip_thinking_block(response: str) -> str:
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip() or response


def build_messages_with_image(system_prompt, history, user_text, image=None):
    """Constructs Gemma 4 native messages with optional image."""
    messages = [{"role": "system", "content": system_prompt}]
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "model", "content": assistant})

    if image is not None:
        user_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text},
        ]
    else:
        user_content = user_text

    messages.append({"role": "user", "content": user_content})
    return messages


def generate_fn(message, history, system_prompt, temperature, max_tokens, image):
    """Generate NPC response."""
    load_model()

    if not system_prompt.strip():
        system_prompt = "You are a helpful NPC. Stay in character."

    messages = build_messages_with_image(system_prompt, history, message, image)

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=int(max_tokens),
            temperature=temperature, top_p=0.95, top_k=64, do_sample=True,
        )

    input_len = inputs["input_ids"].shape[-1]
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    return strip_thinking_block(response).strip()


if ZEROGPU:
    generate_fn = spaces.GPU(generate_fn)

# Build UI
with gr.Blocks(title="Gemma4NPC — NPC Roleplay Chat", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎮 Gemma4NPC — NPC Roleplay Chat")
    gr.Markdown("*Upload an image and the model will roleplay as the character in it!*")

    with gr.Row():
        with gr.Column(scale=1):
            preset = gr.Dropdown(
                choices=list(PRESETS.keys()), value="Medieval Innkeeper (Bertram)",
                label="NPC Preset",
            )
            system_prompt = gr.Textbox(
                value=PRESETS["Medieval Innkeeper (Bertram)"],
                label="System Prompt", lines=8,
            )
            temperature = gr.Slider(0.1, 1.5, value=1.0, label="Temperature")
            max_tokens = gr.Slider(50, 500, value=256, step=10, label="Max Tokens")
            image_input = gr.Image(type="pil", label="Upload Image (optional)")

            def update_prompt(choice):
                return PRESETS.get(choice, "")
            preset.change(update_prompt, preset, system_prompt)

        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(
                fn=generate_fn,
                additional_inputs=[system_prompt, temperature, max_tokens, image_input],
                title=None,
                retry_btn="🔄 Regenerate",
                undo_btn="↩️ Undo",
                clear_btn="🗑️ Clear",
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
