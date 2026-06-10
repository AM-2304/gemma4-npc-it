#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
OpenAI-Compatible REST API for Gemma4NPC.
Game engines can use the standard OpenAI client library to call this server.

Usage:
    python serving/openai_compatible_wrapper.py --model-path path/to/gguf --port 8080

Game engine integration (any language with OpenAI SDK):
    client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
    response = client.chat.completions.create(
        model="gemma4npc-it",
        messages=[{"role": "system", "content": npc_system_prompt}, ...],
    )
"""

import argparse
import json
import logging
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Gemma4NPC API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model reference
llm = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "gemma4npc-it"
    messages: list[ChatMessage]
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 256
    stream: bool = False
    stop: Optional[list[str]] = None
    response_format: Optional[dict] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": llm is not None}


@app.get("/v1/models")
async def list_models():
    return {
        "data": [{"id": "gemma4npc-it", "object": "model", "owned_by": "gemma4npc"}],
        "object": "list",
    }


@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    stop = request.stop or ["<end_of_turn>", "<eos>", "<eos_token>"]

    if request.stream:
        async def stream_generator():
            for chunk in llm.create_chat_completion(
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=stop,
                stream=True,
                response_format=request.response_format,
            ):
                delta = chunk["choices"][0].get("delta", {})
                data = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    result = llm.create_chat_completion(
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=stop,
        response_format=request.response_format,
    )

    choice = result["choices"][0]
    usage = result.get("usage", {})

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content=choice["message"]["content"]),
            finish_reason=choice.get("finish_reason", "stop"),
        )],
        usage=Usage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        ),
    )


def main():
    global llm

    parser = argparse.ArgumentParser(description="Gemma4NPC OpenAI-compatible API")
    parser.add_argument("--model-path", required=True, help="Path to GGUF model")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    args = parser.parse_args()

    from llama_cpp import Llama

    logger.info(f"Loading model: {args.model_path}")
    llm = Llama(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        chat_format="gemma",
        verbose=False,
    )
    logger.info("Model loaded!")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
