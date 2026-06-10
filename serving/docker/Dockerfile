FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir llama-cpp-python fastapi uvicorn

# Copy application code
COPY serving/ ./serving/
COPY inference/ ./inference/
COPY configs/ ./configs/

# Model path (mount at runtime or set via build arg)
ARG MODEL_PATH="models/Gemma4NPC-it-Q4_K_M.gguf"
ENV MODEL_PATH=${MODEL_PATH}

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python3", "serving/openai_compatible_wrapper.py", "--model-path", "${MODEL_PATH}", "--port", "8080"]
