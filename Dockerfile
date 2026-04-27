# Base: CUDA 12.4 runtime (good baseline for L4 / sm_89)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# PyTorch (CUDA 12.4 build)
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch torchvision torchaudio

# vLLM + Runpod handler deps
# Pin Transformers exactly to 5.5.3 as requested
RUN pip install \
    "vllm==0.19.1" \
    runpod \
    "transformers==5.5.3" \
    accelerate \
    safetensors \
    huggingface_hub

# Copy your handler
COPY handler.py /handler.py

# Optional: Hugging Face auth token for private/gated models
# ENV HF_TOKEN=...

# vLLM runtime option
ENV VLLM_ENABLE_CUDA_COMPATIBILITY=1

CMD ["python3", "-u", "/handler.py"]
