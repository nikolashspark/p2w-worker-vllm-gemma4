# CUDA 12.4 runtime is a good baseline for L4 (sm_89)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch for CUDA 12.4 (cu124)
# If you prefer cu121, tell me and I’ll adjust.
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch torchvision torchaudio

# Install vLLM + server libs
# transformers is kept in <5 for stability; if you truly need bleeding-edge, see note below.
RUN pip install \
    "vllm==0.19.1" \
    runpod \
    "transformers>=4.45.0,<5" \
    accelerate \
    safetensors \
    huggingface_hub

# Copy your handler
COPY handler.py /handler.py

# Optional: model download auth
# ENV HF_TOKEN=...

# vLLM specific (NOTE: your earlier env var name is not recognized by vLLM 0.19.1)
# Keep trust_remote_code in your vLLM config/args instead of env var.
ENV VLLM_ENABLE_CUDA_COMPATIBILITY=1

CMD ["python3", "-u", "/handler.py"]
