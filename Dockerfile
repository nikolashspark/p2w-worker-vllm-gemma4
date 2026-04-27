# Base: CUDA 12.4 runtime (good baseline for L4 / sm_89)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    # (Optional but helps Triton/Inductor reliably find a compiler)
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++

# System dependencies
# - python3-dev/python3.10-dev: provides Python.h needed by Triton/Inductor JIT compilation
# - build-essential: gcc/g++ and build tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    python3-dev python3.10-dev \
    git curl ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Make sure `python` and `pip` exist (some tools expect these)
RUN ln -sf /usr/bin/python3 /usr/local/bin/python && \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip

# Upgrade pip tooling
RUN python3 -m pip install --upgrade pip setuptools wheel

# PyTorch (CUDA 12.4 wheels)
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch torchvision torchaudio

# vLLM + Runpod handler deps
# Keep your Transformers pin
RUN pip install \
    "vllm==0.19.1" \
    runpod \
    "transformers==5.5.3" \
    accelerate \
    safetensors \
    huggingface_hub

# vLLM runtime option
ENV VLLM_ENABLE_CUDA_COMPATIBILITY=1

# Copy handler
COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
