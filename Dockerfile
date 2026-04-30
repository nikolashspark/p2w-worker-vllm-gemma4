FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    python3-dev python3.10-dev \
    git curl ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python && \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch torchvision torchaudio

RUN pip install \
    "vllm==0.19.1" \
    runpod \
    aiohttp \
    "transformers==5.5.3" \
    accelerate \
    safetensors \
    huggingface_hub

ENV VLLM_ENABLE_CUDA_COMPATIBILITY=1

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]