"""
RunPod Serverless handler — thin OpenAI-compat proxy to vLLM's built-in server.

Why this architecture?
----------------------
vLLM ships a full OpenAI-compatible HTTP server at
`vllm.entrypoints.openai.api_server`. It supports streaming, tool calling,
structured output (json_schema via outlines/xgrammar), multimodal, logprobs —
everything AI SDK v6 expects. Reimplementing any of that ourselves is wasted
effort and fragile (tied to vLLM internals).

So: we spawn vLLM's server on 127.0.0.1:8000 as a subprocess, wait for
/health, then proxy every RunPod job's `openai_input` to it, streaming SSE
chunks back as RunPod-yield payloads. RunPod + `RAW_OPENAI_OUTPUT=1` env var
emits them as SSE to the client (AI SDK reads them natively).

Cold-start timeline (~60–120s for Gemma 4 E4B):
1. Container starts, CMD runs this file.
2. _start_vllm_server() spawns subprocess, model begins downloading/loading.
3. _wait_for_vllm_ready() polls :8000/health until 200 OK.
4. runpod.serverless.start(...) registers worker. RunPod dispatches queued jobs.
5. Per job: HTTP POST to local vLLM → yield chunks → client sees live tokens.

Env vars (все optional):
    MODEL_ID                google/gemma-4-E4B-it   HF repo id (or local path)
    MAX_MODEL_LEN           16384                   Max context length
    GPU_UTILIZATION         0.90                    vLLM GPU mem fraction
    TRUST_REMOTE_CODE       true                    --trust-remote-code
    ENABLE_PREFIX_CACHING   true                    vLLM prefix cache for shared system prompts
    TOOL_CALL_PARSER        hermes                  vLLM tool-call-parser (hermes|pythonic|llama3_json|...)
    DISABLE_TOOL_CHOICE     false                   Set true to NOT enable auto tool choice
    VLLM_PORT               8000                    Internal subprocess port
    VLLM_READY_TIMEOUT_SEC  300                     How long to wait for /health after spawn
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import AsyncGenerator

import aiohttp
import runpod


# ---------------------------------------------------------------------------
# Config — все читаємо один раз при старті
# ---------------------------------------------------------------------------

MODEL_ID = os.getenv("MODEL_ID", "google/gemma-4-E4B-it")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "16384"))
GPU_UTILIZATION = float(os.getenv("GPU_UTILIZATION", "0.90"))
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true"
ENABLE_PREFIX_CACHING = os.getenv("ENABLE_PREFIX_CACHING", "true").lower() == "true"
TOOL_CALL_PARSER = os.getenv("TOOL_CALL_PARSER", "hermes")
DISABLE_TOOL_CHOICE = os.getenv("DISABLE_TOOL_CHOICE", "false").lower() == "true"
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
VLLM_BASE_URL = f"http://127.0.0.1:{VLLM_PORT}"
VLLM_READY_TIMEOUT_SEC = int(os.getenv("VLLM_READY_TIMEOUT_SEC", "300"))

_vllm_proc: subprocess.Popen | None = None


# ---------------------------------------------------------------------------
# vLLM subprocess lifecycle
# ---------------------------------------------------------------------------

def _build_vllm_cmd() -> list[str]:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_ID,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_UTILIZATION),
        "--host", "127.0.0.1",
        "--port", str(VLLM_PORT),
        # served-model-name сапожно дорівнює MODEL_ID — так AI SDK/клієнт
        # зможуть відправляти ту саму строку у `model` полі, що ми й бачимо
        # у логах. Якщо хочеш псевдонім — додай env var нижче.
        "--served-model-name", os.getenv("SERVED_MODEL_NAME", MODEL_ID),
    ]
    if TRUST_REMOTE_CODE:
        cmd.append("--trust-remote-code")
    if ENABLE_PREFIX_CACHING:
        cmd.append("--enable-prefix-caching")
    if not DISABLE_TOOL_CHOICE:
        # Вмикає tool calling (OpenAI-compat function calling). На reasoning
        # response parsing (thinking channel) це не впливає — це про tools.
        cmd.extend(["--enable-auto-tool-choice", "--tool-call-parser", TOOL_CALL_PARSER])
    return cmd


def _start_vllm_server() -> subprocess.Popen:
    cmd = _build_vllm_cmd()
    print(f"[handler] spawning vLLM: {' '.join(cmd)}", flush=True)
    # stdout/stderr inherit → logs стрімляться у RunPod console (критично
    # щоб бачити OOM/template errors).
    return subprocess.Popen(cmd)


def _wait_for_vllm_ready(timeout: int) -> None:
    health_url = f"{VLLM_BASE_URL}/health"
    deadline = time.time() + timeout
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                if resp.status == 200:
                    print(f"[handler] vLLM /health OK after {attempt}s", flush=True)
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        if _vllm_proc and _vllm_proc.poll() is not None:
            raise RuntimeError(
                f"vLLM subprocess exited prematurely with code {_vllm_proc.returncode} "
                f"(after {attempt}s). Check logs above for the real cause."
            )
        if attempt % 10 == 0:
            print(f"[handler] waiting for vLLM... ({attempt}s / {timeout}s)", flush=True)
        time.sleep(1)
    raise RuntimeError(f"vLLM server not ready after {timeout}s")


def _shutdown_vllm(*_):
    global _vllm_proc
    if _vllm_proc and _vllm_proc.poll() is None:
        print("[handler] terminating vLLM subprocess...", flush=True)
        _vllm_proc.terminate()
        try:
            _vllm_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("[handler] vLLM did not terminate in 30s, killing", flush=True)
            _vllm_proc.kill()


# ---------------------------------------------------------------------------
# RunPod handler — async generator proxying to local vLLM
# ---------------------------------------------------------------------------

async def handler(job) -> AsyncGenerator[dict, None]:
    job_input = job.get("input", {}) or {}

    # Два підтримуваних формати:
    #   1) {"input": {"openai_route": "/v1/chat/completions", "openai_input": <body>}}
    #      — так RunPod gateway перетворює HTTP POST до /openai/v1/chat/completions
    #   2) {"input": <body>}
    #      — fallback для прямого /run з уже OpenAI-shaped body (зручно для тестів)
    openai_input = job_input.get("openai_input") or job_input
    route = (job_input.get("openai_route") or "/v1/chat/completions").strip()

    if not isinstance(openai_input, dict) or "messages" not in openai_input:
        yield {"error": "expected OpenAI-shaped body with `messages`. "
                        "Use /openai/v1/chat/completions route, or put OpenAI body under input.openai_input."}
        return

    url = f"{VLLM_BASE_URL}{route}"
    is_stream = bool(openai_input.get("stream", False))

    # Timeout=None бо streaming може йти довго при великих max_tokens. Connect
    # timeout ставимо явний — якщо vLLM помер, ми маємо це помітити швидко.
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                url,
                json=openai_input,
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    yield {
                        "error": f"vLLM returned HTTP {resp.status}",
                        "body_preview": body[:2000],
                    }
                    return

                if is_stream:
                    # Парсимо SSE-потік з vLLM і yield-имо кожен chunk як dict.
                    # RunPod з RAW_OPENAI_OUTPUT=1 пересилає їх клієнту як SSE
                    # `data: {...}` — тобто AI SDK бачить стандартний OpenAI
                    # chat.completion.chunk stream.
                    buffer = b""
                    async for chunk in resp.content.iter_any():
                        buffer += chunk
                        while b"\n\n" in buffer:
                            event_bytes, buffer = buffer.split(b"\n\n", 1)
                            for raw_line in event_bytes.split(b"\n"):
                                line = raw_line.strip()
                                if not line or not line.startswith(b"data:"):
                                    continue
                                data_str = line[5:].strip().decode("utf-8", errors="replace")
                                if data_str == "[DONE]":
                                    return
                                try:
                                    yield json.loads(data_str)
                                except json.JSONDecodeError:
                                    # vLLM іноді шле порожні keep-alive commenti —
                                    # ігноруємо без галасу.
                                    continue
                else:
                    # Non-streaming path (generateText, не streamText).
                    yield await resp.json()
    except asyncio.CancelledError:
        # Клієнт (AI SDK stop() / user закрив таб) → RunPod відміняє job →
        # вилітає CancelledError. Нехай прокинеться вище, щоб aiohttp закрив
        # connection і vLLM отримав client disconnect (engine.abort() зчитує це).
        print(f"[handler] cancelled job {job.get('id')}", flush=True)
        raise


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Коли RunPod зупиняє контейнер, SIGTERM має дійти і до vLLM subprocess-а,
    # інакше KV-cache/HF-downloads лишаються у напіврозкладеному стані.
    signal.signal(signal.SIGTERM, _shutdown_vllm)
    signal.signal(signal.SIGINT, _shutdown_vllm)

    _vllm_proc = _start_vllm_server()
    try:
        _wait_for_vllm_ready(VLLM_READY_TIMEOUT_SEC)
    except Exception:
        _shutdown_vllm()
        raise

    print(f"[handler] vLLM is ready, registering RunPod worker", flush=True)
    runpod.serverless.start({
        "handler": handler,
        "return_aggregate_stream": True,  # non-streaming клієнти отримають склеєний output
    })