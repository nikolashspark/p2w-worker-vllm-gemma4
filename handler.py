import os
import uuid

import runpod
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer


# ---------------------------
# Configuration (via env vars)
# ---------------------------
MODEL_ID = os.getenv("MODEL_ID", "google/gemma-4-E4B-it")

# NOTE:
# 32k context can be very VRAM-hungry due to KV cache.
# Use a safer default, and override only if you know your GPU has headroom.
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "16384"))

GPU_UTILIZATION = float(os.getenv("GPU_UTILIZATION", "0.90"))
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true"


# ---------------------------
# Engine init (once per worker)
# ---------------------------
engine_args = AsyncEngineArgs(
    model=MODEL_ID,
    max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=GPU_UTILIZATION,
    trust_remote_code=TRUST_REMOTE_CODE,
    enable_prefix_caching=True,
)

engine = AsyncLLMEngine.from_engine_args(engine_args)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=TRUST_REMOTE_CODE)


# ---------------------------
# Runpod handler
# ---------------------------
async def handler(job):
    job_input = job.get("input", {}) or {}

    messages = job_input.get("messages", []) or []
    system_prompt = job_input.get("system_prompt", "") or ""
    enable_thinking = bool(job_input.get("enable_thinking", True))

    # Ensure system prompt is present if provided separately
    if system_prompt:
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": system_prompt})

    # Build chat prompt from tokenizer template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Gemma 4 "thinking" channel fix
    if enable_thinking and not prompt.endswith("<|channel>thought\n"):
        prompt += "<|channel>thought\n"

    # Sampling parameters
    u_params = job_input.get("sampling_params", {}) or {}
    sampling_params = SamplingParams(
        max_tokens=int(u_params.get("max_tokens", 2048)),
        temperature=float(u_params.get("temperature", 0.2)),
        top_p=float(u_params.get("top_p", 0.95)),
        stop=u_params.get("stop", ["<|turn>user", "<|turn>model"]),
        skip_special_tokens=False,  # keep thinking tags visible
    )

    # Generate
    request_id = str(uuid.uuid4())
    results_generator = engine.generate(prompt, sampling_params, request_id)

    full_text = ""
    last_output = None

    async for request_output in results_generator:
        last_output = request_output

        # vLLM returns a list of outputs (n candidates). We use the first.
        if request_output.outputs and len(request_output.outputs) > 0:
            full_text = request_output.outputs[0].text
        else:
            full_text = ""

    # Safety: if something went wrong and nothing streamed back
    if last_output is None:
        return {
            "text": "",
            "usage": {"input": 0, "output": 0},
        }

    out_tokens = 0
    if last_output.outputs and len(last_output.outputs) > 0:
        out_tokens = len(last_output.outputs[0].token_ids)

    return {
        "text": full_text,
        "usage": {
            "input": len(last_output.prompt_token_ids),
            "output": out_tokens,
        },
    }


runpod.serverless.start({"handler": handler})
