import os
import runpod
import uuid
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer

# --- 1. Ініціалізація Двигуна (Налаштування з RunPod UI) ---
MODEL_ID = os.getenv("MODEL_ID", "google/gemma-4-E4B-it")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "32768"))
GPU_UTILIZATION = float(os.getenv("GPU_UTILIZATION", "0.92"))

engine_args = AsyncEngineArgs(
    model=MODEL_ID,
    max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=GPU_UTILIZATION,
    trust_remote_code=True,
    enable_prefix_caching=True,
)

engine = AsyncLLMEngine.from_engine_args(engine_args)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

async def handler(job):
    job_input = job['input']
    
    # --- 2. Обробка Промпту (Налаштування з твого UI) ---
    messages = job_input.get("messages", [])
    system_prompt = job_input.get("system_prompt", "")
    enable_thinking = job_input.get("enable_thinking", True)

    # Додаємо системний промпт, якщо він переданий окремо
    if system_prompt:
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": system_prompt})

    # Формування шаблону чату
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Фікс для Thinking каналу Gemma 4 E4B
    if enable_thinking and not prompt.endswith("<|channel>thought\n"):
        prompt += "<|channel>thought\n"

    # --- 3. Параметри Генерації (Налаштування з твого UI) ---
    u_params = job_input.get("sampling_params", {})
    
    sampling_params = SamplingParams(
        max_tokens=u_params.get("max_tokens", 4096),
        temperature=u_params.get("temperature", 0.2),
        top_p=u_params.get("top_p", 0.95),
        stop=u_params.get("stop", ["<|turn>user", "<|turn>model"]),
        skip_special_tokens=False # Потрібно для бачення тегів мислення
    )

    # --- 4. Виконання ---
    request_id = str(uuid.uuid4())
    results_generator = engine.generate(prompt, sampling_params, request_id)
    
    full_text = ""
    async for request_output in results_generator:
        full_text = request_output.outputs.text
    
    return {
        "text": full_text,
        "usage": {
            "input": len(request_output.prompt_token_ids),
            "output": len(request_output.outputs.token_ids)
        }
    }

runpod.serverless.start({"handler": handler})
