FROM vllm/vllm-openai:v0.19.1

# Скидаємо вхідну точку vLLM, щоб дозволити запуск нашого handler
ENTRYPOINT []

# Оновлюємо залежності
RUN pip install --no-cache-dir runpod transformers>=5.5.3

# Копіюємо код обробника
COPY handler.py /handler.py

# Налаштування оточення
ENV VLLM_TRUST_REMOTE_CODE=1
ENV PYTHONUNBUFFERED=1

# Тепер ця команда виконається коректно як запуск Python-скрипта
CMD ["python3", "-u", "/handler.py"]
