FROM vllm/vllm-openai:v0.19.1

# Оновлюємо залежності для сумісності з Gemma 4
RUN pip install --no-cache-dir runpod transformers>=5.5.3

COPY handler.py /handler.py

ENV VLLM_TRUST_REMOTE_CODE=1
ENV PYTHONUNBUFFERED=1

CMD ["python3", "-u", "/handler.py"]
