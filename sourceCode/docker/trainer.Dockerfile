# ---- SpamGuard Trainer (train_all + feedback loop) ----
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    SPAMGUARD_ROOT=/app \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app
COPY requirements-api.txt requirements-train.txt ./
# torch bản CPU để image gọn (~1GB thay vì ~5GB bản CUDA)
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-train.txt

COPY configs ./configs
COPY src ./src
COPY scripts ./scripts

CMD ["python", "scripts/retrain_from_feedback.py"]
