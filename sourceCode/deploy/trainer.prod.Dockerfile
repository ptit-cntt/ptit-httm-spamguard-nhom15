# Trainer cho Feedback Loop trên daemon ở xa: dữ liệu đã xử lý được COPY vào image
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    SPAMGUARD_ROOT=/app \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app
COPY requirements-api.txt requirements-train.txt ./
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-train.txt

COPY configs ./configs
COPY src ./src
COPY scripts ./scripts
COPY data/processed ./data/processed

CMD ["python", "scripts/retrain_scheduler.py"]
