# ---- SpamGuard API (serving) ----
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app/services/api \
    SPAMGUARD_ROOT=/app

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY configs ./configs
COPY src ./src
COPY services/api ./services/api
COPY admin-web ./admin-web

# Chạy bằng user không phải root
RUN useradd --create-home --uid 1000 spamguard && mkdir -p /app/models /app/data \
    && chown -R spamguard /app
USER spamguard

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/ready')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
