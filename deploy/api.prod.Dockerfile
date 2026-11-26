# ---- SpamGuard API – bản production cho Docker daemon ở xa (192.168.1.5) ----
# Daemon ở xa không thấy ổ đĩa của máy chạy lệnh deploy -> không bind mount được.
# Vì vậy Model Registry được đóng gói vào image và khởi tạo named volume ở lần chạy đầu.
FROM python:3.12-slim

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
# Registry đã lọc (xem deploy/deploy.sh) – được copy vào named volume ở lần chạy đầu tiên
COPY deploy/.build/registry ./models/registry

RUN useradd --create-home --uid 1000 spamguard && mkdir -p /app/data \
    && chown -R spamguard /app
USER spamguard

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/ready')"
# --proxy-headers: tin X-Forwarded-Proto từ Nginx để URL redirect dùng https
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
