"""Prometheus metrics: vận hành (latency, lỗi) + chất lượng model (phân phối nhãn, confidence)."""

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("spamguard_http_requests_total", "Số HTTP request", ["method", "path", "status"])
REQUEST_LATENCY = Histogram(
    "spamguard_http_request_duration_seconds", "Độ trễ HTTP request", ["path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
INFERENCE_LATENCY = Histogram(
    "spamguard_inference_duration_seconds", "Độ trễ suy luận model / tin nhắn",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)
PREDICTIONS = Counter("spamguard_predictions_total", "Số tin nhắn đã phân loại",
                      ["label", "action", "channel"])
CONFIDENCE = Histogram("spamguard_prediction_confidence", "Phân phối confidence (theo dõi drift)",
                       ["label"], buckets=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0))
FEEDBACK = Counter("spamguard_feedback_total", "Số feedback nhận được", ["label", "source"])
MODEL_INFO = Gauge("spamguard_model_info", "Model đang phục vụ (value=1)", ["version", "model_type"])
MODEL_RELOADS = Counter("spamguard_model_reloads_total", "Số lần hot-reload model")
