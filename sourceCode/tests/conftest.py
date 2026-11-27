"""Fixture dùng chung: registry tạm chứa 1 model SVM tí hon -> test API không cần model thật."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "services" / "api")]
# Đặt trước khi collect test: app.auth đọc biến môi trường ngay lúc import
os.environ["ADMIN_TOKEN"] = "test-token"
os.environ["ADMIN_USERNAME"] = "admin"
os.environ["ADMIN_PASSWORD"] = "1234"

TINY_CORPUS = {
    "ham": ["see you at lunch tomorrow", "mai đi học nhé bạn", "ok thanks for the update",
            "hôm nay trời đẹp quá", "call me when you get home", "sản phẩm dùng tốt, giao nhanh",
            "meeting moved to 3pm", "cảm ơn shop nhiều nha", "happy birthday my friend",
            "tối nay ăn gì vậy", "let us watch a movie tonight", "mình đã nhận được hàng rồi"],
    "spam": ["win cash prize now call", "free entry win a prize claim now", "trúng thưởng iphone nhấn link ngay",
             "khuyến mãi khủng nhận quà miễn phí", "urgent you have won cash prize", "click link to claim free gift",
             "nhận ngay voucher miễn phí bấm link", "congratulations you won free cash", "vay tiền nhanh lãi suất 0",
             "free ringtone text win now", "nạp thẻ nhận quà trúng thưởng", "claim your free prize today"],
    "toxic": ["you are a stupid idiot", "đồ ngu như bò", "shut up you dumb loser", "mày là thằng ngu",
              "you stupid worthless trash", "câm mồm đi đồ ngu", "idiot loser go away", "thằng óc chó",
              "you are pathetic and stupid", "đồ mất dạy ngu si", "dumb idiot trash", "ngu như lợn"],
}


@pytest.fixture(scope="session")
def tiny_registry(tmp_path_factory) -> Path:
    from spamguard.models.classical import train_classical
    from spamguard.registry import ModelRegistry

    registry_dir = tmp_path_factory.mktemp("registry")
    texts = [t for label in TINY_CORPUS for t in TINY_CORPUS[label]]
    labels = [label for label in TINY_CORPUS for _ in TINY_CORPUS[label]]
    cfg = {"C": 1.0, "word_ngram_max": 1, "char_ngram_max": 3, "max_word_features": 500, "max_char_features": 500}
    artifact = train_classical("svm", cfg, 42, texts, labels, ["ham", "spam", "toxic"],
                               tmp_path_factory.mktemp("artifact") / "svm")
    registry = ModelRegistry(registry_dir)
    version = registry.register("svm", artifact, {"macro_f1": 1.0, "p95_latency_ms": 1.0}, cfg)
    registry.promote(version, reason="test")
    return registry_dir


@pytest.fixture(scope="session")
def client(tiny_registry, tmp_path_factory):
    os.environ["SPAMGUARD_REGISTRY_DIR"] = str(tiny_registry)
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path_factory.mktemp('db') / 'test.db'}"
    from app.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client
