"""Quản lý model production trong API: load từ registry + hot-reload không downtime."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from spamguard.config import get_labels
from spamguard.inference import Predictor, load_predictor
from spamguard.registry import ModelRegistry

logger = logging.getLogger(__name__)

# Tần suất tối đa kiểm tra registry.json thay đổi (giây)
_RELOAD_CHECK_INTERVAL_SECONDS = 10


@dataclass(frozen=True)
class LoadedModel:
    version: int
    model_type: str
    predictor: Predictor


class ModelManager:
    def __init__(self, registry: ModelRegistry | None = None):
        self.registry = registry or ModelRegistry()
        self.labels = get_labels()
        self._lock = threading.Lock()
        self._current: LoadedModel | None = None
        self._seen_mtime = 0.0
        self._last_check = 0.0

    @property
    def current(self) -> LoadedModel:
        if self._current is None:
            raise RuntimeError("Chưa có model production trong registry")
        return self._current

    @property
    def is_ready(self) -> bool:
        return self._current is not None

    def load_production(self, force: bool = False) -> bool:
        """Load model theo alias production. Trả về True nếu đã đổi model."""
        version = self.registry.get_alias("production")
        if version is None:
            logger.warning("Registry chưa có alias production")
            return False
        if not force and self._current and self._current.version == version:
            return False
        meta = self.registry.get_meta(version)
        predictor = load_predictor(meta["model_type"], self.registry.artifact_dir(version), self.labels, meta)
        predictor.predict_proba(["warmup"])  # warm-up: tránh request đầu tiên bị chậm
        with self._lock:
            # Swap nguyên tử: request đang chạy vẫn dùng object cũ cho tới khi xong
            self._current = LoadedModel(version, meta["model_type"], predictor)
        self._seen_mtime = self.registry.index_mtime()
        logger.info("Đang phục vụ model v%d (%s)", version, meta["model_type"])
        return True

    def maybe_reload(self) -> None:
        now = time.monotonic()
        if now - self._last_check < _RELOAD_CHECK_INTERVAL_SECONDS:
            return
        self._last_check = now
        if self.registry.index_mtime() != self._seen_mtime:
            try:
                self.load_production()
            except Exception:  # noqa: BLE001 - giữ model cũ nếu model mới lỗi
                logger.exception("Hot-reload thất bại, tiếp tục dùng model cũ")
                self._seen_mtime = self.registry.index_mtime()

    def predict(self, texts: list[str]) -> tuple[LoadedModel, np.ndarray]:
        model = self.current
        return model, model.predictor.predict_proba(texts)
