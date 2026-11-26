"""Model Registry dạng file (tương thích khái niệm của MLflow Model Registry).

Cấu trúc:
    models/registry/
      registry.json            # danh sách version + alias "production"
      v1/ meta.json + artifact # mỗi version là 1 thư mục bất biến (immutable)

Lý do không phụ thuộc hẳn vào MLflow server lúc serving: API chỉ cần đọc
file -> không có single point of failure; MLflow vẫn được dùng để tracking
thí nghiệm và registry song song (xem training/tracking.py).
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spamguard.config import load_params, resolve_path

REGISTRY_FILENAME = "registry.json"
META_FILENAME = "meta.json"


class ModelRegistry:
    def __init__(self, root: Path | None = None):
        # SPAMGUARD_REGISTRY_DIR cho phép container/test trỏ tới registry khác
        configured = os.getenv("SPAMGUARD_REGISTRY_DIR") or load_params()["registry"]["dir"]
        self.root = root or resolve_path(configured)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / REGISTRY_FILENAME

    @contextmanager
    def _locked(self):
        # Khoá file để tránh race condition khi retrain job và admin promote cùng lúc
        with (self.root / ".lock").open("w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _read_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {"model_name": load_params()["registry"]["model_name"], "versions": [],
                    "aliases": {}, "history": []}
        return json.loads(self.index_path.read_text())

    def _write_index(self, index: dict[str, Any]) -> None:
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, indent=2, ensure_ascii=False))
        tmp.replace(self.index_path)  # atomic rename -> API không bao giờ đọc file dở dang

    def register(self, model_type: str, artifact_dir: Path, metrics: dict, params: dict,
                 extra: dict | None = None) -> int:
        with self._locked():
            index = self._read_index()
            version = max((v["version"] for v in index["versions"]), default=0) + 1
            target = self.root / f"v{version}"
            shutil.copytree(artifact_dir, target)
            meta = {
                "version": version, "model_type": model_type, "metrics": metrics,
                "params": params, "created_at": datetime.now(timezone.utc).isoformat(),
                **(extra or {}),
            }
            (target / META_FILENAME).write_text(json.dumps(meta, indent=2, ensure_ascii=False))
            index["versions"].append({
                "version": version, "model_type": model_type, "created_at": meta["created_at"],
                "macro_f1": metrics.get("macro_f1"), "p95_latency_ms": metrics.get("p95_latency_ms"),
                "source": (extra or {}).get("source", "train_all"),
            })
            self._write_index(index)
            return version

    def promote(self, version: int, alias: str = "production", reason: str = "") -> None:
        with self._locked():
            index = self._read_index()
            if not any(v["version"] == version for v in index["versions"]):
                raise KeyError(f"Version {version} không tồn tại")
            previous = index["aliases"].get(alias)
            index["aliases"][alias] = version
            index["history"].append({
                "alias": alias, "from": previous, "to": version, "reason": reason,
                "at": datetime.now(timezone.utc).isoformat(),
            })
            self._write_index(index)

    def get_alias(self, alias: str = "production") -> int | None:
        return self._read_index()["aliases"].get(alias)

    def get_meta(self, version: int) -> dict:
        return json.loads((self.root / f"v{version}" / META_FILENAME).read_text())

    def artifact_dir(self, version: int) -> Path:
        return self.root / f"v{version}"

    def list_versions(self) -> dict[str, Any]:
        return self._read_index()

    def index_mtime(self) -> float:
        return self.index_path.stat().st_mtime if self.index_path.exists() else 0.0
