"""Orchestrate: train -> export -> evaluate (trên artifact serving) -> track MLflow -> register."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from spamguard.config import PROJECT_ROOT, get_labels, load_params, resolve_path
from spamguard.evaluation import artifact_size_mb, classification_metrics, measure_latency
from spamguard.inference import load_predictor
from spamguard.registry import ModelRegistry

logger = logging.getLogger(__name__)

MODEL_TYPES = ["svm", "random_forest", "bigru", "distilbert"]
MODEL_DISPLAY_NAMES = {"svm": "SVM (TF-IDF)", "random_forest": "Random Forest (TF-IDF)",
                       "bigru": "Bi-GRU", "distilbert": "DistilBERT"}


def load_splits() -> dict[str, pd.DataFrame]:
    processed = resolve_path(load_params()["data"]["processed_dir"])
    return {name: pd.read_csv(processed / f"{name}.csv", keep_default_na=False)
            for name in ("train", "val", "test")}


def _mlflow():
    """MLflow là tuỳ chọn: thiếu package/server thì pipeline vẫn chạy (chỉ log cảnh báo)."""
    try:
        import mlflow
    except ImportError:
        logger.warning("Không có mlflow -> bỏ qua experiment tracking")
        return None
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "spamguard"))
    return mlflow


def train_one(model_type: str, train_df: pd.DataFrame, val_df: pd.DataFrame, output_dir: Path,
              sample_weight: np.ndarray | None = None) -> dict:
    params = load_params()
    cfg, seed, labels = params["models"][model_type], params["seed"], get_labels()
    texts, y = train_df["text"].tolist(), train_df["label"].tolist()
    info: dict = {}
    start = time.perf_counter()
    if model_type in ("svm", "random_forest"):
        from spamguard.models.classical import train_classical
        # Cố ý chỉ fit trên train (không gộp val) để so sánh công bằng với 2 model deep learning
        train_classical(model_type, cfg, seed, texts, y, labels, output_dir, sample_weight)
    elif model_type == "bigru":
        from spamguard.models.bigru import train_bigru
        _, info = train_bigru(cfg, seed, texts, y, val_df["text"].tolist(), val_df["label"].tolist(),
                              labels, output_dir, sample_weight)
    elif model_type == "distilbert":
        from spamguard.models.distilbert import train_distilbert
        _, info = train_distilbert(cfg, seed, texts, y, val_df["text"].tolist(), val_df["label"].tolist(),
                                   labels, output_dir, sample_weight)
    else:
        raise ValueError(model_type)
    info["train_seconds"] = time.perf_counter() - start
    return info


def evaluate_artifact(model_type: str, artifact_dir: Path, test_df: pd.DataFrame, params: dict) -> dict:
    labels = get_labels()
    predictor = load_predictor(model_type, artifact_dir, labels, {"params": params})
    proba = predictor.predict_proba(test_df["text"].tolist())
    metrics = classification_metrics(test_df["label"].tolist(), proba, labels, test_df["lang"].tolist())
    metrics.update(measure_latency(predictor, test_df["text"].tolist()))
    metrics["model_size_mb"] = artifact_size_mb(artifact_dir)
    return metrics


def _flat_metrics(metrics: dict) -> dict[str, float]:
    flat = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    for label, values in metrics.get("per_class", {}).items():
        flat.update({f"{label}_{k}": v for k, v in values.items() if k != "support"})
    flat.update({f"f1_{lang}": v for lang, v in metrics.get("per_lang_macro_f1", {}).items()})
    return flat


def train_evaluate_register(model_type: str, splits: dict[str, pd.DataFrame], data_version: str,
                            source: str = "train_all", sample_weight: np.ndarray | None = None,
                            train_df: pd.DataFrame | None = None, extra_eval: dict | None = None) -> tuple[int, dict]:
    """Chạy trọn vòng đời 1 model và trả về (registry_version, metrics)."""
    params = load_params()
    train_df = train_df if train_df is not None else splits["train"]
    mlflow = _mlflow()
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = Path(tmp) / model_type
        info = train_one(model_type, train_df, splits["val"], artifact_dir, sample_weight)
        metrics = evaluate_artifact(model_type, artifact_dir, splits["test"], params["models"][model_type])
        metrics["train_seconds"] = info["train_seconds"]
        if extra_eval:
            metrics.update(extra_eval(model_type, artifact_dir))
        run_id = None
        if mlflow is not None:
            with mlflow.start_run(run_name=f"{model_type}-{source}") as run:
                run_id = run.info.run_id
                mlflow.set_tags({"model_type": model_type, "data_version": data_version, "source": source})
                mlflow.log_params({f"{model_type}.{k}": v for k, v in params["models"][model_type].items()})
                mlflow.log_param("train_rows", len(train_df))
                mlflow.log_metrics(_flat_metrics(metrics))
                for epoch in info.get("history", []):
                    mlflow.log_metric("val_macro_f1_epoch", epoch["val_macro_f1"], step=epoch["epoch"])
                    mlflow.log_metric("train_loss_epoch", epoch["train_loss"], step=epoch["epoch"])
                mlflow.log_dict(metrics, "metrics.json")
        version = ModelRegistry().register(
            model_type, artifact_dir, metrics, params["models"][model_type],
            {"data_version": data_version, "mlflow_run_id": run_id, "source": source,
             "train_rows": len(train_df), "history": info.get("history", [])},
        )
        if mlflow is not None and run_id:
            with mlflow.start_run(run_id=run_id):
                mlflow.set_tag("registry_version", version)
    logger.info("Đã đăng ký %s thành v%d macro_f1=%.4f p95=%.1fms", model_type, version,
                metrics["macro_f1"], metrics["p95_latency_ms"])
    return version, metrics


def select_champion(results: dict[str, dict]) -> str:
    """Chọn model có metric cao nhất trong số model thoả SLA độ trễ."""
    selection = load_params()["selection"]
    eligible = {m: r for m, r in results.items() if r["p95_latency_ms"] <= selection["max_p95_latency_ms"]}
    pool = eligible or results
    return max(pool, key=lambda m: pool[m][selection["metric"]])


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float))
