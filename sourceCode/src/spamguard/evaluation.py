"""Đánh giá model trên chính artifact được serving (ONNX/joblib) -> số đo phản ánh production."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score

from spamguard.inference import Predictor

_LATENCY_SAMPLES = 300
_WARMUP_CALLS = 10
_THROUGHPUT_BATCH = 64


def classification_metrics(y_true: list[str], proba: np.ndarray, labels: list[str],
                           langs: list[str] | None = None) -> dict:
    y_pred = [labels[i] for i in proba.argmax(1)]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted"),
        "roc_auc_ovr": roc_auc_score(pd.get_dummies(pd.Categorical(y_true, categories=labels)).values,
                                     proba, average="macro", multi_class="ovr"),
        "per_class": {label: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
                      for label, p, r, f, s in zip(labels, precision, recall, f1, support)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    # Tỉ lệ tin hợp lệ bị chặn nhầm - chỉ số nghiệp vụ quan trọng nhất của bộ lọc
    ham_mask = np.array(y_true) == "ham"
    metrics["ham_false_block_rate"] = float((np.array(y_pred)[ham_mask] != "ham").mean())
    if langs is not None:
        lang_arr = np.array(langs)
        metrics["per_lang_macro_f1"] = {
            lang: f1_score(np.array(y_true)[lang_arr == lang], np.array(y_pred)[lang_arr == lang],
                           labels=labels, average="macro")
            for lang in sorted(set(langs))
        }
    return metrics


def measure_latency(predictor: Predictor, texts: list[str], seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    sample = list(rng.choice(texts, size=min(_LATENCY_SAMPLES, len(texts)), replace=False))
    for text in sample[:_WARMUP_CALLS]:
        predictor.predict_proba([text])
    timings = []
    for text in sample:
        start = time.perf_counter()
        predictor.predict_proba([text])
        timings.append((time.perf_counter() - start) * 1000)
    batch = sample[:_THROUGHPUT_BATCH]
    start = time.perf_counter()
    predictor.predict_proba(batch)
    batch_seconds = time.perf_counter() - start
    return {
        "p50_latency_ms": float(np.percentile(timings, 50)),
        "p95_latency_ms": float(np.percentile(timings, 95)),
        "mean_latency_ms": float(np.mean(timings)),
        "throughput_msg_per_s": len(batch) / batch_seconds,
    }


def artifact_size_mb(artifact_dir: Path) -> float:
    # Chỉ tính file phục vụ inference (bỏ hf_tokenizer trùng lặp)
    total = sum(f.stat().st_size for f in artifact_dir.rglob("*")
                if f.is_file() and "hf_tokenizer" not in f.parts)
    return total / (1024 * 1024)
