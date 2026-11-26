"""Runtime predictors dùng khi serving.

Không import torch/transformers: API container chỉ cần onnxruntime + tokenizers +
scikit-learn -> image nhẹ, khởi động nhanh. Mọi predictor có cùng interface
`predict_proba(texts) -> ndarray[n, n_labels]` theo thứ tự `labels`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from spamguard.text_preprocessing import WordVocabulary, normalize_text


class Predictor(Protocol):
    labels: list[str]

    def predict_proba(self, texts: list[str]) -> np.ndarray: ...


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _onnx_session(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Giới hạn thread để nhiều worker uvicorn không tranh chấp CPU
    options.intra_op_num_threads = 2
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


class SklearnPredictor:
    def __init__(self, artifact_dir: Path, labels: list[str]):
        import joblib

        self.labels = labels
        self.pipeline = joblib.load(artifact_dir / "model.joblib")

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba([normalize_text(t) for t in texts])


class BiGRUOnnxPredictor:
    def __init__(self, artifact_dir: Path, labels: list[str]):
        self.labels = labels
        self.vocab = WordVocabulary.from_dict(json.loads((artifact_dir / "vocab.json").read_text()))
        self.session = _onnx_session(artifact_dir / "model.onnx")

    def predict_proba(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        outputs = []
        for start in range(0, len(texts), batch_size):
            chunk = [normalize_text(t) for t in texts[start:start + batch_size]]
            ids, _ = self.vocab.encode_batch(chunk)
            logits = self.session.run(["logits"], {"input_ids": np.asarray(ids, dtype=np.int64)})[0]
            outputs.append(_softmax(logits))
        return np.concatenate(outputs)


class TransformerOnnxPredictor:
    def __init__(self, artifact_dir: Path, labels: list[str], max_len: int = 96):
        from tokenizers import Tokenizer

        self.labels = labels
        self.tokenizer = Tokenizer.from_file(str(artifact_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_len)
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("[PAD]") or 0, pad_token="[PAD]")
        self.session = _onnx_session(artifact_dir / "model.onnx")

    def predict_proba(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        outputs = []
        for start in range(0, len(texts), batch_size):
            chunk = [normalize_text(t, lowercase=False) for t in texts[start:start + batch_size]]
            encodings = self.tokenizer.encode_batch(chunk)
            feeds = {
                "input_ids": np.asarray([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.asarray([e.attention_mask for e in encodings], dtype=np.int64),
            }
            outputs.append(_softmax(self.session.run(["logits"], feeds)[0]))
        return np.concatenate(outputs)


def load_predictor(model_type: str, artifact_dir: Path, labels: list[str], meta: dict | None = None) -> Predictor:
    meta = meta or {}
    if model_type in ("svm", "random_forest"):
        return SklearnPredictor(artifact_dir, labels)
    if model_type == "bigru":
        return BiGRUOnnxPredictor(artifact_dir, labels)
    if model_type == "distilbert":
        return TransformerOnnxPredictor(artifact_dir, labels, meta.get("params", {}).get("max_len", 96))
    raise ValueError(f"model_type không hỗ trợ: {model_type}")
