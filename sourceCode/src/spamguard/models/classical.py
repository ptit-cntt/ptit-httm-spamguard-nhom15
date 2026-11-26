"""Hai baseline học máy cổ điển: SVM (LinearSVC) và Random Forest trên TF-IDF."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from spamguard.text_preprocessing import normalize_text

ARTIFACT_FILENAME = "model.joblib"


def _build_features(word_ngram_max: int, char_ngram_max: int,
                    max_word_features: int, max_char_features: int) -> FeatureUnion:
    # char n-gram bắt được biến thể cố tình viết sai (fr33, v.i.p, đ.m) - rất phổ biến
    # trong spam/toxic; word n-gram bắt cụm từ ngữ nghĩa ("call now", "trúng thưởng").
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, word_ngram_max),
                                 max_features=max_word_features, sublinear_tf=True,
                                 token_pattern=r"(?u)<\w+>|\b\w+\b", min_df=2)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, char_ngram_max),
                                 max_features=max_char_features, sublinear_tf=True, min_df=2)),
    ])


def build_svm(cfg: dict, seed: int) -> Pipeline:
    svm = LinearSVC(C=cfg["C"], class_weight="balanced", random_state=seed)
    # LinearSVC không có predict_proba -> calibrate (sigmoid) để có xác suất cho ngưỡng block/review
    calibrated = CalibratedClassifierCV(svm, method="sigmoid", cv=3)
    features = _build_features(cfg["word_ngram_max"], cfg["char_ngram_max"],
                               cfg["max_word_features"], cfg["max_char_features"])
    return Pipeline([("features", features), ("clf", calibrated)])


def build_random_forest(cfg: dict, seed: int) -> Pipeline:
    # RF không hiệu quả với không gian đặc trưng quá lớn -> giới hạn tổng số đặc trưng
    half = cfg["max_features"] // 2
    features = _build_features(2, 4, half, half)
    forest = RandomForestClassifier(
        n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
        min_samples_leaf=cfg["min_samples_leaf"], class_weight="balanced_subsample",
        n_jobs=-1, random_state=seed,
    )
    return Pipeline([("features", features), ("clf", forest)])


def train_classical(model_type: str, cfg: dict, seed: int, train_texts: list[str],
                    train_labels: list[str], labels: list[str], output_dir: Path,
                    sample_weight: np.ndarray | None = None) -> Path:
    builder = {"svm": build_svm, "random_forest": build_random_forest}[model_type]
    pipeline = builder(cfg, seed)
    normalized = [normalize_text(t) for t in train_texts]
    fit_params = {"clf__sample_weight": sample_weight} if sample_weight is not None else {}
    pipeline.fit(normalized, train_labels, **fit_params)
    # Đảm bảo thứ tự cột predict_proba khớp thứ tự labels chung của hệ thống
    if list(pipeline.classes_) != labels:
        raise ValueError(f"Thứ tự lớp {pipeline.classes_} khác cấu hình {labels}")
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / ARTIFACT_FILENAME, compress=3)
    return output_dir
