"""MLOps Feedback Loop: feedback đã duyệt -> retrain -> đánh giá -> gate -> promote.

Chạy định kỳ (cron / GitHub Actions schedule / docker compose run trainer):
    python scripts/retrain_from_feedback.py [--force] [--model svm]

Chỉ dùng feedback do admin xác nhận (source=admin). Feedback từ người dùng cuối
có thể bị lạm dụng (spammer tự báo "không phải spam") nên phải qua hàng đợi review.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spamguard.config import PROJECT_ROOT, get_labels, load_params, resolve_path  # noqa: E402
from spamguard.evaluation import classification_metrics  # noqa: E402
from spamguard.inference import load_predictor  # noqa: E402
from spamguard.registry import ModelRegistry  # noqa: E402
from spamguard.training import load_splits, train_evaluate_register  # noqa: E402

logger = logging.getLogger("feedback_loop")
_MIN_HOLDOUT = 10
_HOLDOUT_RATIO = 0.2


def fetch_feedback(database_url: str) -> pd.DataFrame:
    engine = create_engine(database_url)
    query = text("SELECT id, text, label, created_at FROM feedback "
                 "WHERE source = 'admin' AND used_in_version IS NULL ORDER BY id")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    # Cùng 1 nội dung được review nhiều lần -> lấy nhãn mới nhất
    return df.drop_duplicates(subset=["text"], keep="last").reset_index(drop=True)


def mark_used(database_url: str, feedback_ids: list[int], version: int) -> None:
    engine = create_engine(database_url)
    with engine.begin() as conn:
        conn.execute(text("UPDATE feedback SET used_in_version = :v WHERE id = :id"),
                     [{"v": version, "id": i} for i in feedback_ids])


def holdout_accuracy(model_type: str, artifact_dir: Path, holdout: pd.DataFrame, meta: dict) -> dict:
    if holdout.empty:
        return {}
    labels = get_labels()
    proba = load_predictor(model_type, artifact_dir, labels, meta).predict_proba(holdout["text"].tolist())
    m = classification_metrics(holdout["label"].tolist(), proba, labels)
    return {"feedback_holdout_accuracy": m["accuracy"], "feedback_holdout_macro_f1": m["macro_f1"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--model", default=None, help="Loại model retrain (mặc định = model production)")
    parser.add_argument("--force", action="store_true", help="Bỏ qua ngưỡng số feedback tối thiểu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    params = load_params()
    fb_cfg = params["feedback"]
    database_url = args.database_url or os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'data' / 'spamguard.db'}")
    registry = ModelRegistry()
    prod_version = registry.get_alias("production")
    if prod_version is None:
        logger.error("Chưa có model production -> chạy scripts/train_all.py trước")
        return 1
    prod_meta = registry.get_meta(prod_version)

    feedback = fetch_feedback(database_url)
    logger.info("Có %d feedback (admin) chưa dùng; ngưỡng = %d", len(feedback), fb_cfg["min_new_samples"])
    if len(feedback) < fb_cfg["min_new_samples"] and not args.force:
        logger.info("Chưa đủ feedback -> bỏ qua lần retrain này")
        return 0
    if feedback.empty:
        logger.info("Không có feedback nào")
        return 0

    rng = np.random.default_rng(params["seed"])
    shuffled = feedback.iloc[rng.permutation(len(feedback))]
    n_holdout = int(len(shuffled) * _HOLDOUT_RATIO) if len(shuffled) * _HOLDOUT_RATIO >= _MIN_HOLDOUT else 0
    holdout, fb_train = shuffled.iloc[:n_holdout], shuffled.iloc[n_holdout:]

    splits = load_splits()
    base_train = splits["train"]
    # Loại khỏi train gốc các câu trùng feedback (nhãn feedback được ưu tiên)
    base_train = base_train[~base_train["text"].isin(feedback["text"])]
    # Chống rò rỉ dữ liệu: tin feedback có thể trùng tập test cố định -> bỏ khỏi tập đánh giá,
    # và đánh giá lại model production trên CÙNG tập test đã lọc để so sánh công bằng.
    test_eval = splits["test"][~splits["test"]["text"].isin(feedback["text"])].reset_index(drop=True)
    n_test_removed = len(splits["test"]) - len(test_eval)
    splits = {**splits, "test": test_eval}
    fb_rows = fb_train.assign(lang="unk", source="feedback")[["text", "label", "lang", "source"]]
    augmented = pd.concat([base_train, *[fb_rows] * fb_cfg["upsample_factor"]], ignore_index=True)

    model_type = args.model or fb_cfg.get("retrain_model") or prod_meta["model_type"]
    summary = json.loads((resolve_path(params["data"]["processed_dir"]) / "dataset_summary.json").read_text())
    data_version = f"{summary['data_version']}+fb{len(feedback)}"

    new_version, new_metrics = train_evaluate_register(
        model_type, splits, data_version, source="feedback_loop", train_df=augmented,
        extra_eval=lambda mt, art: holdout_accuracy(mt, art, holdout, {"params": params["models"][mt]}),
    )
    prod_holdout = holdout_accuracy(prod_meta["model_type"], registry.artifact_dir(prod_version), holdout, prod_meta)

    # Quality gate: không được tụt trên test cố định VÀ phải không kém hơn trên feedback holdout
    prod_proba = load_predictor(prod_meta["model_type"], registry.artifact_dir(prod_version), get_labels(),
                                prod_meta).predict_proba(test_eval["text"].tolist())
    prod_f1 = classification_metrics(test_eval["label"].tolist(), prod_proba, get_labels())["macro_f1"]
    f1_ok = new_metrics["macro_f1"] >= prod_f1 - fb_cfg["max_macro_f1_drop"]
    holdout_ok = (not prod_holdout) or (new_metrics.get("feedback_holdout_accuracy", 0)
                                        >= prod_holdout["feedback_holdout_accuracy"])
    promoted = f1_ok and holdout_ok
    if promoted:
        registry.promote(new_version, reason=(
            f"feedback_loop: macro_f1 {prod_f1:.4f}->{new_metrics['macro_f1']:.4f}, "
            f"{len(feedback)} feedback"))
        mark_used(database_url, feedback["id"].tolist(), new_version)
        logger.info("PROMOTED v%d -> production (API sẽ tự hot-reload)", new_version)
    else:
        logger.warning("Gate FAILED: giữ nguyên production v%d", prod_version)

    report = {
        "at": datetime.now(timezone.utc).isoformat(), "model_type": model_type,
        "production_before": prod_version, "candidate": new_version, "promoted": promoted,
        "feedback_used": len(feedback), "feedback_holdout": int(n_holdout),
        "test_rows_eval": len(test_eval), "test_rows_removed_overlap": n_test_removed,
        "gate": {"prod_macro_f1": prod_f1, "candidate_macro_f1": new_metrics["macro_f1"],
                 "max_drop": fb_cfg["max_macro_f1_drop"], "f1_ok": f1_ok, "holdout_ok": holdout_ok,
                 "prod_holdout": prod_holdout,
                 "candidate_holdout": {k: v for k, v in new_metrics.items() if k.startswith("feedback_holdout")}},
    }
    out = PROJECT_ROOT / "reports" / "feedback_loop"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"run_{datetime.now():%Y%m%d_%H%M%S}.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
