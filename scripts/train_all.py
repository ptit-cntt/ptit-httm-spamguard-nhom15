"""Thí nghiệm CLO1: huấn luyện & so sánh SVM, Random Forest, Bi-GRU, DistilBERT.

Chạy:  python scripts/train_all.py [--models svm bigru] [--no-promote]
Đầu ra: models/registry/v*, reports/model_comparison.{csv,md}, reports/metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spamguard.config import PROJECT_ROOT, load_params, resolve_path  # noqa: E402
from spamguard.registry import ModelRegistry  # noqa: E402
from spamguard.training import (  # noqa: E402
    MODEL_DISPLAY_NAMES,
    MODEL_TYPES,
    load_splits,
    save_json,
    select_champion,
    train_evaluate_register,
)

logger = logging.getLogger("train_all")
REPORT_DIR = PROJECT_ROOT / "reports"


def write_comparison(results: dict[str, dict], versions: dict[str, int], champion: str) -> pd.DataFrame:
    rows = []
    for model_type, m in results.items():
        rows.append({
            "Model": MODEL_DISPLAY_NAMES[model_type], "Version": f"v{versions[model_type]}",
            "Accuracy": m["accuracy"], "Macro-P": m["macro_precision"], "Macro-R": m["macro_recall"],
            "Macro-F1": m["macro_f1"], "F1-ham": m["per_class"]["ham"]["f1"],
            "F1-spam": m["per_class"]["spam"]["f1"], "F1-toxic": m["per_class"]["toxic"]["f1"],
            "F1-EN": m["per_lang_macro_f1"].get("en"), "F1-VI": m["per_lang_macro_f1"].get("vi"),
            "ROC-AUC": m["roc_auc_ovr"], "Ham false-block": m["ham_false_block_rate"],
            "p50 (ms)": m["p50_latency_ms"], "p95 (ms)": m["p95_latency_ms"],
            "Throughput (msg/s)": m["throughput_msg_per_s"], "Size (MB)": m["model_size_mb"],
            "Train (s)": m["train_seconds"], "Champion": "★" if model_type == champion else "",
        })
    table = pd.DataFrame(rows)
    table.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    (REPORT_DIR / "model_comparison.md").write_text(table.round(4).to_markdown(index=False))
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODEL_TYPES, choices=MODEL_TYPES)
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    params = load_params()
    summary = json.loads((resolve_path(params["data"]["processed_dir"]) / "dataset_summary.json").read_text())
    splits = load_splits()

    metrics_path = REPORT_DIR / "metrics.json"
    previous = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    results = previous.get("results", {})
    versions = previous.get("versions", {})
    for model_type in args.models:
        versions[model_type], results[model_type] = train_evaluate_register(
            model_type, splits, summary["data_version"])

    champion = select_champion(results)
    save_json(metrics_path, {"data_version": summary["data_version"], "results": results,
                             "versions": versions, "champion": champion})
    table = write_comparison(results, versions, champion)
    print(table.round(4).to_string(index=False))

    if not args.no_promote:
        ModelRegistry().promote(versions[champion], reason=(
            f"train_all: {champion} có {params['selection']['metric']} cao nhất "
            f"trong các model thoả p95 <= {params['selection']['max_p95_latency_ms']}ms"))
        logger.info("Champion: %s (v%d) -> production", champion, versions[champion])


if __name__ == "__main__":
    main()
