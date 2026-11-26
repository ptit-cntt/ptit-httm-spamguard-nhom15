"""Hợp nhất 4 nguồn dữ liệu về schema chung và chia train/val/test.

Schema đầu ra: text, label (ham|spam|toxic), lang (en|vi), source.
Chia phân tầng theo (label, lang) để tập test đại diện đủ cả 2 ngôn ngữ.
"""

from __future__ import annotations

import hashlib
import json
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from spamguard.config import load_params, resolve_path

logger = logging.getLogger(__name__)

# Davidson: 0=hate speech, 1=offensive language, 2=neither
_DAVIDSON_LABELS = {0: "toxic", 1: "toxic", 2: "ham"}
# UIT-ViHSD: 0=CLEAN, 1=OFFENSIVE, 2=HATE
_VIHSD_LABELS = {0: "ham", 1: "toxic", 2: "toxic"}
# ViSpamReviews: Label 0=không spam, 1=spam
_VISPAM_LABELS = {0: "ham", 1: "spam"}


def _load_sources(raw_dir) -> pd.DataFrame:
    frames = []

    sms = pd.read_csv(raw_dir / "SMSSpamCollection", sep="\t", header=None,
                      names=["label", "text"], quoting=3, encoding="utf-8")
    frames.append(sms.assign(lang="en", source="sms_spam_en"))

    davidson = pd.read_csv(raw_dir / "davidson.csv")
    frames.append(pd.DataFrame({
        "text": davidson["tweet"], "label": davidson["class"].map(_DAVIDSON_LABELS),
        "lang": "en", "source": "davidson_en",
    }))

    vispam = pd.concat([pd.read_csv(raw_dir / f"vispam_{s}.csv") for s in ("train", "dev", "test")])
    frames.append(pd.DataFrame({
        "text": vispam["Comment"], "label": vispam["Label"].map(_VISPAM_LABELS),
        "lang": "vi", "source": "vispam_vi",
    }))

    vihsd = pd.concat([pd.read_csv(raw_dir / f"vihsd_{s}.csv") for s in ("train", "dev", "test")])
    frames.append(pd.DataFrame({
        "text": vihsd["free_text"], "label": vihsd["label_id"].map(_VIHSD_LABELS),
        "lang": "vi", "source": "vihsd_vi",
    }))
    return pd.concat(frames, ignore_index=True)


def _clean(df: pd.DataFrame, min_chars: int) -> tuple[pd.DataFrame, dict]:
    stats = {"raw_rows": int(len(df))}
    df = df.dropna(subset=["text", "label"]).copy()
    stats["after_dropna"] = int(len(df))
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= min_chars]
    stats["after_min_chars"] = int(len(df))
    # Loại trùng lặp chính xác; nếu cùng text mà khác nhãn thì bỏ hẳn (nhãn mâu thuẫn)
    conflict = df.groupby("text")["label"].nunique()
    df = df[~df["text"].isin(conflict[conflict > 1].index)]
    stats["conflicting_texts_removed"] = int((conflict > 1).sum())
    df = df.drop_duplicates(subset=["text"])
    stats["after_dedup"] = int(len(df))
    return df, stats


def _sample(df: pd.DataFrame, sampling: dict, seed: int) -> pd.DataFrame:
    parts = []
    for source, quotas in sampling.items():
        for label, limit in quotas.items():
            subset = df[(df["source"] == source) & (df["label"] == label)]
            if limit is not None and len(subset) > limit:
                subset = subset.sample(n=limit, random_state=seed)
            parts.append(subset)
    return pd.concat(parts, ignore_index=True)


def build_dataset() -> dict:
    params = load_params()
    data_cfg = params["data"]
    seed = params["seed"]
    raw_dir = resolve_path(data_cfg["raw_dir"])
    out_dir = resolve_path(data_cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df, stats = _clean(_load_sources(raw_dir), data_cfg["min_chars"])
    stats["per_source_after_clean"] = df.groupby(["source", "label"]).size().to_dict()
    df = _sample(df, data_cfg["sampling"], seed)

    strata = df["label"] + "_" + df["lang"]
    train_val, test = train_test_split(df, test_size=data_cfg["test_size"], stratify=strata,
                                       random_state=seed)
    val_ratio = data_cfg["val_size"] / (1 - data_cfg["test_size"])
    train, val = train_test_split(train_val, test_size=val_ratio,
                                  stratify=train_val["label"] + "_" + train_val["lang"],
                                  random_state=seed)

    splits = {"train": train, "val": val, "test": test}
    summary: dict = {"cleaning": {k: (v if not isinstance(v, dict) else {"|".join(kk): vv for kk, vv in v.items()})
                                  for k, v in stats.items()}}
    for name, split in splits.items():
        split = split.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        split.to_csv(out_dir / f"{name}.csv", index=False)
        summary[name] = {
            "rows": int(len(split)),
            "by_label": split["label"].value_counts().to_dict(),
            "by_lang": split["lang"].value_counts().to_dict(),
            "by_source_label": {f"{s}|{l}": int(c) for (s, l), c in
                                split.groupby(["source", "label"]).size().items()},
        }
    # Hash dữ liệu để truy vết model được train trên phiên bản dữ liệu nào (data lineage)
    digest = hashlib.sha256()
    for name in ("train", "val", "test"):
        digest.update((out_dir / f"{name}.csv").read_bytes())
    summary["data_version"] = digest.hexdigest()[:12]
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("Dataset %s: train=%d val=%d test=%d", summary["data_version"],
                len(train), len(val), len(test))
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_dataset()
