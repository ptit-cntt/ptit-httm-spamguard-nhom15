"""Tải 4 bộ dữ liệu công khai (2 tiếng Anh, 2 tiếng Việt) về data/raw.

| Nguồn            | Ngôn ngữ | Loại văn bản        | Nhãn gốc dùng cho       |
|------------------|----------|---------------------|-------------------------|
| UCI SMS Spam     | en       | SMS                 | ham / spam              |
| Davidson et al.  | en       | Tweet (comment)     | ham / toxic             |
| ViSpamReviews    | vi       | Bình luận TMĐT      | ham / spam              |
| UIT-ViHSD        | vi       | Bình luận MXH       | ham / toxic             |
"""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from pathlib import Path

from spamguard.config import load_params, resolve_path

logger = logging.getLogger(__name__)

SOURCES = {
    "sms_spam_en": {
        "url": "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        "zip_members": {"SMSSpamCollection": "SMSSpamCollection"},
    },
    "davidson_en": {
        "url": "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv",
        "filename": "davidson.csv",
    },
    "vispam_vi": {
        "url": "https://raw.githubusercontent.com/sonlam1102/vispamdetection/main/dataset/vispamdetection_dataset.zip",
        "zip_members": {
            "dataset/train.csv": "vispam_train.csv",
            "dataset/dev.csv": "vispam_dev.csv",
            "dataset/test.csv": "vispam_test.csv",
        },
    },
    "vihsd_vi": {
        "urls": {
            "vihsd_train.csv": "https://huggingface.co/datasets/ura-hcmut/UIT-ViHSD/resolve/main/ViHSD_train.csv",
            "vihsd_dev.csv": "https://huggingface.co/datasets/ura-hcmut/UIT-ViHSD/resolve/main/ViHSD_validation.csv",
            "vihsd_test.csv": "https://huggingface.co/datasets/ura-hcmut/UIT-ViHSD/resolve/main/ViHSD_test.csv",
        }
    },
}

_TIMEOUT_SECONDS = 120


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "spamguard-data-downloader"})
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
        return response.read()


def _save(path: Path, content: bytes) -> None:
    path.write_bytes(content)
    logger.info("Đã lưu %s (%.1f KB)", path, len(content) / 1024)


def download_all(force: bool = False) -> Path:
    raw_dir = resolve_path(load_params()["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    for name, spec in SOURCES.items():
        if "zip_members" in spec:
            targets = list(spec["zip_members"].values())
            if not force and all((raw_dir / t).exists() for t in targets):
                logger.info("Bỏ qua %s (đã có)", name)
                continue
            archive = zipfile.ZipFile(io.BytesIO(_fetch(spec["url"])))
            for member, target in spec["zip_members"].items():
                # Chỉ đọc đúng member cần thiết, không extractall (tránh path traversal)
                _save(raw_dir / target, archive.read(member))
        elif "filename" in spec:
            target = raw_dir / spec["filename"]
            if force or not target.exists():
                _save(target, _fetch(spec["url"]))
        else:
            for filename, url in spec["urls"].items():
                target = raw_dir / filename
                if force or not target.exists():
                    _save(target, _fetch(url))
    return raw_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    download_all()
