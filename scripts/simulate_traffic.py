"""Mô phỏng nền tảng nhắn tin gửi lưu lượng thật vào API + admin duyệt hàng đợi review.

Dùng để demo end-to-end Feedback Loop:
  1. Gửi N tin nhắn (lấy từ tập test, có nhãn thật) vào /v1/messages/classify-batch
  2. "Admin" duyệt các tin action=review hoặc bị phân loại sai -> POST review với nhãn thật
     (mô phỏng người kiểm duyệt; trong thực tế admin thao tác trên Web Admin)

Chạy: python scripts/simulate_traffic.py --n 1500 --review 120
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHANNEL_BY_SOURCE = {"sms_spam_en": "sms", "davidson_en": "comment", "vispam_vi": "comment", "vihsd_vi": "chat"}
BATCH_SIZE = 50


def post(base_url: str, path: str, payload: dict, token: str | None = None) -> dict:
    # Cloudflare chặn User-Agent mặc định "Python-urllib" (bot) -> tự khai báo
    headers = {"Content-Type": "application/json", "User-Agent": "spamguard-traffic-simulator/1.0"}
    if token:
        headers["X-Admin-Token"] = token
    request = urllib.request.Request(f"{base_url}{path}", data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--token", default="admin-dev-token")
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--review", type=int, default=120, help="Số tin admin duyệt")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    test = pd.read_csv(ROOT / "data" / "processed" / "test.csv", keep_default_na=False)
    sample = test.sample(n=min(args.n, len(test)), random_state=args.seed).reset_index(drop=True)

    results = []
    for start in range(0, len(sample), BATCH_SIZE):
        chunk = sample.iloc[start:start + BATCH_SIZE]
        items = [{"text": row.text[:2000], "channel": CHANNEL_BY_SOURCE[row.source],
                  "sender_id": f"{row.source.split('_')[0]}-{random.randint(1, 60):03d}"}
                 for row in chunk.itertuples()]
        response = post(args.base_url, "/v1/messages/classify-batch", {"items": items})
        results.extend(zip(chunk["label"], response["results"]))
    wrong = sum(1 for truth, r in results if truth != r["label"])
    print(f"Đã gửi {len(results)} tin; sai {wrong} ({wrong / len(results):.1%})")

    # Ưu tiên duyệt: tin cần review trước, sau đó tin bị phân loại sai (giống quy trình kiểm duyệt)
    candidates = [(t, r) for t, r in results if r["action"] == "review"]
    candidates += [(t, r) for t, r in results if r["action"] != "review" and t != r["label"]]
    reviewed = 0
    for truth, result in candidates[:args.review]:
        post(args.base_url, f"/v1/admin/messages/{result['id']}/review", {"label": truth}, args.token)
        reviewed += 1
    print(f"Admin đã duyệt {reviewed} tin -> feedback cho vòng retrain")


if __name__ == "__main__":
    main()
