"""Load test API lọc tin nhắn.

Chạy: locust -f loadtest/locustfile.py --host http://localhost:8000 \
             --headless -u 50 -r 10 -t 1m --csv reports/locust
"""

import random

from locust import HttpUser, between, task

SAMPLES = [
    "Hey, are we still meeting at 6pm today?",
    "URGENT! You have won a 1 week FREE membership in our £100,000 prize Jackpot! Txt CLAIM to 81010",
    "Mai mình đi cà phê không bạn ơi?",
    "Chúc mừng bạn trúng thưởng iPhone 17, bấm vào link bit.ly/abc để nhận quà",
    "you are such a stupid idiot, nobody likes you",
    "Shop giao hàng nhanh, đóng gói cẩn thận, sẽ ủng hộ tiếp",
    "Đồ ngu, câm mồm lại đi",
]
CHANNELS = ["sms", "chat", "comment"]


class MessagingPlatformUser(HttpUser):
    wait_time = between(0.05, 0.2)

    @task(10)
    def classify_single(self):
        self.client.post("/v1/messages/classify", name="/v1/messages/classify", json={
            "text": random.choice(SAMPLES), "channel": random.choice(CHANNELS),
            "sender_id": f"user-{random.randint(1, 500)}",
        })

    @task(1)
    def classify_batch(self):
        items = [{"text": random.choice(SAMPLES), "channel": "comment"} for _ in range(20)]
        self.client.post("/v1/messages/classify-batch", name="/v1/messages/classify-batch", json={"items": items})
