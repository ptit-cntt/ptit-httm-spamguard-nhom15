"""Scheduler đơn giản cho container retrainer: chạy feedback loop theo chu kỳ.

Một lần chạy lỗi không làm chết scheduler (log lại và chờ chu kỳ sau).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_INTERVAL_SECONDS = 86400
SCRIPT = Path(__file__).with_name("retrain_from_feedback.py")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s retrain_scheduler %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    interval = int(os.getenv("RETRAIN_INTERVAL_SECONDS", DEFAULT_INTERVAL_SECONDS))
    while True:
        result = subprocess.run([sys.executable, str(SCRIPT)], check=False)
        if result.returncode != 0:
            logger.error("Feedback loop kết thúc với mã lỗi %d", result.returncode)
        logger.info("Chờ %d giây tới lần chạy tiếp theo", interval)
        time.sleep(interval)


if __name__ == "__main__":
    main()
