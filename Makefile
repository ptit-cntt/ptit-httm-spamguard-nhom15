PY ?= .venv/bin/python
export PYTHONPATH := src:services/api
export MLFLOW_DISABLE_AGENT_HINT := 1
export TOKENIZERS_PARALLELISM := false

.PHONY: setup data train figures test lint serve simulate retrain loadtest mlflow-ui docker-up docker-down

setup:            ## Tạo virtualenv và cài dependency
	python3 -m venv .venv && $(PY) -m pip install -r requirements-dev.txt

data:             ## Tải 4 bộ dữ liệu + làm sạch + chia train/val/test
	$(PY) -m spamguard.data.download && $(PY) -m spamguard.data.build_dataset

train:            ## Huấn luyện + so sánh 4 model, đăng ký registry, promote champion
	$(PY) scripts/train_all.py

figures:          ## Sinh biểu đồ EDA / so sánh model / kiến trúc
	$(PY) scripts/make_figures.py

test:
	$(PY) -m pytest -q tests

lint:
	$(PY) -m ruff check src services scripts tests

serve:            ## Chạy API + Web Admin tại http://localhost:8000/admin
	$(PY) -m uvicorn app.main:app --host 0.0.0.0 --port 8000

simulate:         ## Bơm lưu lượng mẫu + mô phỏng admin review
	$(PY) scripts/simulate_traffic.py --n 1500 --review 150

retrain:          ## Chạy 1 vòng MLOps feedback loop
	$(PY) scripts/retrain_from_feedback.py

loadtest:
	$(PY) -m locust -f loadtest/locustfile.py --host http://localhost:8000 --headless -u 50 -r 10 -t 1m --csv reports/locust

mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down
