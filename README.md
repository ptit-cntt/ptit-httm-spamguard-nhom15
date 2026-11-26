# SpamGuard – Hệ thống lọc Spam & Tin nhắn độc hại

**Smart Anti-Spam & Malicious Content Filtering System** – Đề tài 12, học phần *Phát triển các hệ thống thông minh*
(INT14151), Học viện Công nghệ Bưu chính Viễn thông – **Nhóm 15**.

Phân loại tin nhắn ngắn (SMS, chat, bình luận) **song ngữ Anh – Việt** thành 3 lớp `ham` (hợp lệ) · `spam` (rác,
quảng cáo, lừa đảo) · `toxic` (xúc phạm, thù ghét) và quyết định `allow | review | block`.

| Thành viên | MSSV | Vai trò |
|---|---|---|
| Nguyễn Quốc Cường (trưởng nhóm) | K24DTCN045 | Quản lý dự án, tổng hợp báo cáo; dữ liệu (thu thập, làm sạch, EDA); mô hình SVM & Random Forest |
| Phạm Vương Mạnh | K24DTCN073 | Kiến trúc, MLOps (registry, feedback loop, CI/CD), triển khai |
| Nguyễn Hữu Hoà | K24DTCN056 | Học sâu: Bi-GRU, DistilBERT, export ONNX, đánh giá |
| Nguyễn Vũ Thắng | K24DTCN091 | API Service, Web Admin, Docker, Monitoring, kiểm thử & load test |

---

## 📌 Dành cho người đánh giá

```
├── docs/          📄 TÀI LIỆU BÁO CÁO
│   ├── Project_Action Plan.docx              Kế hoạch dự án
│   ├── Project_Final Report.docx             Báo cáo cuối kỳ (27 trang)
│   ├── Project_Work Breakdown Structure.xlsx Phân chia công việc + Gantt
│   ├── SpamGuard_Slides.pptx                 Slide thuyết trình (16 slide)
│   └── SpamGuard_Slides.pdf                  Slide thuyết trình bản PDF
├── sourceCode/    💻 TOÀN BỘ MÃ NGUỒN ỨNG DỤNG
└── README.md      📘 Tệp này: giới thiệu + hướng dẫn chạy
```

| Muốn xem | Ở đâu |
|---|---|
| **Demo đang chạy** | https://spamguard.pages.dev/admin/ · API docs https://spamguard.pages.dev/docs |
| So sánh 4 mô hình (CLO1) | `sourceCode/reports/model_comparison.md`, `sourceCode/reports/figures/` |
| Mã 4 mô hình | `sourceCode/src/spamguard/models/` (classical.py = SVM + RF, bigru.py, distilbert.py) |
| API Service lọc tin nhắn | `sourceCode/services/api/app/main.py` |
| Web Admin thống kê spam | `sourceCode/admin-web/` |
| MLOps Feedback Loop | `sourceCode/scripts/retrain_from_feedback.py`, `sourceCode/src/spamguard/registry.py` |
| Docker, CI/CD, giám sát, triển khai | `sourceCode/docker-compose.yml`, `.github/workflows/`, `sourceCode/monitoring/`, `sourceCode/deploy/` |
| Kiểm thử | `sourceCode/tests/`, `sourceCode/loadtest/` |

**Kết quả chính** (tập test 5.080 tin, CPU):

| Mô hình | Macro-F1 | F1 EN | F1 VI | ROC-AUC | p95 (ms) | Size (MB) |
|---|---|---|---|---|---|---|
| SVM (TF-IDF) | 0.837 | 0.942 | 0.787 | 0.954 | 1.2 | 9.5 |
| Random Forest | 0.800 | 0.910 | 0.739 | 0.945 | 28.5 | 88.2 |
| Bi-GRU (ONNX) | 0.811 | 0.919 | 0.753 | 0.947 | 0.26 | 13.6 |
| **DistilBERT (ONNX INT8)** ★ | **0.856** | **0.951** | **0.805** | **0.964** | 7.9 | 132.3 |

Load test (Locust, 50 user): ~334 req/s, 0 lỗi, p95 `/classify` 19 ms · Feedback Loop: quality gate đã chặn đúng một
mô hình ứng viên kém tổng quát hơn (`sourceCode/reports/feedback_loop/`).

---

## 🚀 Demo trực tuyến

| Thành phần | Link |
|---|---|
| Web Admin (cần Admin Token) | https://spamguard.pages.dev/admin/ |
| Tài liệu API (Swagger) | https://spamguard.pages.dev/docs |
| Health check | https://spamguard.pages.dev/ready |

```bash
curl -X POST https://spamguard.pages.dev/v1/messages/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Chúc mừng bạn trúng thưởng iPhone 17, bấm link bit.ly/abc để nhận quà", "channel": "sms"}'
# → {"label":"spam","confidence":0.88,"action":"block",...}
```

Hạ tầng: Cloudflare Pages Function (HTTPS, reverse proxy) → Nginx trên server → Docker trên PC LAN
(api, postgres, mlflow, prometheus, grafana). Chi tiết: `sourceCode/deploy/`.

---

## 🛠 Hướng dẫn chạy

Yêu cầu: Python 3.11+, `make`, Internet (tải thư viện, dữ liệu, model nền DistilBERT). Mọi lệnh chạy trong `sourceCode/`.

**Chạy demo nhanh nhất (~5 phút, không cần GPU):**
```bash
cd sourceCode
make setup          # tạo .venv, cài thư viện (~2 GB, tải tự động)
make train-fast     # chỉ train SVM (~10 giây) từ data/processed có sẵn
make serve          # API: http://localhost:8000/docs · Admin: http://localhost:8000/admin
```
Admin token mặc định khi chạy local: `admin-dev-token`. Cổng 8000 bị chiếm thì dùng `make serve PORT=8010`.

> Gói nộp **không kèm model đã huấn luyện** (giảm dung lượng). `make train-fast` cho bản demo bằng SVM;
> muốn chạy y hệt bản production (DistilBERT ONNX INT8) thì chạy `make train` (~10 phút có GPU, lâu hơn trên CPU).

**Tái lập toàn bộ thí nghiệm:**
```bash
cd sourceCode
make setup          # cài thư viện
make data           # tải UCI SMS, Davidson, ViSpamReviews, UIT-ViHSD → data/processed
make train          # train + so sánh 4 mô hình (~10 phút có GPU), promote champion
make figures        # vẽ lại biểu đồ vào reports/figures
make serve          # chạy API + Web Admin
make simulate       # bơm 1.500 tin mẫu + mô phỏng admin review
make retrain        # 1 vòng MLOps feedback loop (retrain → quality gate → promote → hot-reload)
make test lint      # 24 test + kiểm tra mã
```

**Chạy bằng Docker:**
```bash
cd sourceCode
cp .env.example .env            # đổi mật khẩu / token
make train-fast                 # API cần có model trong models/registry
docker compose up -d --build    # api:8000, mlflow:5000, prometheus:9090, grafana:3000
```

**Đóng gói bài nộp:** `cd sourceCode && make package` → `sourceCode/dist/HTTM_Nhom15_SpamGuard.zip`
(chỉ gồm `docs/`, mã nguồn và `README.md`; bỏ `.venv`, model đã train, dữ liệu gốc, database, bí mật triển khai).

---

## 📂 Cấu trúc `sourceCode/`

```
sourceCode/
├── src/spamguard/          🧠 thư viện ML dùng chung train & serve
│   ├── data/               #   tải + làm sạch + chia dữ liệu
│   ├── models/             #   SVM, Random Forest, Bi-GRU, DistilBERT
│   ├── text_preprocessing.py · inference.py · evaluation.py · registry.py · training.py
├── services/api/app/       🌐 API FastAPI: lọc tin, feedback, admin API, /metrics
├── admin-web/              🖥  Web Admin (HTML/JS + Chart.js), phục vụ tại /admin
├── scripts/                ⚙️  train_all, retrain_from_feedback, simulate_traffic, make_figures, package_submission
├── tests/ · loadtest/      ✅ pytest (unit + API) · Locust
├── configs/params.yaml     🔧 siêu tham số + ngưỡng nghiệp vụ
├── data/processed/         📊 dữ liệu đã làm sạch (train/val/test)
├── reports/                📈 kết quả thí nghiệm: metrics, bảng so sánh, biểu đồ, load test, feedback loop, logs
├── docker/ · docker-compose.yml   🐳 image API/trainer, stack local
├── deploy/                 ☁️  triển khai thật (Cloudflare Pages + Nginx + Docker LAN)
├── monitoring/             📟 Prometheus, alert rules, Grafana dashboard
└── Makefile · pyproject.toml · requirements-{api,train,dev}.txt
```
Sinh ra khi chạy (không nằm trong repo / gói nộp): `.venv/`, `data/raw/`, `models/`, `deploy/.build/`, `*.db`, `mlruns/`.

## 📚 Dữ liệu
UCI SMS Spam Collection (EN) · Davidson et al., *Hate Speech and Offensive Language* (EN) · ViSpamReviews (VI) ·
UIT-ViHSD (VI). Chỉ dùng cho mục đích học tập/nghiên cứu theo giấy phép của tác giả.
