"""SpamGuard API Service: lọc spam / tin nhắn độc hại + API cho Web Admin."""

from __future__ import annotations

import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from spamguard import __version__
from spamguard.config import PROJECT_ROOT, load_params

from . import metrics
from .db import Feedback, MessageLog, SessionLocal, init_db
from .model_manager import ModelManager
from .schemas import (
    BatchClassifyRequest,
    BatchClassifyResponse,
    ClassifyRequest,
    ClassifyResponse,
    FeedbackRequest,
    FeedbackResponse,
    MessageOut,
    PromoteRequest,
    ReviewRequest,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("spamguard.api")

_DEFAULT_ADMIN_TOKEN = "admin-dev-token"
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", _DEFAULT_ADMIN_TOKEN)
ADMIN_WEB_DIR = Path(os.getenv("ADMIN_WEB_DIR", PROJECT_ROOT / "admin-web"))
MAX_PAGE_SIZE = 200
MAX_STATS_DAYS = 90

model_manager = ModelManager()


def _sync_model_gauge() -> None:
    metrics.MODEL_INFO.clear()
    if model_manager.is_ready:
        current = model_manager.current
        metrics.MODEL_INFO.labels(str(current.version), current.model_type).set(1)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if ADMIN_TOKEN == _DEFAULT_ADMIN_TOKEN:
        logger.warning("ADMIN_TOKEN đang dùng giá trị mặc định - KHÔNG dùng cho production")
    init_db()
    model_manager.load_production(force=True)
    _sync_model_gauge()
    yield


app = FastAPI(title="SpamGuard API", version=__version__, lifespan=lifespan,
              description="Smart Anti-Spam & Malicious Content Filtering System")


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    # Dùng route template (vd /v1/admin/messages/{message_id}) tránh bùng nổ cardinality
    path = route.path if route is not None else "unmatched"
    metrics.REQUESTS.labels(request.method, path, str(response.status_code)).inc()
    metrics.REQUEST_LATENCY.labels(path).observe(time.perf_counter() - start)
    return response


def get_db():
    with SessionLocal() as session:
        yield session


def require_admin(x_admin_token: str = Header(default="")) -> None:
    # compare_digest: so sánh thời gian hằng, chống timing attack
    if not secrets.compare_digest(x_admin_token.encode(), ADMIN_TOKEN.encode()):
        raise HTTPException(status_code=401, detail="Admin token không hợp lệ")


def _decide_action(label: str, confidence: float) -> str:
    serving = load_params()["serving"]
    if confidence < serving["review_threshold"]:
        return "review"
    if label != "ham" and confidence >= serving["block_threshold"]:
        return "block"
    return "review" if label != "ham" else "allow"


def _refresh_model() -> None:
    """Kiểm tra registry (có giới hạn tần suất) và hot-reload nếu alias production đổi."""
    if not model_manager.is_ready:
        model_manager.load_production()
        _sync_model_gauge()
        return
    previous_version = model_manager.current.version
    model_manager.maybe_reload()
    if model_manager.current.version != previous_version:
        metrics.MODEL_RELOADS.inc()
        _sync_model_gauge()


def _classify(items: list[ClassifyRequest], db: Session) -> list[ClassifyResponse]:
    _refresh_model()
    if not model_manager.is_ready:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng")

    start = time.perf_counter()
    model, proba = model_manager.predict([item.text for item in items])
    latency_ms = (time.perf_counter() - start) * 1000 / len(items)

    logs = []
    for item, row in zip(items, proba):
        label_idx = int(row.argmax())
        label, confidence = model_manager.labels[label_idx], float(row[label_idx])
        action = _decide_action(label, confidence)
        logs.append(MessageLog(
            text=item.text, channel=item.channel, sender_id=item.sender_id, predicted_label=label,
            confidence=confidence, scores={l: round(float(p), 6) for l, p in zip(model_manager.labels, row)},
            action=action, model_version=model.version, latency_ms=latency_ms,
        ))
        metrics.PREDICTIONS.labels(label, action, item.channel).inc()
        metrics.CONFIDENCE.labels(label).observe(confidence)
        metrics.INFERENCE_LATENCY.observe(latency_ms / 1000)
    db.add_all(logs)
    db.commit()
    return [ClassifyResponse(id=log.id, label=log.predicted_label, confidence=round(log.confidence, 4),
                             action=log.action, scores=log.scores, model_version=log.model_version,
                             latency_ms=round(log.latency_ms, 3)) for log in logs]


# ----------------------------- Public API -----------------------------------

@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "version": __version__}


@app.get("/ready", tags=["ops"])
def ready():
    _refresh_model()
    if not model_manager.is_ready:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng")
    current = model_manager.current
    return {"status": "ready", "model_version": current.version, "model_type": current.model_type}


@app.get("/metrics", tags=["ops"])
def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/messages/classify", response_model=ClassifyResponse, tags=["filter"])
def classify(request: ClassifyRequest, db: Session = Depends(get_db)):
    return _classify([request], db)[0]


@app.post("/v1/messages/classify-batch", response_model=BatchClassifyResponse, tags=["filter"])
def classify_batch(request: BatchClassifyRequest, db: Session = Depends(get_db)):
    return BatchClassifyResponse(results=_classify(request.items, db))


@app.post("/v1/feedback", response_model=FeedbackResponse, status_code=201, tags=["feedback"])
def submit_feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
    """Người dùng cuối báo cáo nhãn sai (vd: 'đây không phải spam')."""
    text = request.text
    if request.message_id is not None:
        message = db.get(MessageLog, request.message_id)
        if message is None:
            raise HTTPException(status_code=404, detail="message_id không tồn tại")
        text = message.text
    if not text:
        raise HTTPException(status_code=422, detail="Cần message_id hoặc text")
    feedback = Feedback(message_id=request.message_id, text=text, label=request.correct_label, source="user")
    db.add(feedback)
    db.commit()
    metrics.FEEDBACK.labels(request.correct_label, "user").inc()
    return FeedbackResponse(id=feedback.id)


# ----------------------------- Admin API ------------------------------------

@app.get("/v1/admin/stats", dependencies=[Depends(require_admin)], tags=["admin"])
def admin_stats(days: int = Query(default=7, ge=1, le=MAX_STATS_DAYS), db: Session = Depends(get_db)):
    _refresh_model()
    since = datetime.now(timezone.utc) - timedelta(days=days)
    in_range = MessageLog.created_at >= since

    by_label = dict(db.execute(select(MessageLog.predicted_label, func.count())
                               .where(in_range).group_by(MessageLog.predicted_label)).all())
    by_action = dict(db.execute(select(MessageLog.action, func.count())
                                .where(in_range).group_by(MessageLog.action)).all())
    by_channel = [{"channel": c, "label": l, "count": n} for c, l, n in db.execute(
        select(MessageLog.channel, MessageLog.predicted_label, func.count())
        .where(in_range).group_by(MessageLog.channel, MessageLog.predicted_label)).all()]
    day = func.date(MessageLog.created_at)
    timeline = [{"date": str(d), "label": l, "count": n} for d, l, n in db.execute(
        select(day, MessageLog.predicted_label, func.count())
        .where(in_range).group_by(day, MessageLog.predicted_label).order_by(day)).all()]
    top_senders = [{"sender_id": s, "flagged": n} for s, n in db.execute(
        select(MessageLog.sender_id, func.count()).where(in_range, MessageLog.sender_id.is_not(None),
                                                         MessageLog.predicted_label != "ham")
        .group_by(MessageLog.sender_id).order_by(func.count().desc()).limit(10)).all()]
    total = sum(by_label.values())
    avg_latency = db.scalar(select(func.avg(MessageLog.latency_ms)).where(in_range)) or 0.0
    pending_review = db.scalar(select(func.count()).select_from(MessageLog)
                               .where(MessageLog.action == "review", MessageLog.reviewed_label.is_(None))) or 0
    # Chỉ feedback do admin duyệt mới được dùng để retrain (xem retrain_from_feedback.py)
    pending_feedback = db.scalar(select(func.count()).select_from(Feedback)
                                 .where(Feedback.used_in_version.is_(None),
                                        Feedback.source == "admin")) or 0
    user_reports = db.scalar(select(func.count()).select_from(Feedback)
                             .where(Feedback.created_at >= since, Feedback.source == "user")) or 0
    current = model_manager.current if model_manager.is_ready else None
    return {
        "days": days, "total": total, "by_label": by_label, "by_action": by_action,
        "spam_rate": by_label.get("spam", 0) / total if total else 0.0,
        "toxic_rate": by_label.get("toxic", 0) / total if total else 0.0,
        "avg_latency_ms": float(avg_latency), "pending_review": pending_review,
        "pending_feedback": pending_feedback, "user_reports": user_reports,
        "timeline": timeline, "by_channel": by_channel, "top_senders": top_senders,
        "model": {"version": current.version, "model_type": current.model_type} if current else None,
        "retrain_min_feedback": load_params()["feedback"]["min_new_samples"],
    }


@app.get("/v1/admin/messages", response_model=list[MessageOut], dependencies=[Depends(require_admin)],
         tags=["admin"])
def admin_messages(label: str | None = None, action: str | None = None, q: str | None = Query(None, max_length=100),
                   only_unreviewed: bool = False, limit: int = Query(default=50, ge=1, le=MAX_PAGE_SIZE),
                   offset: int = Query(default=0, ge=0), db: Session = Depends(get_db)):
    query = select(MessageLog).order_by(MessageLog.id.desc())
    if label:
        query = query.where(MessageLog.predicted_label == label)
    if action:
        query = query.where(MessageLog.action == action)
    if q:
        # SQLAlchemy bind parameter -> an toàn trước SQL injection
        query = query.where(MessageLog.text.ilike(f"%{q}%"))
    if only_unreviewed:
        query = query.where(MessageLog.reviewed_label.is_(None))
    return db.scalars(query.limit(limit).offset(offset)).all()


@app.post("/v1/admin/messages/{message_id}/review", dependencies=[Depends(require_admin)], tags=["admin"])
def admin_review(message_id: int, request: ReviewRequest, db: Session = Depends(get_db)):
    """Admin gán nhãn đúng -> tạo feedback chất lượng cao cho vòng retrain."""
    message = db.get(MessageLog, message_id)
    if message is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy tin nhắn")
    message.reviewed_label = request.label
    db.add(Feedback(message_id=message_id, text=message.text, label=request.label, source="admin"))
    db.commit()
    metrics.FEEDBACK.labels(request.label, "admin").inc()
    return {"id": message_id, "reviewed_label": request.label}


@app.get("/v1/admin/feedback", dependencies=[Depends(require_admin)], tags=["admin"])
def admin_feedback(limit: int = Query(default=50, ge=1, le=MAX_PAGE_SIZE), db: Session = Depends(get_db)):
    rows = db.scalars(select(Feedback).order_by(Feedback.id.desc()).limit(limit)).all()
    return [{"id": f.id, "text": f.text, "label": f.label, "source": f.source, "used_in_version": f.used_in_version,
             "created_at": f.created_at if f.created_at.tzinfo else f.created_at.replace(tzinfo=timezone.utc)}
            for f in rows]


@app.get("/v1/admin/models", dependencies=[Depends(require_admin)], tags=["admin"])
def admin_models():
    index = model_manager.registry.list_versions()
    for version in index["versions"]:
        meta = model_manager.registry.get_meta(version["version"])
        version["metrics"] = {k: meta["metrics"].get(k) for k in
                              ("accuracy", "macro_f1", "ham_false_block_rate", "p95_latency_ms", "model_size_mb")}
        version["data_version"] = meta.get("data_version")
    return index


@app.post("/v1/admin/models/{version}/promote", dependencies=[Depends(require_admin)], tags=["admin"])
def admin_promote(version: int, request: PromoteRequest):
    try:
        model_manager.registry.promote(version, reason=request.reason)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    model_manager.load_production(force=True)
    metrics.MODEL_RELOADS.inc()
    _sync_model_gauge()
    return {"production": version}


@app.post("/v1/admin/models/reload", dependencies=[Depends(require_admin)], tags=["admin"])
def admin_reload():
    changed = model_manager.load_production(force=True)
    _sync_model_gauge()
    return {"reloaded": changed, "version": model_manager.current.version}


# ----------------------------- Web Admin ------------------------------------

if ADMIN_WEB_DIR.exists():
    app.mount("/admin", StaticFiles(directory=ADMIN_WEB_DIR, html=True), name="admin")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/admin/")
