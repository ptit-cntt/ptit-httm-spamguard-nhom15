"""Lưu vết dự đoán và feedback (nguồn dữ liệu cho Web Admin + MLOps Feedback Loop)."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/spamguard.db")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class MessageLog(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    channel: Mapped[str] = mapped_column(String(32), default="sms", index=True)
    sender_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    predicted_label: Mapped[str] = mapped_column(String(16), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    scores: Mapped[dict] = mapped_column(JSON)
    action: Mapped[str] = mapped_column(String(16), index=True)
    model_version: Mapped[int] = mapped_column(Integer, index=True)
    latency_ms: Mapped[float] = mapped_column(Float)
    reviewed_label: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[int | None] = mapped_column(ForeignKey("messages.id"), nullable=True)
    text: Mapped[str] = mapped_column(Text)
    label: Mapped[str] = mapped_column(String(16))
    source: Mapped[str] = mapped_column(String(16))  # user | admin
    # Version model đã dùng feedback này để retrain (null = chưa dùng)
    used_in_version: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)


def _make_engine(url: str):
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    if url.startswith("sqlite:///"):
        db_path = url.replace("sqlite:///", "", 1)
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    return create_engine(url, connect_args=connect_args, pool_pre_ping=True)


engine = _make_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    # Nhiều worker uvicorn khởi động đồng thời có thể cùng tạo bảng (race condition):
    # worker thua sẽ gặp lỗi "table already exists" -> thử lại, lần 2 checkfirst sẽ bỏ qua.
    try:
        Base.metadata.create_all(engine)
    except SQLAlchemyError:
        logger.warning("create_all xung đột với worker khác, thử lại")
        Base.metadata.create_all(engine)
