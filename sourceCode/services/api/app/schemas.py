from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_TEXT_LENGTH = 2000
MAX_BATCH_SIZE = 100

Label = Literal["ham", "spam", "toxic"]
Action = Literal["allow", "review", "block"]
Channel = Literal["sms", "chat", "comment", "email", "other"]


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    channel: Channel = "sms"
    sender_id: str | None = Field(default=None, max_length=128)

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text không được rỗng")
        return value


class ClassifyResponse(BaseModel):
    id: int
    label: Label
    confidence: float
    action: Action
    scores: dict[str, float]
    model_version: int
    latency_ms: float


class BatchClassifyRequest(BaseModel):
    items: list[ClassifyRequest] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class BatchClassifyResponse(BaseModel):
    results: list[ClassifyResponse]


class FeedbackRequest(BaseModel):
    message_id: int | None = None
    text: str | None = Field(default=None, max_length=MAX_TEXT_LENGTH)
    correct_label: Label

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        return value.strip() if value else value


class FeedbackResponse(BaseModel):
    id: int
    status: str = "accepted"


class ReviewRequest(BaseModel):
    label: Label


class PromoteRequest(BaseModel):
    reason: str = Field(default="manual promote from admin", max_length=300)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    username: str


class MessageOut(BaseModel):
    id: int
    text: str
    channel: str
    predicted_label: str
    confidence: float
    action: str
    model_version: int
    reviewed_label: str | None
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def assume_utc(cls, value: datetime) -> datetime:
        # SQLite không lưu timezone -> giá trị đọc ra là naive nhưng thực chất là UTC
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
