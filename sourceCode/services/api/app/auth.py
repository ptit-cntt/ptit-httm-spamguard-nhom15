"""Xác thực quản trị: đăng nhập tài khoản/mật khẩu -> session token có hạn.

Tạm thời tài khoản cố định trong biến môi trường (ADMIN_USERNAME / ADMIN_PASSWORD).
Session token tự chứa (stateless) dạng `<payload_b64>.<chữ_ký_b64>`, ký HMAC-SHA256 bằng
ADMIN_TOKEN -> API không cần lưu phiên, restart container không làm mất phiên đang dùng.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time

DEFAULT_ADMIN_TOKEN = "admin-dev-token"
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "1234"
SESSION_TTL_SECONDS = 8 * 3600

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", DEFAULT_ADMIN_TOKEN)
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", DEFAULT_ADMIN_USERNAME)
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", DEFAULT_ADMIN_PASSWORD)


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _b64decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(payload: str) -> str:
    return _b64encode(hmac.new(ADMIN_TOKEN.encode(), payload.encode(), hashlib.sha256).digest())


def _safe_equals(left: str, right: str) -> bool:
    # compare_digest: so sánh thời gian hằng, chống timing attack
    return secrets.compare_digest(left.encode(), right.encode())


def verify_credentials(username: str, password: str) -> bool:
    # Luôn so sánh cả hai trường để thời gian phản hồi không lộ việc username đúng hay sai
    is_username_valid = _safe_equals(username, ADMIN_USERNAME)
    is_password_valid = _safe_equals(password, ADMIN_PASSWORD)
    return is_username_valid and is_password_valid


def create_session_token(username: str, now: float | None = None) -> str:
    issued_at = int(now if now is not None else time.time())
    payload = _b64encode(json.dumps({"sub": username, "exp": issued_at + SESSION_TTL_SECONDS}).encode())
    return f"{payload}.{_sign(payload)}"


def is_session_token_valid(token: str, now: float | None = None) -> bool:
    payload, separator, signature = token.partition(".")
    if not separator or not _safe_equals(signature, _sign(payload)):
        return False
    try:
        claims = json.loads(_b64decode(payload))
    except ValueError:
        return False
    current_time = now if now is not None else time.time()
    return claims.get("sub") == ADMIN_USERNAME and current_time < claims.get("exp", 0)


def is_admin_token_valid(token: str) -> bool:
    """Chấp nhận session token (Web Admin) hoặc ADMIN_TOKEN tĩnh (script, CI, curl)."""
    if not token:
        return False
    return _safe_equals(token, ADMIN_TOKEN) or is_session_token_valid(token)
