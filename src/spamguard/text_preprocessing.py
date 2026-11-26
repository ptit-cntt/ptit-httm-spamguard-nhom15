"""Chuẩn hoá văn bản ngắn song ngữ (Anh/Việt).

Module này được dùng chung cho cả training và serving để tránh training-serving
skew: mọi model nhìn thấy văn bản đã qua cùng một hàm normalize.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

URL_TOKEN = "<url>"
EMAIL_TOKEN = "<email>"
PHONE_TOKEN = "<phone>"
USER_TOKEN = "<user>"
NUMBER_TOKEN = "<num>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

_URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(?:com|net|org|vn|ly|co|info)(?:/\S*)?\b)", re.I)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\s.-]{7,}\d(?!\w)")
# Bỏ dấu "RT" và @mention của tweet: đây là artefact của nguồn dữ liệu Twitter,
# nếu giữ lại model sẽ học "nguồn" thay vì "nội dung độc hại".
_RETWEET_RE = re.compile(r"(^|\s)rt\s+(?=@)", re.I)
_MENTION_RE = re.compile(r"@\w+")
_NUMBER_RE = re.compile(r"\b\d{3,}\b")
_REPEAT_CHAR_RE = re.compile(r"(\w)\1{2,}")
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"<\w+>|\w+|[^\w\s]", re.UNICODE)


def normalize_text(text: str, lowercase: bool = True) -> str:
    """Chuẩn hoá 1 tin nhắn.

    - lowercase=False dùng cho DistilBERT *cased* (giữ chữ hoa vì CAPS LOCK là
      tín hiệu spam/toxic quan trọng).
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = html.unescape(text)
    # NFC để các dạng gõ dấu tiếng Việt khác nhau (tổ hợp / dựng sẵn) về cùng 1 dạng
    text = unicodedata.normalize("NFC", text)
    # Email phải thay trước URL, nếu không phần domain của email bị bắt nhầm thành URL
    text = _EMAIL_RE.sub(f" {EMAIL_TOKEN} ", text)
    text = _URL_RE.sub(f" {URL_TOKEN} ", text)
    text = _PHONE_RE.sub(f" {PHONE_TOKEN} ", text)
    text = _RETWEET_RE.sub(" ", text)
    text = _MENTION_RE.sub(f" {USER_TOKEN} ", text)
    text = _NUMBER_RE.sub(f" {NUMBER_TOKEN} ", text)
    text = _REPEAT_CHAR_RE.sub(r"\1\1", text)
    if lowercase:
        text = text.lower()
    return _WHITESPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Tokenizer cấp từ cho Bi-GRU (văn bản đầu vào đã normalize lowercase)."""
    return _TOKEN_RE.findall(text)


class WordVocabulary:
    """Từ điển word -> id cho Bi-GRU; serialize được ra JSON để serving dùng lại."""

    def __init__(self, token_to_id: dict[str, int], max_len: int):
        self.token_to_id = token_to_id
        self.max_len = max_len
        self.pad_id = token_to_id[PAD_TOKEN]
        self.unk_id = token_to_id[UNK_TOKEN]

    @classmethod
    def build(cls, texts: Iterable[str], vocab_size: int, max_len: int) -> WordVocabulary:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize(text))
        token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for token, _ in counter.most_common(vocab_size - len(token_to_id)):
            token_to_id[token] = len(token_to_id)
        return cls(token_to_id, max_len)

    def encode(self, text: str) -> list[int]:
        ids = [self.token_to_id.get(tok, self.unk_id) for tok in tokenize(text)][: self.max_len]
        # Đảm bảo luôn có ít nhất 1 token để tránh chuỗi rỗng ở GRU
        return ids or [self.unk_id]

    def encode_batch(self, texts: list[str]) -> tuple[list[list[int]], list[int]]:
        """Trả về (ids đã pad về cùng độ dài, độ dài thực của từng chuỗi)."""
        encoded = [self.encode(t) for t in texts]
        lengths = [len(ids) for ids in encoded]
        width = max(lengths)
        padded = [ids + [self.pad_id] * (width - len(ids)) for ids in encoded]
        return padded, lengths

    def to_dict(self) -> dict:
        return {"token_to_id": self.token_to_id, "max_len": self.max_len}

    @classmethod
    def from_dict(cls, payload: dict) -> WordVocabulary:
        return cls(payload["token_to_id"], int(payload["max_len"]))
