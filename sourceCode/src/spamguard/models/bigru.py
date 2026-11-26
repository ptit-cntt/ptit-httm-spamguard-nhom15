"""Bi-GRU (học sâu) cho phân loại văn bản ngắn, export ONNX để serving không cần PyTorch."""

from __future__ import annotations

import copy
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from spamguard.text_preprocessing import WordVocabulary, normalize_text

logger = logging.getLogger(__name__)

ONNX_FILENAME = "model.onnx"
VOCAB_FILENAME = "vocab.json"


class BiGRUClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_classes: int, dropout: float, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        # Ghép mean-pool và max-pool của 2 chiều -> 4 * hidden_dim
        self.classifier = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = (input_ids != self.pad_id).unsqueeze(-1).float()
        outputs, _ = self.gru(self.dropout(self.embedding(input_ids)))
        # Masked pooling: không cho vị trí padding ảnh hưởng tới biểu diễn câu
        mean_pool = (outputs * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        max_pool = outputs.masked_fill(mask == 0, -1e4).max(1).values
        return self.classifier(self.dropout(torch.cat([mean_pool, max_pool], dim=1)))


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batches(ids: list[list[int]], labels: np.ndarray | None, batch_size: int, shuffle: bool,
             pad_id: int, weights: np.ndarray | None = None):
    order = list(range(len(ids)))
    if shuffle:
        random.shuffle(order)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        width = max(len(ids[i]) for i in idx)
        batch = torch.full((len(idx), width), pad_id, dtype=torch.long)
        for row, i in enumerate(idx):
            batch[row, :len(ids[i])] = torch.tensor(ids[i])
        y = torch.tensor(labels[idx]) if labels is not None else None
        w = torch.tensor(weights[idx], dtype=torch.float) if weights is not None else None
        yield batch, y, w


@torch.no_grad()
def _predict_logits(model: nn.Module, ids: list[list[int]], batch_size: int, pad_id: int,
                    device: torch.device) -> np.ndarray:
    model.eval()
    chunks = [model(x.to(device)).cpu().numpy() for x, _, _ in _batches(ids, None, batch_size, False, pad_id)]
    return np.concatenate(chunks)


def train_bigru(cfg: dict, seed: int, train_texts: list[str], train_labels: list[str],
                val_texts: list[str], val_labels: list[str], labels: list[str],
                output_dir: Path, sample_weight: np.ndarray | None = None) -> tuple[Path, dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _device()
    label_to_id = {label: i for i, label in enumerate(labels)}

    train_norm = [normalize_text(t) for t in train_texts]
    val_norm = [normalize_text(t) for t in val_texts]
    vocab = WordVocabulary.build(train_norm, cfg["vocab_size"], cfg["max_len"])
    train_ids = [vocab.encode(t) for t in train_norm]
    val_ids = [vocab.encode(t) for t in val_norm]
    y_train = np.array([label_to_id[label] for label in train_labels])
    y_val = np.array([label_to_id[label] for label in val_labels])

    model = BiGRUClassifier(len(vocab.token_to_id), cfg["embed_dim"], cfg["hidden_dim"],
                            len(labels), cfg["dropout"], vocab.pad_id).to(device)
    # Trọng số lớp nghịch đảo tần suất để bù mất cân bằng (spam ít hơn ham)
    counts = np.bincount(y_train, minlength=len(labels))
    class_weight = torch.tensor(len(y_train) / (len(labels) * counts), dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weight, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    best_f1, best_state, history, bad_epochs = -1.0, None, [], 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for x, y, w in _batches(train_ids, y_train, cfg["batch_size"], True, vocab.pad_id, sample_weight):
            optimizer.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            loss = (loss * w.to(device)).mean() if w is not None else loss.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(x)
        val_pred = _predict_logits(model, val_ids, 256, vocab.pad_id, device).argmax(1)
        val_f1 = f1_score(y_val, val_pred, average="macro")
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_ids), "val_macro_f1": val_f1})
        logger.info("[Bi-GRU] epoch %d loss=%.4f val_macro_f1=%.4f", epoch,
                    total_loss / len(train_ids), val_f1)
        if val_f1 > best_f1:
            best_f1, best_state, bad_epochs = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                logger.info("[Bi-GRU] early stopping tại epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / VOCAB_FILENAME).write_text(json.dumps(vocab.to_dict(), ensure_ascii=False))
    export_onnx(model.cpu().eval(), output_dir / ONNX_FILENAME)
    return output_dir, {"history": history, "best_val_macro_f1": best_f1}


def export_onnx(model: nn.Module, path: Path) -> None:
    dummy = torch.randint(2, 100, (2, 16), dtype=torch.long)
    torch.onnx.export(
        model, (dummy,), str(path), input_names=["input_ids"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
