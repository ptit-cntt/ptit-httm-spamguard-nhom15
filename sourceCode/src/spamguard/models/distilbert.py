"""Fine-tune DistilBERT đa ngôn ngữ, export ONNX + lượng tử hoá INT8 cho serving CPU."""

from __future__ import annotations

import copy
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from spamguard.text_preprocessing import normalize_text

logger = logging.getLogger(__name__)

ONNX_FP32_FILENAME = "model_fp32.onnx"
ONNX_FILENAME = "model.onnx"  # bản INT8 dùng cho serving
TOKENIZER_FILENAME = "tokenizer.json"


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _encode(tokenizer, texts: list[str], max_len: int) -> dict[str, torch.Tensor]:
    return tokenizer(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")


@torch.no_grad()
def _predict(model, tokenizer, texts: list[str], max_len: int, device, batch_size: int = 128) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(texts), batch_size):
        batch = {k: v.to(device) for k, v in _encode(tokenizer, texts[start:start + batch_size], max_len).items()}
        outputs.append(model(**batch).logits.float().cpu().numpy())
    return np.concatenate(outputs)


def train_distilbert(cfg: dict, seed: int, train_texts: list[str], train_labels: list[str],
                     val_texts: list[str], val_labels: list[str], labels: list[str],
                     output_dir: Path, sample_weight: np.ndarray | None = None) -> tuple[Path, dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _device()
    label_to_id = {label: i for i, label in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(cfg["pretrained"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["pretrained"], num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id=label_to_id,
    ).to(device)

    # Model cased: giữ chữ hoa (lowercase=False)
    train_norm = [normalize_text(t, lowercase=False) for t in train_texts]
    val_norm = [normalize_text(t, lowercase=False) for t in val_texts]
    y_train = np.array([label_to_id[label] for label in train_labels])
    y_val = np.array([label_to_id[label] for label in val_labels])
    weights = sample_weight if sample_weight is not None else np.ones(len(y_train))

    counts = np.bincount(y_train, minlength=len(labels))
    class_weight = torch.tensor(len(y_train) / (len(labels) * counts), dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weight, reduction="none")

    no_decay = ("bias", "LayerNorm.weight")
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=cfg["lr"])
    steps_per_epoch = math.ceil(len(train_norm) / cfg["batch_size"])
    total_steps = steps_per_epoch * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * cfg["warmup_ratio"]), total_steps)

    best_f1, best_state, history = -1.0, None, []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        order = np.random.permutation(len(train_norm))
        running = 0.0
        for step, start in enumerate(range(0, len(order), cfg["batch_size"]), start=1):
            idx = order[start:start + cfg["batch_size"]]
            encoded = _encode(tokenizer, [train_norm[i] for i in idx], cfg["max_len"])
            batch = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**batch).logits
            loss = criterion(logits, torch.tensor(y_train[idx], device=device))
            loss = (loss * torch.tensor(weights[idx], dtype=torch.float, device=device)).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if step % 100 == 0:
                logger.info("[DistilBERT] epoch %d step %d/%d loss=%.4f", epoch, step,
                            steps_per_epoch, running / step)
        val_pred = _predict(model, tokenizer, val_norm, cfg["max_len"], device).argmax(1)
        val_f1 = f1_score(y_val, val_pred, average="macro")
        history.append({"epoch": epoch, "train_loss": running / steps_per_epoch, "val_macro_f1": val_f1})
        logger.info("[DistilBERT] epoch %d val_macro_f1=%.4f", epoch, val_f1)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model = model.cpu().eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir / "hf_tokenizer")
    # tokenizer.json (fast tokenizer) dùng trực tiếp bởi thư viện `tokenizers` khi serving
    (output_dir / TOKENIZER_FILENAME).write_bytes((output_dir / "hf_tokenizer" / TOKENIZER_FILENAME).read_bytes())
    export_onnx(model, tokenizer, output_dir, cfg["max_len"])
    return output_dir, {"history": history, "best_val_macro_f1": best_f1}


class _LogitsOnly(nn.Module):
    """Bọc model HF để ONNX graph chỉ trả về logits (tensor), không trả ModelOutput."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export_onnx(model, tokenizer, output_dir: Path, max_len: int) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    sample = tokenizer(["xin chào", "free entry win cash now"], padding="max_length",
                       max_length=16, return_tensors="pt")
    fp32_path = output_dir / ONNX_FP32_FILENAME
    torch.onnx.export(
        _LogitsOnly(copy.deepcopy(model)), (sample["input_ids"], sample["attention_mask"]), str(fp32_path),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    # Dynamic INT8 quantization: giảm ~4x kích thước, tăng tốc CPU, sai số nhỏ
    quantize_dynamic(str(fp32_path), str(output_dir / ONNX_FILENAME), weight_type=QuantType.QInt8)
    fp32_path.unlink()
    logger.info("Đã export ONNX INT8 tại %s", output_dir / ONNX_FILENAME)
