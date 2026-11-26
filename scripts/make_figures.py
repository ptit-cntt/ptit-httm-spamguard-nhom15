"""Sinh biểu đồ EDA, so sánh model và sơ đồ kiến trúc vào reports/figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spamguard.registry import ModelRegistry  # noqa: E402
from spamguard.text_preprocessing import normalize_text  # noqa: E402
from spamguard.training import MODEL_DISPLAY_NAMES  # noqa: E402

FIG_DIR = ROOT / "reports" / "figures"
COLORS = {"ham": "#16a34a", "spam": "#f59e0b", "toxic": "#dc2626"}
MODEL_COLORS = ["#64748b", "#0ea5e9", "#8b5cf6", "#2563eb"]
DPI = 160

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _upper_ratio(text: str) -> float:
    letters = sum(c.isalpha() for c in text)
    return sum(c.isupper() for c in text) / max(1, letters)


def eda(df: pd.DataFrame) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    pivot = df.groupby(["source", "label"]).size().unstack(fill_value=0)[["ham", "spam", "toxic"]]
    pivot.plot(kind="barh", stacked=True, ax=axes[0], color=[COLORS[c] for c in pivot.columns])
    axes[0].set_title("Số mẫu theo nguồn và nhãn")
    axes[0].set_xlabel("Số tin nhắn")
    axes[0].set_ylabel("")
    lang = df.groupby(["lang", "label"]).size().unstack(fill_value=0)[["ham", "spam", "toxic"]]
    lang.plot(kind="bar", ax=axes[1], color=[COLORS[c] for c in lang.columns], rot=0)
    axes[1].set_title("Phân bố nhãn theo ngôn ngữ")
    axes[1].set_xlabel("")
    _save(fig, "eda_distribution.png")

    df = df.assign(n_chars=df["text"].str.len(), n_words=df["text"].str.split().str.len())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for label, color in COLORS.items():
        subset = df[df["label"] == label]
        axes[0].hist(subset["n_words"].clip(upper=80), bins=40, alpha=0.55, color=color, label=label, density=True)
    axes[0].set_title("Phân phối độ dài (số từ, cắt tại 80)")
    axes[0].legend()
    stats = df.groupby(["lang", "label"])["n_words"].median().unstack()
    stats.plot(kind="bar", ax=axes[1], color=[COLORS[c] for c in stats.columns], rot=0)
    axes[1].set_title("Trung vị số từ theo ngôn ngữ / nhãn")
    _save(fig, "eda_length.png")

    # Tỉ lệ tin nhắn chứa các tín hiệu đặc biệt sau chuẩn hoá
    norm = df["text"].map(normalize_text)
    signals = {
        "Chứa URL": norm.str.contains("<url>", regex=False),
        "Chứa số ĐT": norm.str.contains("<phone>", regex=False),
        "Chứa số >=3 chữ số": norm.str.contains("<num>", regex=False),
        "Tỉ lệ chữ HOA > 30%": df["text"].map(_upper_ratio) > 0.3,
    }
    signal_df = pd.DataFrame({name: s.groupby(df["label"]).mean() for name, s in signals.items()}).T
    fig, ax = plt.subplots(figsize=(8, 3.4))
    signal_df[["ham", "spam", "toxic"]].plot(kind="barh", ax=ax, color=[COLORS[c] for c in ("ham", "spam", "toxic")])
    ax.set_title("Tỉ lệ tin nhắn có tín hiệu đặc trưng theo nhãn")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _save(fig, "eda_signals.png")
    return {
        "length_median": df.groupby("label")["n_words"].median().to_dict(),
        "length_mean": df.groupby("label")["n_words"].mean().round(1).to_dict(),
        "signals": signal_df.round(4).to_dict(),
    }


def model_charts(metrics: dict) -> None:
    results = metrics["results"]
    models = [m for m in ("svm", "random_forest", "bigru", "distilbert") if m in results]
    names = [MODEL_DISPLAY_NAMES[m] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    groups = [("Macro-F1", lambda r: r["macro_f1"]), ("F1 tiếng Anh", lambda r: r["per_lang_macro_f1"]["en"]),
              ("F1 tiếng Việt", lambda r: r["per_lang_macro_f1"]["vi"]), ("ROC-AUC", lambda r: r["roc_auc_ovr"])]
    x = np.arange(len(groups))
    width = 0.2
    for i, model in enumerate(models):
        values = [fn(results[model]) for _, fn in groups]
        bars = axes[0].bar(x + (i - 1.5) * width, values, width, label=names[i], color=MODEL_COLORS[i])
        axes[0].bar_label(bars, fmt="%.3f", fontsize=6.5, rotation=90, padding=2)
    axes[0].set_xticks(x, [g for g, _ in groups])
    axes[0].set_ylim(0.6, 1.04)
    axes[0].set_title("Chất lượng trên tập test (5.080 tin)")
    axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False)

    latency = [results[m]["p95_latency_ms"] for m in models]
    bars = axes[1].bar(names, latency, color=MODEL_COLORS[:len(models)])
    axes[1].bar_label(bars, fmt="%.2f ms", fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_title("Độ trễ p95 / 1 tin nhắn trên CPU (thang log)")
    axes[1].tick_params(axis="x", labelsize=8)
    _save(fig, "model_comparison.png")

    labels = ["ham", "spam", "toxic"]
    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.2))
    for ax, model, name in zip(np.atleast_1d(axes), models, names):
        cm = np.array(results[model]["confusion_matrix"], dtype=float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=7.5,
                        color="white" if cm_norm[i, j] > 0.6 else "black")
        ax.set_xticks(range(3), labels)
        ax.set_yticks(range(3), labels)
        ax.set_xlabel("Dự đoán")
        ax.set_title(name, fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("Thực tế")
    _save(fig, "confusion_matrices.png")

    registry = ModelRegistry()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for model, color in (("bigru", MODEL_COLORS[2]), ("distilbert", MODEL_COLORS[3])):
        if model not in metrics["versions"]:
            continue
        history = registry.get_meta(metrics["versions"][model]).get("history", [])
        ax.plot([h["epoch"] for h in history], [h["val_macro_f1"] for h in history], marker="o",
                color=color, label=MODEL_DISPLAY_NAMES[model])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro-F1 (validation)")
    ax.set_title("Đường học của 2 mô hình học sâu")
    ax.legend()
    _save(fig, "training_curves.png")


def _box(ax, xy, w, h, text, color, fontsize=8.5):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=color, edgecolor="#334155", linewidth=1))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def _arrow(ax, start, end, text="", style="-|>", color="#334155", rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=12, color=color,
                                 connectionstyle=f"arc3,rad={rad}", linewidth=1.2))
    if text:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.12, text, fontsize=7, ha="center", color="#475569")


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    client, serving, data, mlops, ops = "#e0f2fe", "#dcfce7", "#fef9c3", "#ede9fe", "#fee2e2"

    ax.text(0.2, 7.1, "SERVING LAYER", fontsize=9, weight="bold", color="#166534")
    _box(ax, (0.2, 5.6), 2.2, 1.1, "Nền tảng nhắn tin\n(SMS / Chat / Comment)", client)
    _box(ax, (0.2, 4.0), 2.2, 1.1, "Quản trị viên\n(trình duyệt)", client)
    _box(ax, (3.2, 4.0), 3.4, 2.7, "SpamGuard API (FastAPI)\n─────────────\n/v1/messages/classify\n/v1/feedback\n"
         "/v1/admin/* · /metrics\nWeb Admin (/admin)\nModelManager + hot-reload\nONNX Runtime / scikit-learn", serving)
    _box(ax, (7.4, 5.6), 2.4, 1.1, "PostgreSQL\nmessages · feedback", data)
    _box(ax, (7.4, 4.0), 2.4, 1.1, "Model Registry\nv1..vN + alias production", data)
    _box(ax, (10.6, 5.6), 2.2, 1.1, "Prometheus\n+ Alert rules", ops)
    _box(ax, (10.6, 4.0), 2.2, 1.1, "Grafana\nDashboard", ops)

    ax.text(0.2, 3.3, "MLOps LAYER", fontsize=9, weight="bold", color="#5b21b6")
    _box(ax, (0.2, 1.6), 2.2, 1.3, "Data sources\nUCI SMS · Davidson\nViSpam · UIT-ViHSD", data)
    _box(ax, (3.0, 1.6), 2.2, 1.3, "Data pipeline\ndownload → clean →\ndedup → split", mlops)
    _box(ax, (5.8, 1.6), 2.4, 1.3, "Training & Eval\nSVM · RF · Bi-GRU\nDistilBERT → ONNX", mlops)
    _box(ax, (8.8, 1.6), 1.9, 1.3, "MLflow\nTracking", mlops)
    _box(ax, (11.0, 1.6), 1.8, 1.3, "Quality gate\n& promote", mlops)
    _box(ax, (3.0, 0.1), 7.7, 1.0, "CI/CD (GitHub Actions): lint → test → build Docker image → push GHCR → deploy  |  "
         "mlops-retrain: cron → retrain_from_feedback.py", "#f1f5f9", fontsize=8)

    _arrow(ax, (2.4, 6.15), (3.2, 6.15), "classify")
    _arrow(ax, (2.4, 4.55), (3.2, 4.55), "dashboard")
    _arrow(ax, (6.6, 6.15), (7.4, 6.15), "log")
    _arrow(ax, (7.4, 4.55), (6.6, 4.55), "load")
    _arrow(ax, (11.7, 6.7), (5.6, 6.7), rad=0.18)
    ax.text(8.6, 7.15, "scrape /metrics", fontsize=7, ha="center", color="#475569")
    _arrow(ax, (11.7, 5.6), (11.7, 5.1))
    _arrow(ax, (2.4, 2.25), (3.0, 2.25))
    _arrow(ax, (5.2, 2.25), (5.8, 2.25))
    _arrow(ax, (8.2, 2.25), (8.8, 2.25))
    _arrow(ax, (10.7, 2.25), (11.0, 2.25))
    _arrow(ax, (11.9, 2.9), (9.8, 4.0), "register\n+ promote", rad=0.1)
    _arrow(ax, (7.0, 6.1), (7.0, 2.9), color="#7c3aed")
    ax.text(6.9, 3.4, "feedback\n(admin review)", fontsize=7, ha="right", color="#7c3aed")
    _save(fig, "architecture.png")


def feedback_loop() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(-2.9, 2.9)
    ax.axis("off")
    steps = ["1. Serve\nAPI phân loại,\nlog dự đoán", "2. Monitor\nPrometheus: latency,\ntỉ lệ spam, drift",
             "3. Review\nAdmin gán nhãn tin\nconf. thấp / bị báo cáo", "4. Collect\nFeedback (admin)\nvào PostgreSQL",
             "5. Retrain\nGộp train + feedback\n(upsample x3)", "6. Evaluate & Gate\nTest cố định + holdout\nfeedback",
             "7. Register & Promote\nRegistry v(N+1) →\nproduction, hot-reload"]
    n = len(steps)
    angles = [np.pi / 2 - 2 * np.pi * i / n for i in range(n)]
    positions = [(4.0 * np.cos(a), 2.0 * np.sin(a)) for a in angles]
    for i, ((x, y), text) in enumerate(zip(positions, steps)):
        _box(ax, (x - 1.1, y - 0.45), 2.2, 0.9, text, "#ede9fe" if i >= 3 else "#dcfce7", fontsize=7.5)
    for i in range(n):
        (x1, y1), (x2, y2) = positions[i], positions[(i + 1) % n]
        _arrow(ax, (x1 + (x2 - x1) * 0.3, y1 + (y2 - y1) * 0.3), (x1 + (x2 - x1) * 0.7, y1 + (y2 - y1) * 0.7),
               color="#7c3aed")
    ax.text(0, 0, "MLOps\nFeedback Loop", ha="center", va="center", fontsize=13, weight="bold", color="#5b21b6")
    _save(fig, "feedback_loop.png")


def deployment() -> None:
    """Sơ đồ triển khai thực tế: Cloudflare Pages -> Nginx server -> Docker PC LAN."""
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.8)
    ax.axis("off")
    cloud, server, lan, user = "#fde7d4", "#e0f2fe", "#dcfce7", "#f1f5f9"
    _box(ax, (0.1, 2.4), 1.9, 1.2, "Người dùng /\nNền tảng nhắn tin", user)
    ax.add_patch(FancyBboxPatch((2.5, 1.0), 2.8, 4.1, boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor="none", edgecolor="#f97316", linestyle="--", linewidth=1.2))
    ax.text(3.9, 4.85, "Cloudflare (Internet)", ha="center", fontsize=9, weight="bold", color="#c2410c")
    _box(ax, (2.75, 2.2), 2.3, 1.6, "spamguard.pages.dev\nPages Function\n(HTTPS, reverse proxy\n+ header bí mật)", cloud)
    ax.add_patch(FancyBboxPatch((5.9, 1.0), 2.9, 4.1, boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor="none", edgecolor="#0284c7", linestyle="--", linewidth=1.2))
    ax.text(7.35, 4.85, "Server 58.187.192.54 (LAN .222)", ha="center", fontsize=9, weight="bold", color="#0369a1")
    _box(ax, (6.15, 2.2), 2.4, 1.6, "Nginx :80\nspamguard.58-187-192-54\n.sslip.io\n(chặn nếu thiếu header)", server)
    ax.add_patch(FancyBboxPatch((9.4, 0.3), 3.5, 4.8, boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor="none", edgecolor="#16a34a", linestyle="--", linewidth=1.2))
    ax.text(11.15, 4.85, "PC LAN 192.168.1.5 – Docker", ha="center", fontsize=9, weight="bold", color="#15803d")
    for i, (label, y) in enumerate([("api :8010 (FastAPI + ONNX)", 3.95), ("postgres (volume pgdata)", 3.2),
                                     ("prometheus :9095", 2.45), ("grafana :3010", 1.7), ("mlflow :5010", 0.95)]):
        _box(ax, (9.6, y - 0.25), 3.1, 0.55, label, lan if i == 0 else "#f0fdf4", fontsize=8)
    _arrow(ax, (2.0, 3.0), (2.75, 3.0))
    _arrow(ax, (5.05, 3.0), (6.15, 3.0))
    _arrow(ax, (8.55, 3.0), (9.6, 4.0))
    for x, y, label in ((2.38, 3.3, "HTTPS"), (5.6, 3.3, "HTTP +\nX-Origin-Auth"), (8.85, 3.75, "LAN\n:8010")):
        ax.text(x, y, label, fontsize=7.5, ha="center", va="bottom", color="#475569")
    ax.text(6.5, 0.45, "Điều khiển Docker: server → tcp://192.168.1.5:2375 (chỉ mở cho IP server)",
            fontsize=8, ha="center", color="#475569")
    _save(fig, "deployment.png")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    processed = ROOT / "data" / "processed"
    df = pd.concat([pd.read_csv(processed / f"{s}.csv", keep_default_na=False) for s in ("train", "val", "test")])
    eda_stats = eda(df)
    (ROOT / "reports" / "eda_stats.json").write_text(json.dumps(eda_stats, indent=2, ensure_ascii=False))
    metrics_path = ROOT / "reports" / "metrics.json"
    if metrics_path.exists():
        model_charts(json.loads(metrics_path.read_text()))
    architecture()
    feedback_loop()
    deployment()
    print("Đã sinh hình tại", FIG_DIR)


if __name__ == "__main__":
    main()
