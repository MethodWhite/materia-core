#!/usr/bin/env python3
"""
MATERIA V4 1B Training Log Plotter
Parses training logs and generates scientific plots with matplotlib.
"""

import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─── Config ────────────────────────────────────────────────────────────────
LOG_PATH = "/home/methodwhite/MATERIA/logs/training_e2_bs64_sl128_full.log"
OUT_DIR  = "/home/methodwhite/MATERIA/outputs/plots"
DPI      = 150

# Color palette (professional dark tones)
C_BLUE   = "#1f77b4"
C_RED    = "#d62728"
C_GREEN  = "#2ca02c"
C_ORANGE = "#ff7f0e"
C_PURPLE = "#9467bd"
C_CYAN   = "#17becf"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 100,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
})


def parse_log(path: str) -> list[dict]:
    """Parse training log lines into a list of metric dicts."""
    if not os.path.isfile(path):
        print(f"[ERROR] Log file not found: {path}")
        sys.exit(1)

    records = []
    # Match the core training step line (might have extended metrics after |)
    pattern = re.compile(
        r"E2/2\s+\[(\d+)/\d+\]\s+"
        r"loss=([\d.e+\-]+)\s+"
        r"tok=([\d.e+\-]+)\s+"
        r"jepa=([\d.e+\-]+)\s+"
        r"acc=([\d.e+\-]+)\s+"
        r"ppl=([\d.e+\-]+)\s+"
        r"spike=([\d.e+\-]+)\s+"
        r"spk_reg=([\d.e+\-]+)\s+"
        r"hsaq_sp=([\d.e+\-]+)\(tgt=([\d.e+\-]+)\)\s+"
        r"th=([\d.e+\-]+)\s+"
        r"lr=([\d.e+\-]+)"
    )

    skipped = 0
    with open(path, "r") as f:
        for lineno, line in enumerate(f, 1):
            m = pattern.search(line)
            if not m:
                continue
            try:
                rec = {
                    "batch":    int(m.group(1)),
                    "loss":     float(m.group(2)),
                    "tok_loss": float(m.group(3)),
                    "jepa":     float(m.group(4)),
                    "acc":      float(m.group(5)),
                    "ppl":      float(m.group(6)),
                    "spike":    float(m.group(7)),
                    "spk_reg":  float(m.group(8)),
                    "hsaq_sp":  float(m.group(9)),
                    "hsaq_tgt": float(m.group(10)),
                    "th":       float(m.group(11)),
                    "lr":       float(m.group(12)),
                }
                records.append(rec)
            except (ValueError, IndexError):
                skipped += 1

    if not records:
        print("[ERROR] No training records parsed. Check log format.")
        sys.exit(1)

    print(f"  Parsed {len(records)} records, {skipped} skipped malformed lines")
    return records


def _smooth(y: np.ndarray, window: int = 11) -> np.ndarray:
    """Simple moving average for smoothing noisy curves."""
    if window < 3 or len(y) < window:
        return y
    half = window // 2
    kernel = np.ones(window) / window
    padded = np.pad(y, (half, half), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[:len(y)]


def _style_ax(ax: plt.Axes, title: str, xlabel: str, ylabel: str):
    """Apply scientific styling to an axis."""
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()
    ax.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))


def plot_loss(records: list[dict], save_path: str):
    """Token loss + JEPA loss vs batches."""
    batches = np.array([r["batch"] for r in records])
    tok     = np.array([r["tok_loss"] for r in records])
    jepa    = np.array([r["jepa"] for r in records])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(batches, _smooth(tok),  color=C_BLUE,  lw=1.2, label="Token Loss (smooth)")
    ax.plot(batches, tok,           color=C_BLUE,  lw=0.3, alpha=0.3)
    ax.plot(batches, _smooth(jepa), color=C_RED,   lw=1.2, label="JEPA Loss (smooth)")
    ax.plot(batches, jepa,          color=C_RED,   lw=0.3, alpha=0.3)
    _style_ax(ax, "MATERIA V4 1B — Training Loss", "Batch", "Loss")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [OK] {save_path}")


def plot_accuracy(records: list[dict], save_path: str):
    """Accuracy vs batches."""
    batches = np.array([r["batch"] for r in records])
    acc     = np.array([r["acc"] for r in records])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(batches, _smooth(acc),  color=C_GREEN, lw=1.2, label="Accuracy (smooth)")
    ax.plot(batches, acc,           color=C_GREEN, lw=0.3, alpha=0.3)
    _style_ax(ax, "MATERIA V4 1B — Training Accuracy", "Batch", "Accuracy")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [OK] {save_path}")


def plot_perplexity(records: list[dict], save_path: str):
    """Perplexity vs batches."""
    batches = np.array([r["batch"] for r in records])
    ppl     = np.array([r["ppl"] for r in records])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(batches, _smooth(ppl),  color=C_PURPLE, lw=1.2, label="Perplexity (smooth)")
    ax.plot(batches, ppl,           color=C_PURPLE, lw=0.3, alpha=0.3)
    _style_ax(ax, "MATERIA V4 1B — Perplexity", "Batch", "Perplexity")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [OK] {save_path}")


def plot_spike_rate(records: list[dict], save_path: str):
    """SNN Spike rate vs batches."""
    batches = np.array([r["batch"] for r in records])
    spike   = np.array([r["spike"] for r in records])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(batches, _smooth(spike), color=C_ORANGE, lw=1.2, label="Spike Rate (smooth)")
    ax.plot(batches, spike,          color=C_ORANGE, lw=0.3, alpha=0.3)
    _style_ax(ax, "MATERIA V4 1B — SNN Spike Rate", "Batch", "Spike Rate")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [OK] {save_path}")


def plot_hsaq_sparsity(records: list[dict], save_path: str):
    """HSAQ sparsity real vs target vs batches."""
    batches = np.array([r["batch"] for r in records])
    real    = np.array([r["hsaq_sp"] for r in records])
    target  = np.array([r["hsaq_tgt"] for r in records])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(batches, _smooth(real),  color=C_BLUE, lw=1.2, label="HSAQ Sparsity (smooth)")
    ax.plot(batches, real,           color=C_BLUE, lw=0.3, alpha=0.3)
    ax.axhline(y=np.mean(target), color=C_RED, ls="--", lw=1.2,
               label=f"Target ({np.mean(target):.3f})")
    ax.fill_between(batches, target.min(), target.max(),
                    alpha=0.08, color=C_RED, label="Target range")
    _style_ax(ax, "MATERIA V4 1B — HSAQ Sparsity", "Batch", "Sparsity")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [OK] {save_path}")


def plot_learning_rate(records: list[dict], save_path: str):
    """Learning rate schedule vs batches."""
    batches = np.array([r["batch"] for r in records])
    lr      = np.array([r["lr"] for r in records])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(batches, lr, color=C_GREEN, lw=1.0, label="LR")
    ax.fill_between(batches, 0, lr, alpha=0.15, color=C_GREEN)
    _style_ax(ax, "MATERIA V4 1B — Learning Rate Schedule", "Batch", "Learning Rate")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [OK] {save_path}")


def plot_combined(records: list[dict], save_path: str):
    """3×2 combined figure with all metrics."""
    batches = np.array([r["batch"] for r in records])
    tok     = np.array([r["tok_loss"] for r in records])
    jepa    = np.array([r["jepa"] for r in records])
    acc     = np.array([r["acc"] for r in records])
    ppl     = np.array([r["ppl"] for r in records])
    spike   = np.array([r["spike"] for r in records])
    hsaq_r  = np.array([r["hsaq_sp"] for r in records])
    hsaq_t  = np.array([r["hsaq_tgt"] for r in records])
    lr      = np.array([r["lr"] for r in records])

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    # (1,1) Loss
    ax = axes[0, 0]
    ax.plot(batches, _smooth(tok),  color=C_BLUE, lw=1.0, label="Token Loss")
    ax.plot(batches, tok,           color=C_BLUE, lw=0.2, alpha=0.2)
    ax.plot(batches, _smooth(jepa), color=C_RED,  lw=1.0, label="JEPA Loss")
    ax.plot(batches, jepa,          color=C_RED,  lw=0.2, alpha=0.2)
    ax.set_title("Loss", fontweight="bold"); ax.set_xlabel("Batch"); ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3, ls="--"); ax.legend(fontsize=8)

    # (1,2) Accuracy
    ax = axes[0, 1]
    ax.plot(batches, _smooth(acc), color=C_GREEN, lw=1.0, label="Accuracy")
    ax.plot(batches, acc,          color=C_GREEN, lw=0.2, alpha=0.2)
    ax.set_title("Accuracy", fontweight="bold"); ax.set_xlabel("Batch"); ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3, ls="--"); ax.legend(fontsize=8)

    # (2,1) Perplexity
    ax = axes[1, 0]
    ax.plot(batches, _smooth(ppl), color=C_PURPLE, lw=1.0, label="Perplexity")
    ax.plot(batches, ppl,          color=C_PURPLE, lw=0.2, alpha=0.2)
    ax.set_title("Perplexity", fontweight="bold"); ax.set_xlabel("Batch"); ax.set_ylabel("PPL")
    ax.grid(True, alpha=0.3, ls="--"); ax.legend(fontsize=8)

    # (2,2) Spike Rate
    ax = axes[1, 1]
    ax.plot(batches, _smooth(spike), color=C_ORANGE, lw=1.0, label="Spike Rate")
    ax.plot(batches, spike,          color=C_ORANGE, lw=0.2, alpha=0.2)
    ax.set_title("SNN Spike Rate", fontweight="bold"); ax.set_xlabel("Batch"); ax.set_ylabel("Rate")
    ax.grid(True, alpha=0.3, ls="--"); ax.legend(fontsize=8)

    # (3,1) HSAQ Sparsity
    ax = axes[2, 0]
    ax.plot(batches, _smooth(hsaq_r), color=C_BLUE, lw=1.0, label="Real Sparsity")
    ax.plot(batches, hsaq_r,          color=C_BLUE, lw=0.2, alpha=0.2)
    ax.axhline(y=np.mean(hsaq_t), color=C_RED, ls="--", lw=1.0,
               label=f"Target ({np.mean(hsaq_t):.3f})")
    ax.set_title("HSAQ Sparsity", fontweight="bold"); ax.set_xlabel("Batch"); ax.set_ylabel("Sparsity")
    ax.grid(True, alpha=0.3, ls="--"); ax.legend(fontsize=8)

    # (3,2) Learning Rate
    ax = axes[2, 1]
    ax.plot(batches, lr, color=C_GREEN, lw=1.0)
    ax.fill_between(batches, 0, lr, alpha=0.15, color=C_GREEN)
    ax.set_title("Learning Rate", fontweight="bold"); ax.set_xlabel("Batch"); ax.set_ylabel("LR")
    ax.grid(True, alpha=0.3, ls="--")

    fig.suptitle("MATERIA V4 1B — Training Metrics (Epoch 2)", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [OK] {save_path}")


def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Reading: {LOG_PATH}")
    records = parse_log(LOG_PATH)

    print(f"\nGenerating plots in: {OUT_DIR}/")
    print("-" * 55)

    plots = [
        ("training_loss.png",       plot_loss),
        ("training_accuracy.png",   plot_accuracy),
        ("perplexity.png",          plot_perplexity),
        ("spike_rate.png",          plot_spike_rate),
        ("hsaq_sparsity.png",       plot_hsaq_sparsity),
        ("learning_rate.png",       plot_learning_rate),
        ("all_metrics.png",         plot_combined),
    ]

    for fname, func in plots:
        func(records, os.path.join(OUT_DIR, fname))

    # Report
    print("-" * 55)
    print(f"\n{'File':<30s} {'Size (KB)':>10s}")
    print("-" * 42)
    total_kb = 0
    for fname, _ in plots:
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.isfile(fpath):
            kb = os.path.getsize(fpath) / 1024
            total_kb += kb
            print(f"  {fname:<28s} {kb:>8.1f}")
        else:
            print(f"  {fname:<28s} {'MISSING':>8s}")
    print("-" * 42)
    print(f"  {'Total':<28s} {total_kb:>8.1f} KB")
    print(f"\nAll plots saved to: {OUT_DIR}/")
    print(f"DPI: {DPI} | Records: {len(records)}")


if __name__ == "__main__":
    main()
