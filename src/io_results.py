"""Ecriture CSV / figures de resultats."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt

from src.config import RESULTS_CSV_DIR, RESULTS_FIG_DIR


def ensure_result_dirs() -> None:
    RESULTS_CSV_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIG_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def plot_metric_vs_rate(
    summary_rows: list[dict[str, Any]],
    metric_mean_key: str,
    metric_std_key: str,
    ylabel: str,
    out_path: Path,
    title: str,
) -> None:
    """Courbe mean±std vs anomaly_rate pour GAE et VGAE."""
    ensure_result_dirs()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for model, color in (("GAE", "#1f77b4"), ("VGAE", "#d62728")):
        items = [r for r in summary_rows if r["model"] == model]
        items = sorted(items, key=lambda r: float(r["anomaly_rate"]))
        if not items:
            continue
        xs = [float(r["anomaly_rate"]) * 100 for r in items]
        ys = [float(r[metric_mean_key]) for r in items]
        yerr = [float(r[metric_std_key]) for r in items]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", label=model, color=color, capsize=3)

    ax.set_xlabel("Anomaly rate (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
