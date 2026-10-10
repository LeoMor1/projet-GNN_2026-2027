"""Metriques pour la detection d'anomalies de liens."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.metrics import average_precision_score, roc_auc_score


def precision_at_k(y_true, y_score, k: int) -> float:
    """
    Precision@K : fraction des K scores les plus eleves qui sont des anomalies (label=1).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if k <= 0:
        raise ValueError("k must be > 0")
    k = min(k, len(y_true))
    order = np.argsort(-y_score)[:k]
    return float(y_true[order].sum() / k)


def evaluate_anomaly_ranking(
    rows: list[dict[str, Any]],
    anomaly_edges: set[tuple[int, int]],
    k: int = 10,
) -> dict[str, float]:
    """
    Evalue le ranking des scores d'anomalie contre le ground truth injecte.

    Chaque row doit contenir source, target, anomaly_score.
    """
    y_true = []
    y_score = []
    for r in rows:
        e = (min(r["source"], r["target"]), max(r["source"], r["target"]))
        y_true.append(1 if e in anomaly_edges else 0)
        y_score.append(r["anomaly_score"])

    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_score_arr = np.asarray(y_score, dtype=np.float64)

    n_pos = int(y_true_arr.sum())
    if n_pos == 0:
        raise ValueError("No positive anomalies in scored edges.")

    return {
        "auc": roc_auc_score(y_true_arr, y_score_arr),
        "average_precision": average_precision_score(y_true_arr, y_score_arr),
        "precision_at_k": precision_at_k(y_true_arr, y_score_arr, k),
        "n_edges_scored": float(len(rows)),
        "n_anomalies": float(n_pos),
    }


def summarize_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Agrege mean ± std par (model, anomaly_rate).

    rows: dicts avec model, anomaly_rate, auc, average_precision, precision_at_10
    """
    from collections import defaultdict

    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["model"], float(r["anomaly_rate"]))].append(r)

    summary = []
    for (model, rate), items in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        aucs = np.array([i["auc"] for i in items], dtype=np.float64)
        aps = np.array([i["average_precision"] for i in items], dtype=np.float64)
        pks = np.array([i["precision_at_10"] for i in items], dtype=np.float64)
        summary.append(
            {
                "model": model,
                "anomaly_rate": rate,
                "auc_mean": float(aucs.mean()),
                "auc_std": float(aucs.std(ddof=1) if len(aucs) > 1 else 0.0),
                "ap_mean": float(aps.mean()),
                "ap_std": float(aps.std(ddof=1) if len(aps) > 1 else 0.0),
                "precision10_mean": float(pks.mean()),
                "precision10_std": float(pks.std(ddof=1) if len(pks) > 1 else 0.0),
                "n_seeds": len(items),
            }
        )
    return summary
