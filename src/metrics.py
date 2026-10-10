"""ROC-AUC et Average Precision en NumPy (pas de dépendance à sklearn).

Les égalités de scores sont traitées par blocs : des scores constants donnent
AUC = 0,5 et AP = proportion de positifs, comme avec sklearn.
"""

from __future__ import annotations

import numpy as np


def _prepare(y_true, y_score):
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(y_score, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError("y_true et y_score doivent avoir la même taille.")
    return y, s


def _average_ranks(s: np.ndarray) -> np.ndarray:
    """Rangs (1..n) avec rang moyen pour les égalités."""
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    boundaries = np.flatnonzero(np.r_[True, sorted_s[1:] != sorted_s[:-1], True])
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        ranks[order[a:b]] = 0.5 * (a + 1 + b)
    return ranks


def roc_auc_score(y_true, y_score) -> float:
    y, s = _prepare(y_true, y_score)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("L'AUC demande des exemples positifs et négatifs.")
    ranks = _average_ranks(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision_score(y_true, y_score) -> float:
    y, s = _prepare(y_true, y_score)
    n_pos = float((y == 1).sum())
    if n_pos == 0:
        raise ValueError("L'AP demande au moins un exemple positif.")
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys == 1)
    precision = tp / np.arange(1, len(ys) + 1)
    # fin de chaque bloc de scores égaux
    last = np.r_[ss[1:] != ss[:-1], True]
    tp_b, prec_b = tp[last], precision[last]
    gain = np.diff(np.r_[0.0, tp_b])
    return float((prec_b * gain).sum() / n_pos)
