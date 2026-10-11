"""Evaluation de DOMINANT avec des anomalies de noeuds injectees (AUC / AP).

Usage (depuis la racine du depot) :
    python src/evaluate_gcn.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx

from src.gcn import run
from src.metrics import average_precision_score, roc_auc_score

DATA_PATH = ROOT / "data" / "airportsAndCoordAndPop.graphml"
RATES = [0.01, 0.05, 0.10, 0.20]       # part de nœuds déplacés
SEEDS = [42, 123, 456, 789, 2026]
N_EPOCH = 500


def inject_position_anomalies(data, rate, seed):
    """Déplace `rate * N` nœuds en un point tiré uniformément sur la sphère.

    Les liens ne changent pas : le nœud garde ses vrais voisins mais a de
    fausses coordonnées, comme les aéroports mal géolocalisés du fichier.
    Retourne une copie de data et y (1 = anomalie, 0 = normal).
    """
    g = torch.Generator().manual_seed(seed)
    n = data.num_nodes
    k = max(1, int(rate * n))
    idx = torch.randperm(n, generator=g)[:k]

    d = data.clone()
    d.x = data.x.clone()
    # latitude = asin(U(-1, 1)) donne une répartition uniforme sur la sphère
    d.x[idx, 0] = torch.rad2deg(torch.asin(2 * torch.rand(k, generator=g) - 1))
    d.x[idx, 1] = 360 * torch.rand(k, generator=g) - 180

    y = torch.zeros(n)
    y[idx] = 1
    return d, y


def main() -> None:
    """Point d'entree evaluation DOMINANT (racine : python src/evaluate_gcn.py)."""
    G = nx.read_graphml(DATA_PATH, node_type=int)
    data = from_networkx(G, group_node_attrs=["lat", "lon", "population"])

    for rate in RATES:
        aucs, aps = [], []
        for seed in SEEDS:
            d, y = inject_position_anomalies(data, rate, seed)
            torch.manual_seed(seed)
            score = run(d, N_EPOCH, n_runs=1)[0]

            y_np, s_np = y.numpy(), score.numpy()
            aucs.append(roc_auc_score(y_np, s_np))
            aps.append(average_precision_score(y_np, s_np))

        print(f"\n=== r = {rate:.0%} (AP aleatoire = {rate:.3f}) ===")
        print(f"AUC = {np.mean(aucs):.3f} ({np.std(aucs):.3f})")
        print(f"AP  = {np.mean(aps):.3f} ({np.std(aps):.3f})")


if __name__ == "__main__":
    main()