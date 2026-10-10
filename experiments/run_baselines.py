"""
Baselines sans apprentissage pour la détection de liens injectés.

Même protocole que run_comparison.py (taux et graines de src/config.py), mais
les scores viennent directement du graphe :
  Distance     distance grand-cercle entre les deux aéroports (km)
  Adamic-Adar  opposé de la somme des 1/log(degré) des voisins communs
  Jaccard      opposé de |voisins communs| / |union des voisins|

Les liens injectés sont tirés avec NumPy (default_rng(seed)) et non avec
torch.Generator : même loi que pour GAE/VGAE, mais pas les mêmes liens.
Ce script ne demande pas PyTorch.

Usage : python experiments/run_baselines.py
Sortie : results/csv/baselines_liens.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ANOMALY_RATES, DATA_PATH, RESULTS_CSV_DIR, SEEDS
from src.metrics import average_precision_score, roc_auc_score


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p = np.pi / 180
    a = (
        np.sin((lat2 - lat1) * p / 2) ** 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    )
    return float(2 * 6371.0 * np.arcsin(np.sqrt(a)))


def inject_false_edges(G: nx.Graph, rate: float, seed: int):
    """Copie de G + round(rate * |E|) arêtes entre paires non reliées."""
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    existing = {(min(u, v), max(u, v)) for u, v in G.edges()}
    n_anom = int(round(rate * len(existing)))
    anomalies: set[tuple[int, int]] = set()
    while len(anomalies) < n_anom:
        i, j = rng.integers(0, len(nodes), 2)
        if i == j:
            continue
        e = (min(nodes[i], nodes[j]), max(nodes[i], nodes[j]))
        if e not in existing and e not in anomalies:
            anomalies.add(e)
    H = G.copy()
    H.add_edges_from(anomalies)
    return H, anomalies


def edge_scores(H: nx.Graph, edges: list[tuple[int, int]]) -> dict[str, np.ndarray]:
    lat = {n: float(d["lat"]) for n, d in H.nodes(data=True)}
    lon = {n: float(d["lon"]) for n, d in H.nodes(data=True)}
    neigh = {n: set(H[n]) for n in H}
    deg = dict(H.degree())
    dist = np.zeros(len(edges))
    adamic = np.zeros(len(edges))
    jaccard = np.zeros(len(edges))
    for k, (u, v) in enumerate(edges):
        dist[k] = haversine_km(lat[u], lon[u], lat[v], lon[v])
        common = neigh[u] & neigh[v]
        adamic[k] = sum(1.0 / np.log(deg[w]) for w in common if deg[w] > 1)
        union = len(neigh[u] | neigh[v])
        jaccard[k] = len(common) / union if union else 0.0
    return {"Distance": dist, "Adamic-Adar": -adamic, "Jaccard": -jaccard}


def main() -> None:
    G = nx.read_graphml(DATA_PATH, node_type=int)
    rows = []
    for rate in ANOMALY_RATES:
        for seed in SEEDS:
            H, anomalies = inject_false_edges(G, rate, seed)
            edges = [(min(u, v), max(u, v)) for u, v in H.edges() if u != v]
            y = np.array([e in anomalies for e in edges], dtype=float)
            for name, s in edge_scores(H, edges).items():
                rows.append(
                    {
                        "model": name,
                        "seed": seed,
                        "anomaly_rate": rate,
                        "auc": f"{roc_auc_score(y, s):.6f}",
                        "average_precision": f"{average_precision_score(y, s):.6f}",
                    }
                )
            print(f"r={rate:.2f} seed={seed} ok", flush=True)

    out = RESULTS_CSV_DIR / "baselines_liens.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\nméthode       r      AUC            AP")
    for name in ("Distance", "Jaccard", "Adamic-Adar"):
        for rate in ANOMALY_RATES:
            sel = [r for r in rows if r["model"] == name and r["anomaly_rate"] == rate]
            a = np.array([float(r["auc"]) for r in sel])
            p = np.array([float(r["average_precision"]) for r in sel])
            print(f"{name:12s} {rate:5.2f}  {a.mean():.3f}±{a.std(ddof=1):.3f}  {p.mean():.3f}±{p.std(ddof=1):.3f}")
    print(f"\nÉcrit : {out}")


if __name__ == "__main__":
    main()
