"""
Ablation study sur GAE (modele final choisi par parcimonie).

Taux d'anomalie fixe : 10 %. Seeds : celles de src/config.py (5).
Variantes :
  - full          : 2 couches GCN + features normalisees (lat, lon, pop)
  - one_layer     : 1 seule couche GCN
  - no_norm       : features brutes (sans standardisation)
  - no_features   : X = I (pas de lat/lon/population)
  - small_latent  : latent_dim = 8 au lieu de 32

Sortie : results/csv/ablation_gae.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch_geometric.nn import GAE, GCNConv
from torch_geometric.utils import from_networkx

from src.config import (
    EPOCHS,
    HIDDEN_CHANNELS,
    LATENT_DIM,
    LEARNING_RATE,
    NODE_FEATURES,
    PRECISION_AT_K,
    RESULTS_CSV_DIR,
    SEEDS,
)
from src.data import load_airport_graph, make_link_split, normalize_features
from src.evaluation import evaluate_anomaly_ranking, summarize_results
from src.io_results import ensure_result_dirs, write_csv
from src.link_anomaly import compute_edge_anomaly_scores, inject_false_edges
from src.utils import get_device, set_seed


ANOMALY_RATE = 0.10


class Encoder2Layer(torch.nn.Module):
    def __init__(self, in_ch, hidden, out_ch):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden)
        self.conv2 = GCNConv(hidden, out_ch)

    def forward(self, x, edge_index):
        return self.conv2(F.relu(self.conv1(x, edge_index)), edge_index)


class Encoder1Layer(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = GCNConv(in_ch, out_ch)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


def build_data(G_anom, variant: str):
    if variant == "no_features":
        # Pas d'attributs informatifs (constante) : teste l'utilite de lat/lon/pop
        data = from_networkx(G_anom)
        data.x = torch.ones((data.num_nodes, 1), dtype=torch.float)
        data.nx_node_ids = torch.tensor(list(G_anom.nodes()), dtype=torch.long)
        return data

    data = from_networkx(G_anom, group_node_attrs=list(NODE_FEATURES))
    data.x = data.x.float()
    data.nx_node_ids = torch.tensor(list(G_anom.nodes()), dtype=torch.long)
    if variant != "no_norm":
        data.x = normalize_features(data.x)
    return data


def build_model(variant: str, in_channels: int):
    latent = 8 if variant == "small_latent" else LATENT_DIM
    if variant == "one_layer":
        enc = Encoder1Layer(in_channels, latent)
    else:
        enc = Encoder2Layer(in_channels, HIDDEN_CHANNELS, latent)
    return GAE(enc)


def run_one(variant: str, seed: int, G, device):
    set_seed(seed)
    G_anom, anomaly_edges = inject_false_edges(G, ANOMALY_RATE, seed)
    data = build_data(G_anom, variant)
    train_data, _, _ = make_link_split(data)
    train_data = train_data.to(device)
    data_ei = data.edge_index.to(device)

    model = build_model(variant, data.num_node_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
        loss = model.recon_loss(z, train_data.pos_edge_label_index)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        z = model.encode(train_data.x, train_data.edge_index)
        rows = compute_edge_anomaly_scores(model, z, data_ei)

    nx_ids = [int(x) for x in data.nx_node_ids.tolist()]
    nx_to_pyg = {nx_id: i for i, nx_id in enumerate(nx_ids)}
    anomaly_pyg = {
        (min(nx_to_pyg[u], nx_to_pyg[v]), max(nx_to_pyg[u], nx_to_pyg[v]))
        for u, v in anomaly_edges
    }
    metrics = evaluate_anomaly_ranking(rows, anomaly_pyg, k=PRECISION_AT_K)
    return {
        "model": variant,
        "seed": seed,
        "anomaly_rate": ANOMALY_RATE,
        "auc": metrics["auc"],
        "average_precision": metrics["average_precision"],
        "precision_at_10": metrics["precision_at_k"],
    }


def main():
    device = get_device()
    ensure_result_dirs()
    G = load_airport_graph()
    variants = ["full", "one_layer", "no_norm", "no_features", "small_latent"]
    results = []
    total = len(variants) * len(SEEDS)
    done = 0
    for variant in variants:
        for seed in SEEDS:
            done += 1
            print(f"[{done}/{total}] {variant} seed={seed}")
            row = run_one(variant, seed, G, device)
            results.append(row)
            print(
                f"  AUC={row['auc']:.4f} AP={row['average_precision']:.4f} "
                f"P@10={row['precision_at_10']:.4f}"
            )

    out = RESULTS_CSV_DIR / "ablation_gae.csv"
    write_csv(
        out,
        [
            {
                **r,
                "auc": f"{r['auc']:.6f}",
                "average_precision": f"{r['average_precision']:.6f}",
                "precision_at_10": f"{r['precision_at_10']:.6f}",
            }
            for r in results
        ],
        ["model", "seed", "anomaly_rate", "auc", "average_precision", "precision_at_10"],
    )
    summary = summarize_results(results)
    print("\nAblation summary (mean+/-std), anomaly rate 10%:")
    for s in summary:
        print(
            f"{s['model']:12s} AUC={s['auc_mean']:.3f}+/-{s['auc_std']:.3f} "
            f"AP={s['ap_mean']:.3f}+/-{s['ap_std']:.3f}"
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
