"""
Comparaison GAE vs VGAE sur anomalies de liens artificielles.

Pour chaque modele x anomaly_rate x seed :
  1. copie du graphe + injection de fausses aretes (AVANT split)
  2. RandomLinkSplit
  3. entrainement
  4. scores d'anomalie sur les aretes observees (message-passing train)
  5. AUC / AP / Precision@10 vs ground truth injecte

Sorties :
  results/csv/gae_vs_vgae.csv
  results/csv/gae_vs_vgae_summary.csv
  results/figures/gae_vgae_auc.png
  results/figures/gae_vgae_ap.png

Usage :
    python experiments/run_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.config import (
    ANOMALY_RATES,
    EPOCHS,
    HIDDEN_CHANNELS,
    LATENT_DIM,
    LEARNING_RATE,
    PRECISION_AT_K,
    RESULTS_CSV_DIR,
    RESULTS_FIG_DIR,
    SEEDS,
)
from src.data import graph_to_pyg, load_airport_graph, make_link_split
from src.evaluation import evaluate_anomaly_ranking, summarize_results
from src.gae import GAEModel, build_gae_optimizer, train_gae_epoch
from src.io_results import ensure_result_dirs, plot_metric_vs_rate, write_csv
from src.link_anomaly import (
    compute_edge_anomaly_scores,
    inject_false_edges,
)
from src.utils import get_device, set_seed
from src.vgae import VGAEModel, build_vgae_optimizer, train_vgae_epoch


DETAIL_FIELDS = [
    "model",
    "seed",
    "anomaly_rate",
    "auc",
    "average_precision",
    "precision_at_10",
]

SUMMARY_FIELDS = [
    "model",
    "anomaly_rate",
    "auc_mean",
    "auc_std",
    "ap_mean",
    "ap_std",
    "precision10_mean",
    "precision10_std",
]


def _train(model_name: str, train_data, in_channels, device):
    if model_name == "GAE":
        model = GAEModel(in_channels, HIDDEN_CHANNELS, LATENT_DIM).to(device)
        opt = build_gae_optimizer(model, LEARNING_RATE)
        for _ in range(EPOCHS):
            train_gae_epoch(
                model,
                opt,
                train_data.x,
                train_data.edge_index,
                train_data.pos_edge_label_index,
            )
        return model

    if model_name == "VGAE":
        model = VGAEModel(in_channels, HIDDEN_CHANNELS, LATENT_DIM).to(device)
        opt = build_vgae_optimizer(model, LEARNING_RATE)
        for _ in range(EPOCHS):
            train_vgae_epoch(
                model,
                opt,
                train_data.x,
                train_data.edge_index,
                train_data.pos_edge_label_index,
                num_nodes=train_data.num_nodes,
            )
        return model

    raise ValueError(model_name)


def run_one(
    model_name: str,
    anomaly_rate: float,
    seed: int,
    G_original,
    device,
) -> dict:
    set_seed(seed)

    G_anom, anomaly_edges = inject_false_edges(
        G_original,
        anomaly_rate=anomaly_rate,
        seed=seed,
    )
    data = graph_to_pyg(G_anom, normalize=True)
    train_data, val_data, test_data = make_link_split(data)
    train_data = train_data.to(device)

    model = _train(model_name, train_data, data.num_node_features, device)
    model.eval()
    with torch.no_grad():
        z = model.encode(train_data.x, train_data.edge_index)
        # Encode avec train only ; score les aretes du graphe contamine complet
        # (protocole reconstruction / anomaly scoring).
        full_edge_index = data.edge_index.to(device)
        rows = compute_edge_anomaly_scores(model, z, full_edge_index)

    # anomaly_edges utilise les IDs NetworkX ; les rows utilisent les indices PyG
    nx_ids = [int(x) for x in data.nx_node_ids.tolist()]
    nx_to_pyg = {nx_id: i for i, nx_id in enumerate(nx_ids)}
    anomaly_pyg = {
        (min(nx_to_pyg[u], nx_to_pyg[v]), max(nx_to_pyg[u], nx_to_pyg[v]))
        for u, v in anomaly_edges
    }

    metrics = evaluate_anomaly_ranking(
        rows,
        anomaly_pyg,
        k=PRECISION_AT_K,
    )

    return {
        "model": model_name,
        "seed": seed,
        "anomaly_rate": anomaly_rate,
        "auc": metrics["auc"],
        "average_precision": metrics["average_precision"],
        "precision_at_10": metrics["precision_at_k"],
        "n_anomalies": int(metrics["n_anomalies"]),
        "n_edges_scored": int(metrics["n_edges_scored"]),
    }


def main() -> None:
    device = get_device()
    ensure_result_dirs()

    print("=" * 60)
    print("Comparaison GAE vs VGAE (anomalies artificielles)")
    print("=" * 60)
    print(f"device={device}")
    print(f"seeds={SEEDS}")
    print(f"anomaly_rates={ANOMALY_RATES}")
    print(f"epochs={EPOCHS}  hidden={HIDDEN_CHANNELS}  latent={LATENT_DIM}")
    print(
        "Strategie anomalies : AJOUT de fausses aretes (non-edges) "
        "avant RandomLinkSplit."
    )

    G = load_airport_graph()
    print(f"Graphe original : {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    results = []
    total = len(SEEDS) * len(ANOMALY_RATES) * 2
    done = 0

    for model_name in ("GAE", "VGAE"):
        for rate in ANOMALY_RATES:
            for seed in SEEDS:
                done += 1
                print(
                    f"\n[{done}/{total}] {model_name}  rate={rate:.0%}  seed={seed}"
                )
                row = run_one(model_name, rate, seed, G, device)
                results.append(row)
                print(
                    f"  AUC={row['auc']:.4f}  AP={row['average_precision']:.4f}  "
                    f"P@10={row['precision_at_10']:.4f}  "
                    f"(anomalies scored={row['n_anomalies']}/{row['n_edges_scored']})"
                )

    detail_path = RESULTS_CSV_DIR / "gae_vs_vgae.csv"
    write_csv(
        detail_path,
        [
            {
                **r,
                "auc": f"{r['auc']:.6f}",
                "average_precision": f"{r['average_precision']:.6f}",
                "precision_at_10": f"{r['precision_at_10']:.6f}",
            }
            for r in results
        ],
        DETAIL_FIELDS,
    )

    summary = summarize_results(results)
    summary_path = RESULTS_CSV_DIR / "gae_vs_vgae_summary.csv"
    write_csv(
        summary_path,
        [
            {
                "model": s["model"],
                "anomaly_rate": s["anomaly_rate"],
                "auc_mean": f"{s['auc_mean']:.6f}",
                "auc_std": f"{s['auc_std']:.6f}",
                "ap_mean": f"{s['ap_mean']:.6f}",
                "ap_std": f"{s['ap_std']:.6f}",
                "precision10_mean": f"{s['precision10_mean']:.6f}",
                "precision10_std": f"{s['precision10_std']:.6f}",
            }
            for s in summary
        ],
        SUMMARY_FIELDS,
    )

    plot_metric_vs_rate(
        summary,
        "auc_mean",
        "auc_std",
        "ROC-AUC",
        RESULTS_FIG_DIR / "gae_vgae_auc.png",
        "AUC vs anomaly rate (GAE vs VGAE)",
    )
    plot_metric_vs_rate(
        summary,
        "ap_mean",
        "ap_std",
        "Average Precision",
        RESULTS_FIG_DIR / "gae_vgae_ap.png",
        "AP vs anomaly rate (GAE vs VGAE)",
    )

    print("\n" + "=" * 60)
    print("Resume (mean +/- std)")
    print("=" * 60)
    for s in summary:
        print(
            f"{s['model']:4s} | rate={s['anomaly_rate']:.0%} | "
            f"AUC={s['auc_mean']:.3f}+/-{s['auc_std']:.3f} | "
            f"AP={s['ap_mean']:.3f}+/-{s['ap_std']:.3f} | "
            f"P@10={s['precision10_mean']:.3f}+/-{s['precision10_std']:.3f}"
        )

    print(f"\nCSV detail  : {detail_path}")
    print(f"CSV summary : {summary_path}")
    print(f"Figure AUC  : {RESULTS_FIG_DIR / 'gae_vgae_auc.png'}")
    print(f"Figure AP   : {RESULTS_FIG_DIR / 'gae_vgae_ap.png'}")
    print("Comparaison terminee.")


if __name__ == "__main__":
    main()
