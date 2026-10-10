"""
Calcule les Top 10 liens suspects (graphe original, sans anomalies injectees).

Protocole :
  - RandomLinkSplit AVANT entrainement
  - encode avec train_data.edge_index uniquement
  - score les aretes message-passing train (aretes observeés pour l'encodeur)
  - anomaly_score = 1 - p_ij
  - export CSV Top 10 pour GAE et VGAE

Usage :
    python experiments/run_top10_suspicious.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    EPOCHS,
    HIDDEN_CHANNELS,
    LATENT_DIM,
    LEARNING_RATE,
    RESULTS_CSV_DIR,
)
from src.data import load_and_split
from src.gae import GAEModel, build_gae_optimizer, train_gae_epoch
from src.io_results import ensure_result_dirs, write_csv
from src.link_anomaly import compute_edge_anomaly_scores, enrich_with_node_attrs
from src.utils import get_device, set_seed
from src.vgae import VGAEModel, build_vgae_optimizer, train_vgae_epoch


TOP10_FIELDS = [
    "rank",
    "source",
    "target",
    "source_city",
    "source_country",
    "target_city",
    "target_country",
    "reconstruction_probability",
    "anomaly_score",
]


def _train_gae(train_data, in_channels, device):
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


def _train_vgae(train_data, in_channels, device):
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


def _export_top10(model, train_data, G, out_path: Path, model_name: str) -> None:
    model.eval()
    with torch.no_grad():
        z = model.encode(train_data.x, train_data.edge_index)
        rows = compute_edge_anomaly_scores(model, z, train_data.edge_index)
    nx_ids = train_data.nx_node_ids if hasattr(train_data, "nx_node_ids") else None
    # RandomLinkSplit conserve les attributs du Data d'origine
    rows = enrich_with_node_attrs(rows, G, nx_node_ids=nx_ids)[:10]
    export = []
    print(f"\n[{model_name}] Top 10 suspicious edges")
    print(
        f"{'Rank':>4} | {'Source':>6} | {'Target':>6} | "
        f"{'Prob':>6} | {'Anomaly':>7} | Cities"
    )
    for i, r in enumerate(rows, start=1):
        export.append(
            {
                "rank": i,
                "source": r["source"],
                "target": r["target"],
                "source_city": r["source_city"],
                "source_country": r["source_country"],
                "target_city": r["target_city"],
                "target_country": r["target_country"],
                "reconstruction_probability": f"{r['reconstruction_probability']:.6f}",
                "anomaly_score": f"{r['anomaly_score']:.6f}",
            }
        )
        print(
            f"{i:4d} | {r['source']:6d} | {r['target']:6d} | "
            f"{r['reconstruction_probability']:6.3f} | {r['anomaly_score']:7.3f} | "
            f"{r['source_city']} ({r['source_country']}) - "
            f"{r['target_city']} ({r['target_country']})"
        )
    write_csv(out_path, export, TOP10_FIELDS)
    print(f"-> {out_path}")


def main(seed: int = 42) -> None:
    set_seed(seed)
    device = get_device()
    ensure_result_dirs()

    print("=" * 60)
    print("Top 10 suspicious edges (clean graph)")
    print("=" * 60)
    print(f"seed={seed}  epochs={EPOCHS}  device={device}")

    G, data, train_data, _, _ = load_and_split(seed=seed)
    train_data = train_data.to(device)
    in_channels = data.num_node_features

    print("\nTraining GAE...")
    gae = _train_gae(train_data, in_channels, device)
    _export_top10(
        gae,
        train_data,
        G,
        RESULTS_CSV_DIR / "gae_top10_suspicious_edges.csv",
        "GAE",
    )

    print("\nTraining VGAE...")
    vgae = _train_vgae(train_data, in_channels, device)
    _export_top10(
        vgae,
        train_data,
        G,
        RESULTS_CSV_DIR / "vgae_top10_suspicious_edges.csv",
        "VGAE",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
