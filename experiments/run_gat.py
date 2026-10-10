"""
GAT : estimation de la population et nœuds suspects.

Pour chaque graine :
  1. découpage aléatoire des nœuds (80 % train, 20 % test)
  2. entraînement du GAT sur log(1 + population) des nœuds train
  3. MAE / RMSE / R² (échelle log) sur train et test
Puis, pour la première graine, export des 20 nœuds test dont la population
observée s'écarte le plus de la prédiction.

Sorties :
  results/csv/gat_regression.csv
  results/csv/gat_top20_suspicious_nodes.csv

Usage :
    python experiments/run_gat.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.config import RESULTS_CSV_DIR, SEEDS
from src.data import load_airport_graph
from src.gat import (
    GAT_EPOCHS,
    GAT_HEADS,
    GAT_HIDDEN_CHANNELS,
    graph_to_pyg_population,
    random_node_split,
    regression_metrics,
    train_gat,
)
from src.io_results import ensure_result_dirs, write_csv
from src.utils import set_seed


REGRESSION_FIELDS = [
    "seed",
    "train_mae",
    "train_rmse",
    "train_r2",
    "test_mae",
    "test_rmse",
    "test_r2",
]

TOP20_FIELDS = [
    "rank",
    "node",
    "city",
    "country",
    "degree",
    "observed_population",
    "predicted_population",
    "log_error",
]


def run_one(seed: int, data) -> tuple[dict, dict, torch.Tensor]:
    set_seed(seed)
    train_mask, test_mask = random_node_split(data.num_nodes)
    result = train_gat(data, train_mask)

    y, p = result["target_log"], result["predicted_log"]
    train_m = regression_metrics(y[train_mask], p[train_mask])
    test_m = regression_metrics(y[test_mask], p[test_mask])
    row = {
        "seed": seed,
        **{f"train_{k}": v for k, v in train_m.items()},
        **{f"test_{k}": v for k, v in test_m.items()},
    }
    return row, result, test_mask


def top_suspicious(G, data, result, test_mask, k: int = 20) -> list[dict]:
    """Les k nœuds test avec la plus grande erreur en log."""
    errors = result["log_error"].clone()
    errors[~test_mask] = -1.0
    order = errors.argsort(descending=True)[:k]

    rows = []
    for rank, idx in enumerate(order.tolist(), start=1):
        node = int(data.nx_node_ids[idx])
        attrs = G.nodes[node]
        rows.append(
            {
                "rank": rank,
                "node": node,
                "city": attrs.get("city_name", ""),
                "country": attrs.get("country", ""),
                "degree": G.degree(node),
                "observed_population": int(data.population_target[idx]),
                "predicted_population": int(torch.expm1(result["predicted_log"][idx])),
                "log_error": f"{result['log_error'][idx].item():.4f}",
            }
        )
    return rows


def main() -> None:
    ensure_result_dirs()

    print("=" * 60)
    print("GAT : estimation de la population (lat, lon, log(1 + degré))")
    print("=" * 60)
    print(f"seeds={SEEDS}  epochs={GAT_EPOCHS}  hidden={GAT_HIDDEN_CHANNELS}  heads={GAT_HEADS}")

    G = load_airport_graph()
    data = graph_to_pyg_population(G)

    results = []
    top20 = None
    for i, seed in enumerate(SEEDS, start=1):
        print(f"\n[{i}/{len(SEEDS)}] seed={seed}")
        row, result, test_mask = run_one(seed, data)
        results.append(row)
        print(
            f"  train MAE={row['train_mae']:.4f} R2={row['train_r2']:.4f} | "
            f"test MAE={row['test_mae']:.4f} RMSE={row['test_rmse']:.4f} R2={row['test_r2']:.4f}"
        )
        if top20 is None:
            top20 = top_suspicious(G, data, result, test_mask)

    reg_path = RESULTS_CSV_DIR / "gat_regression.csv"
    write_csv(
        reg_path,
        [{k: (f"{v:.6f}" if k != "seed" else v) for k, v in r.items()} for r in results],
        REGRESSION_FIELDS,
    )

    print("\n" + "=" * 60)
    print("Résumé test (moyenne +/- écart-type, échelle log)")
    print("=" * 60)
    for metric in ("test_mae", "test_rmse", "test_r2"):
        values = torch.tensor([r[metric] for r in results])
        print(f"{metric:10s} {values.mean():.3f} +/- {values.std():.3f}")

    print(f"\n[GAT] Top 20 nœuds suspects (seed={SEEDS[0]})")
    for r in top20:
        print(
            f"{r['rank']:2d}. {r['city']} ({r['country']}) | deg={r['degree']} | "
            f"observée={r['observed_population']:,} | prédite={r['predicted_population']:,}"
        )
    top_path = RESULTS_CSV_DIR / "gat_top20_suspicious_nodes.csv"
    write_csv(top_path, top20, TOP20_FIELDS)

    print(f"\nCSV régression : {reg_path}")
    print(f"CSV top 20     : {top_path}")


if __name__ == "__main__":
    main()
