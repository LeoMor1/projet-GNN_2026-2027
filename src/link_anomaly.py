"""Scores d'anomalie de liens et injection de fausses arêtes.

Pour un taux r, on ajoute r * |E| arêtes entre des paires de nœuds non reliées,
tirées uniformément, sur une copie du graphe (avant RandomLinkSplit).
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import torch


def undirected_edge_list(edge_index: torch.Tensor) -> torch.Tensor:
    """Retourne les aretes uniques (i < j) a partir d'un edge_index bi-directionnel."""
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    unique = sorted({(min(i, j), max(i, j)) for i, j in zip(src, dst) if i != j})
    if not unique:
        return torch.empty((2, 0), dtype=torch.long)
    rows = torch.tensor(unique, dtype=torch.long).t().contiguous()
    return rows


@torch.no_grad()
def compute_edge_anomaly_scores(
    model,
    z: torch.Tensor,
    edge_index: torch.Tensor,
) -> list[dict[str, Any]]:
    """
    Pour chaque arete observee (i, j) :
      p_ij = decoder(z_i, z_j)
      anomaly_score = 1 - p_ij

    Retourne une liste triee par anomaly_score DESC.
    """
    edges = undirected_edge_list(edge_index)
    if edges.numel() == 0:
        return []

    device = z.device
    edges = edges.to(device)
    # decode attend edge_index (2, E)
    probs = model.decode(z, edges).view(-1)
    scores = 1.0 - probs

    rows: list[dict[str, Any]] = []
    for k in range(edges.size(1)):
        rows.append(
            {
                "source": int(edges[0, k].item()),
                "target": int(edges[1, k].item()),
                "reconstruction_probability": float(probs[k].item()),
                "anomaly_score": float(scores[k].item()),
            }
        )

    rows.sort(key=lambda r: r["anomaly_score"], reverse=True)
    return rows


def enrich_with_node_attrs(
    rows: list[dict[str, Any]],
    G: nx.Graph,
    nx_node_ids: list[int] | torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    """
    Ajoute city_name / country depuis le graphe NetworkX original.

    Si nx_node_ids est fourni, source/target sont des indices PyG (0..N-1)
    et sont convertis vers les IDs NetworkX (qui peuvent avoir des trous).
    """
    if nx_node_ids is not None and not isinstance(nx_node_ids, list):
        nx_node_ids = [int(x) for x in nx_node_ids.tolist()]

    enriched = []
    for r in rows:
        s_idx, t_idx = r["source"], r["target"]
        s = nx_node_ids[s_idx] if nx_node_ids is not None else s_idx
        t = nx_node_ids[t_idx] if nx_node_ids is not None else t_idx
        s_data = G.nodes[s]
        t_data = G.nodes[t]
        enriched.append(
            {
                **r,
                "source": int(s),
                "target": int(t),
                "source_city": s_data.get("city_name", ""),
                "source_country": s_data.get("country", ""),
                "target_city": t_data.get("city_name", ""),
                "target_country": t_data.get("country", ""),
            }
        )
    return enriched


def inject_false_edges(
    G: nx.Graph,
    anomaly_rate: float,
    seed: int,
) -> tuple[nx.Graph, set[tuple[int, int]]]:
    """
    Copie G et ajoute des fausses aretes (non-edges).

    Nombre d'anomalies = round(anomaly_rate * |E_original|).
    Retourne (G_copie, set_anomalies) avec aretes (min, max).
    """
    if anomaly_rate < 0:
        raise ValueError("anomaly_rate must be >= 0")

    G_copy = G.copy()
    nodes = list(G_copy.nodes())
    n_edges = G_copy.number_of_edges()
    n_anom = int(round(anomaly_rate * n_edges))
    if n_anom == 0:
        return G_copy, set()

    rng = torch.Generator().manual_seed(seed)
    existing = {(min(u, v), max(u, v)) for u, v in G_copy.edges()}
    anomalies: set[tuple[int, int]] = set()

    # Echantillonnage rejection sampling des non-edges
    max_trials = n_anom * 50 + 1000
    trials = 0
    n_nodes = len(nodes)
    while len(anomalies) < n_anom and trials < max_trials:
        trials += 1
        i = int(torch.randint(0, n_nodes, (1,), generator=rng).item())
        j = int(torch.randint(0, n_nodes, (1,), generator=rng).item())
        if i == j:
            continue
        u, v = nodes[i], nodes[j]
        e = (min(u, v), max(u, v))
        if e in existing or e in anomalies:
            continue
        anomalies.add(e)
        G_copy.add_edge(u, v)

    if len(anomalies) < n_anom:
        raise RuntimeError(
            f"Could only inject {len(anomalies)}/{n_anom} anomalous edges "
            f"(rate={anomaly_rate})."
        )

    return G_copy, anomalies
