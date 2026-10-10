"""Chargement du graphe d'aéroports et découpage des liens."""

from __future__ import annotations

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import from_networkx

from src.config import (
    ADD_NEGATIVE_TRAIN_SAMPLES,
    DATA_PATH,
    IS_UNDIRECTED,
    NODE_FEATURES,
    NUM_TEST,
    NUM_VAL,
    SPLIT_LABELS,
)


def load_airport_graph(data_path=DATA_PATH) -> nx.Graph:
    """Charge le GraphML (le fichier n'est jamais modifié)."""
    return nx.read_graphml(data_path, node_type=int)


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    """Standardisation colonne par colonne.

    Nécessaire : la population (~1e7) écrase lat/lon et fait saturer le décodeur.
    """
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-8)
    return (x - mean) / std


def graph_to_pyg(G: nx.Graph, normalize: bool = True) -> Data:
    """NetworkX -> PyG avec lat, lon, population.

    Les identifiants NetworkX ont des trous : `data.nx_node_ids[i]` donne
    l'identifiant d'origine du nœud PyG `i`.
    """
    nx_node_ids = list(G.nodes())
    data = from_networkx(G, group_node_attrs=list(NODE_FEATURES))
    data.x = data.x.float()
    if normalize:
        data.x = normalize_features(data.x)
    data.nx_node_ids = torch.tensor(nx_node_ids, dtype=torch.long)
    return data


def make_link_split(data: Data) -> tuple[Data, Data, Data]:
    """RandomLinkSplit : 10 % validation, 20 % test."""
    transform = RandomLinkSplit(
        num_val=NUM_VAL,
        num_test=NUM_TEST,
        is_undirected=IS_UNDIRECTED,
        add_negative_train_samples=ADD_NEGATIVE_TRAIN_SAMPLES,
        split_labels=SPLIT_LABELS,
    )
    return transform(data)


def load_and_split(seed: int | None = None):
    """Graphe d'origine -> (G, data, train, val, test)."""
    if seed is not None:
        torch.manual_seed(seed)
    G = load_airport_graph()
    data = graph_to_pyg(G)
    train_data, val_data, test_data = make_link_split(data)
    return G, data, train_data, val_data, test_data
