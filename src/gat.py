"""GAT : régression de log(1 + population) à partir de lat, lon et log(1 + degré).

GATConv(in -> 64 x 4 têtes) -> ELU -> GATConv(256 -> 64, 1 tête) -> ELU -> Linear(64 -> 1)

La population n'est jamais une entrée : seule la perte sur les nœuds d'entraînement
l'utilise. Score d'anomalie d'un nœud : |log(1 + pop prédite) - log(1 + pop observée)|.
"""

from __future__ import annotations

import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import degree, from_networkx

from src.data import normalize_features

# Paramètres du GAT (propres à cette partie, src/config.py n'est pas modifié)
GAT_FEATURES = ["lat", "lon"]
GAT_HIDDEN_CHANNELS = 64
GAT_HEADS = 4
GAT_DROPOUT = 0.2
GAT_LEARNING_RATE = 0.005
GAT_WEIGHT_DECAY = 5e-4
GAT_EPOCHS = 200
GAT_TRAIN_RATIO = 0.80


class GATModel(torch.nn.Module):
    """Régression d'une valeur par nœud (log-population standardisée).

    Entrées : x [N, in_channels], edge_index [2, E].
    Sortie  : [N].
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=GAT_HIDDEN_CHANNELS,
        heads=GAT_HEADS,
        dropout=GAT_DROPOUT,
    ):
        super().__init__()
        self.dropout = dropout
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # concat=False : moyenne des têtes, la sortie reste en hidden_channels
        self.gat2 = GATConv(
            hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout
        )
        self.regressor = torch.nn.Linear(hidden_channels, 1)

    def embed(self, x, edge_index):
        """Représentation de chaque nœud avant la couche de régression."""
        h = F.elu(self.gat1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return F.elu(self.gat2(h, edge_index))

    def forward(self, x, edge_index):
        return self.regressor(self.embed(x, edge_index)).squeeze(-1)


def graph_to_pyg_population(G: nx.Graph) -> Data:
    """NetworkX -> PyG avec x = (lat, lon) et population_target séparée.

    `data.nx_node_ids[i]` donne l'identifiant NetworkX du nœud PyG `i`.
    """
    nx_node_ids = list(G.nodes())
    data = from_networkx(G, group_node_attrs=GAT_FEATURES)
    data.x = data.x.float()
    data.population_target = torch.tensor(
        [float(G.nodes[n]["population"]) for n in nx_node_ids], dtype=torch.float
    )
    data.nx_node_ids = torch.tensor(nx_node_ids, dtype=torch.long)
    return data


def build_features(data: Data, use_degree: bool = True) -> torch.Tensor:
    """lat, lon (+ log(1 + degré)), standardisés.

    Le degré est ajouté explicitement : l'attention du GAT fait une moyenne
    pondérée des voisins et ne voit donc pas leur nombre.
    """
    x = data.x.float()
    if use_degree:
        deg = degree(data.edge_index[0], data.num_nodes)
        x = torch.cat([x, torch.log1p(deg).unsqueeze(1)], dim=1)
    return normalize_features(x)


def random_node_split(num_nodes: int, train_ratio: float = GAT_TRAIN_RATIO):
    """Masques (train, test) tirés au hasard (graine fixée par l'appelant)."""
    train_mask = torch.rand(num_nodes) < train_ratio
    return train_mask, ~train_mask


def train_gat(
    data: Data,
    train_mask: torch.Tensor,
    use_degree: bool = True,
    hidden_channels: int = GAT_HIDDEN_CHANNELS,
    heads: int = GAT_HEADS,
    dropout: float = GAT_DROPOUT,
    learning_rate: float = GAT_LEARNING_RATE,
    weight_decay: float = GAT_WEIGHT_DECAY,
    epochs: int = GAT_EPOCHS,
    verbose: bool = False,
) -> dict:
    """Entraîne un GAT sur les nœuds de `train_mask` (graine fixée par l'appelant).

    `data` doit contenir x (lat, lon), edge_index et population_target.

    Retourne un dict :
      model           GAT entraîné
      predicted_log   log(1 + population) prédit, pour tous les nœuds [N]
      target_log      log(1 + population) observé [N]
      log_error       |predicted_log - target_log|, score d'anomalie [N]
    """
    x = build_features(data, use_degree)
    target_log = torch.log1p(data.population_target.float())

    # Cible standardisée avec les statistiques des nœuds d'entraînement seulement
    mean = target_log[train_mask].mean()
    std = target_log[train_mask].std().clamp_min(1e-8)
    y = (target_log - mean) / std

    model = GATModel(x.size(1), hidden_channels, heads, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = F.mse_loss(model(x, data.edge_index)[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        if verbose and epoch % 20 == 0:
            print(f"epoch {epoch:4d}  loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        predicted_log = model(x, data.edge_index) * std + mean

    return {
        "model": model,
        "predicted_log": predicted_log,
        "target_log": target_log,
        "log_error": (predicted_log - target_log).abs(),
    }


def regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
    """MAE, RMSE et R² (échelle log)."""
    mae = (y_pred - y_true).abs().mean().item()
    rmse = ((y_pred - y_true) ** 2).mean().sqrt().item()
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return {"mae": mae, "rmse": rmse, "r2": (1 - ss_res / ss_tot).item()}
