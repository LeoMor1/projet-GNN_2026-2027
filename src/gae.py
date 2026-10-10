"""Graph Autoencoder : GCN(3->64) -> ReLU -> GCN(64->32), décodeur produit scalaire."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GAE, GCNConv

from src.config import HIDDEN_CHANNELS, LATENT_DIM, LEARNING_RATE


class GAEEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=HIDDEN_CHANNELS, out_channels=LATENT_DIM):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv2(F.relu(self.conv1(x, edge_index)), edge_index)


class GAEModel(torch.nn.Module):
    """Enveloppe de `torch_geometric.nn.GAE`."""

    def __init__(self, in_channels, hidden_channels=HIDDEN_CHANNELS, out_channels=LATENT_DIM):
        super().__init__()
        self.gae = GAE(GAEEncoder(in_channels, hidden_channels, out_channels))

    def encode(self, x, edge_index):
        return self.gae.encode(x, edge_index)

    def decode(self, z, edge_index):
        return self.gae.decode(z, edge_index)

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        return self.gae.recon_loss(z, pos_edge_index, neg_edge_index)


def build_gae_optimizer(model, lr: float = LEARNING_RATE):
    return torch.optim.Adam(model.parameters(), lr=lr)


def train_gae_epoch(model, optimizer, x, edge_index, pos_edge_label_index) -> float:
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, edge_index)
    loss = model.recon_loss(z, pos_edge_label_index)
    loss.backward()
    optimizer.step()
    return float(loss.item())
