"""VGAE : GCN(3->64) -> ReLU -> deux GCN (mu, log sigma), perte = reconstruction + KL/N."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE

from src.config import HIDDEN_CHANNELS, LATENT_DIM, LEARNING_RATE


class VGAEEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=HIDDEN_CHANNELS, out_channels=LATENT_DIM):
        super().__init__()
        self.conv_hidden = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        h = F.relu(self.conv_hidden(x, edge_index))
        return self.conv_mu(h, edge_index), self.conv_logstd(h, edge_index)


class VGAEModel(torch.nn.Module):
    """Enveloppe de `torch_geometric.nn.VGAE`.

    En mode eval, `encode` renvoie la moyenne mu (pas d'échantillonnage).
    """

    def __init__(self, in_channels, hidden_channels=HIDDEN_CHANNELS, out_channels=LATENT_DIM):
        super().__init__()
        self.vgae = VGAE(VGAEEncoder(in_channels, hidden_channels, out_channels))

    def encode(self, x, edge_index):
        return self.vgae.encode(x, edge_index)

    def decode(self, z, edge_index):
        return self.vgae.decode(z, edge_index)

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        return self.vgae.recon_loss(z, pos_edge_index, neg_edge_index)

    def kl_loss(self):
        return self.vgae.kl_loss()


def build_vgae_optimizer(model, lr: float = LEARNING_RATE):
    return torch.optim.Adam(model.parameters(), lr=lr)


def train_vgae_epoch(model, optimizer, x, edge_index, pos_edge_label_index, num_nodes=None) -> float:
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, edge_index)
    n = num_nodes if num_nodes is not None else x.size(0)
    loss = model.recon_loss(z, pos_edge_label_index) + (1.0 / n) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss.item())
