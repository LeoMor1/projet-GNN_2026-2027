import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Encoder(nn.Module):
    def __init__(self, in_dim, l1, l2, out):
        super().__init__()
        self.conv1 = GCNConv(in_dim, l1)
        self.conv2 = GCNConv(l1, l2)
        self.conv3 = GCNConv(l2, out)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        z = self.conv3(x, edge_index)
        return z

class AttributeDecoder(nn.Module):
    def __init__(self, embed_dim, out_dim):
        super().__init__()
        self.conv = GCNConv(embed_dim, out_dim)

    def forward(self, z, edge_index):
        return F.relu(self.conv(z, edge_index))

def structure_decoder(z):
    return z @ z.t()
