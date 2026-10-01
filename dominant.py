import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from pathlib import Path
import networkx as nx
from torch_geometric.utils import from_networkx, to_undirected
from sklearn.preprocessing import LabelEncoder

class Encoder(nn.Module):
    def __init__(self, in_dim, l1=32, l2=16, out=8):
        super().__init__()
        self.conv1 = GCNConv(in_dim, l1)
        self.conv2 = GCNConv(l1, l2)
        self.conv3 = GCNConv(l2, out)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        z = self.conv3(h, edge_index)
        return z

class AttributeDecoder(nn.Module):
    def __init__(self, embed_dim, out_dim):
        super().__init__()
        self.conv = GCNConv(embed_dim, out_dim)

    def forward(self, z, edge_index):
        return self.conv(z, edge_index)

def structure_decoder(z):
    return z @ z.t()

class Dominant(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.encoder = Encoder(in_dim)
        self.attr_decoder = AttributeDecoder(embed_dim=8, out_dim=in_dim)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        s_logits = structure_decoder(z)
        x_hat = self.attr_decoder(z, edge_index)
        return s_logits, x_hat

def loss_fn(A, X, s_logits, X_hat, alpha, pos_weight , norm):
    struct_err = norm * F.binary_cross_entropy_with_logits(s_logits, A, pos_weight=pos_weight)
    attr_err = F.mse_loss(X_hat, X)
    return (1 - alpha) * struct_err + alpha * attr_err

def run(data):

    label_encoder = LabelEncoder()
    data.country = label_encoder.fit_transform(data.country)
    data.x[:, 2] = torch.log1p(data.x[:, 2])
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
    data.edge_index = to_undirected(data.edge_index)

    num_nodes = data.num_nodes
    num_edges = data.num_edges
    pos_weight = torch.tensor((num_nodes * num_nodes - num_edges) / num_edges)
    norm = num_nodes * num_nodes / float((num_nodes * num_nodes - num_edges) * 2)

    
    model = Dominant(in_dim=data.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    alpha = 0.6
    n_epochs = 300

    A = torch.zeros(num_nodes, num_nodes)
    A[data.edge_index[0], data.edge_index[1]] = 1.0

    attr_base = F.mse_loss(torch.zeros_like(data.x), data.x)
    print(f"Baseline attr (prédire la moyenne) : {attr_base:.4f}")
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        s_logits, X_hat = model(data.x, data.edge_index)
        loss = loss_fn(A, data.x, s_logits, X_hat, alpha, pos_weight, norm)
        loss.backward()
        optimizer.step()


        if epoch % 50 == 0 or epoch == 1:
            struct_err = norm * F.binary_cross_entropy_with_logits(s_logits, A, pos_weight=pos_weight)
            attr_err = F.mse_loss(X_hat, data.x)
            print(f"epoch {epoch:4d}  total={loss.item():.4f}  "
            f"struct={struct_err.item():.4f}  attr={attr_err:.4f}")

    model.eval()
    with torch.no_grad():
        model.eval()
        s_logits, X_hat = model(data.x, data.edge_index)
        attr_score = ((X_hat - data.x) ** 2).mean(dim=1)
        struct_score = ((s_logits - A) ** 2).mean(dim=1)
        anomaly_score = (1 - alpha) * struct_score + alpha * attr_score
    return anomaly_score

DATA_PATH = Path(__file__).parent / "data" / "airportsAndCoordAndPop.graphml"

G = nx.read_graphml(DATA_PATH, node_type=int)

data = from_networkx(
    G,
    group_node_attrs=["lat", "lon", "population"],
)

anomaly_score = run(data)

names = [G.nodes[n].get("city_name", str(n)) for n in G.nodes()]

top = anomaly_score.topk(10)
print("\nTop 10 anomalies (score, aéroport):")
for score, idx in zip(top.values, top.indices):
    print(f"  {score:.4f}  {names[idx]}")