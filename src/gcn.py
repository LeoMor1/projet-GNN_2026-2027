import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from pathlib import Path
import networkx as nx
from torch_geometric.utils import from_networkx, to_undirected, scatter
from sklearn.preprocessing import LabelEncoder
R_EARTH = 6371.0

def build_features(
        data, 
        use_xyz=True, 
        use_log_pop=True, 
        use_pop_missing=True,
        use_distances=True, 
        standardize=True, 
        use_log_on_pop=True
    ):
    feats = []

    lat = torch.deg2rad(data.x[:, 0])
    lon = torch.deg2rad(data.x[:, 1])

    # On tranforme lon lat en coordonnee sur le globe pour eviter des coordonee qui on des valeur eloigner mais en 
    # realite sont proche
    if use_xyz:
        xyz = torch.stack([
            torch.cos(lat) * torch.cos(lon),
            torch.cos(lat) * torch.sin(lon),
            torch.sin(lat),
        ], dim=1)
        feats.append(xyz)

    pop = data.x[:, 2]

    # 10000 est la valeur standar on ajoute donc une feature pour aider a le model a comprendre si la valuer est
    # basse ou si il manque une donnee. On effectue une transformation pour diminuer le poids des grands ecart
    if use_pop_missing:
            feats.append((pop == 10000).float()[:, None])
    if use_log_pop:
        pop_col = torch.log1p(pop) if use_log_on_pop else pop
        feats.append(pop_col[:, None])

    
    
    if use_distances:
        # distance géographique aux voisins : "ce nœud est-il cohérent
        # avec la position de ses voisins ?"
        x = torch.cos(lat) * torch.cos(lon)
        y = torch.cos(lat) * torch.sin(lon)
        z = torch.sin(lat)
        xyz_all = torch.stack([x, y, z], dim=1)

        src, dst = data.edge_index
        ang = torch.acos((xyz_all[src] * xyz_all[dst]).sum(1).clamp(-1.0, 1.0))
        km = ang * R_EARTH
        n = data.num_nodes

        # On cree les feature distance minimum et moyenne des distance par rapport au voisin du noeud
        # le modele pourra trouver si la distance est coherante avec la position
        d_mean = scatter(km, src, dim=0, dim_size=n, reduce="mean")
        d_min  = scatter(km, src, dim=0, dim_size=n, reduce="min")
        feats.append(torch.log1p(d_mean)[:, None])
        feats.append(torch.log1p(d_min)[:, None])

    x = torch.cat(feats, dim=1)

    
    if standardize:
        mean = x.mean(0)
        std = x.std(0).clamp(1e-6)
        x = (x - mean) / std

    return x


class Encoder(nn.Module):
    def __init__(self, in_dim, l1=64, l2=32, emb=16, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_dim, l1)
        self.conv2 = GCNConv(l1, l2)
        self.conv3 = GCNConv(l2, emb)

    def forward(self, x, edge_index):
        h = F.dropout(F.relu(self.conv1(x, edge_index)), self.dropout, self.training)
        h = F.dropout(F.relu(self.conv2(h, edge_index)), self.dropout, self.training)
        z = self.conv3(h, edge_index)
        return z

class AttributeDecoder(nn.Module):
    def __init__(self, out_dim, embed_dim=16, mid_dim=32, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(embed_dim, mid_dim)
        self.conv2 = GCNConv(mid_dim, out_dim)

    def forward(self, z, edge_index):
        x_hat = F.dropout(F.relu(self.conv1(z, edge_index)), self.dropout, self.training)
        x_hat = self.conv2(x_hat, edge_index)
        return x_hat


class StrucutreDecoder(nn.Module):
    def __init__(self, emb=16):
        super().__init__()
        self.conv = GCNConv(emb, emb)

    def forward(self, z, edge_index):
        z = self.conv(z, edge_index)
        return z @ z.t()

class Dominant(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.encoder = Encoder(in_dim)
        self.attr_decoder = AttributeDecoder(embed_dim=16, out_dim=in_dim)
        self.struct_decoder = StrucutreDecoder(emb=16)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        s_logits = self.struct_decoder(z, edge_index)
        x_hat = self.attr_decoder(z, edge_index)
        return s_logits, x_hat

def loss_fn(A, x, s_logits, x_hat, pos_weight):
    struct_err = F.binary_cross_entropy_with_logits(s_logits, A, pos_weight=pos_weight)
    attr_err = F.mse_loss(x_hat, x)
    loss = struct_err + attr_err
    return loss, struct_err, attr_err

def model_eval(model, x, edge_index, A, pos_weight):
    model.eval()
    with torch.no_grad():
        s_logits, x_hats = model(x, edge_index)

        struct_score = F.binary_cross_entropy_with_logits(s_logits, A, pos_weight=pos_weight, reduction="none").mean(1)
        attr_feat = (x_hats - x) ** 2
        attr_score = attr_feat.mean(1)
        probs = torch.sigmoid(s_logits)
        score = attr_score + struct_score
    return score, struct_score, attr_score, attr_feat, probs

def fit_once(model, x, edge_index, A, n_epochs, lr, seed, pos_weight, verbose=True):

    torch.manual_seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log_every = max(1, n_epochs // 10) 

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        s_logits, x_hats = model(x, edge_index)
        loss, struct_err, attr_err = loss_fn(A, x, s_logits, x_hats, pos_weight)
        loss.backward()
        optimizer.step()

        if verbose and (epoch % log_every == 0 or epoch == n_epochs - 1):
            pct = 100 * (epoch + 1) / n_epochs
            print(f"epoch {epoch + 1:4d}/{n_epochs} ({pct:3.0f} %)  "
                  f"loss={loss.item():.4f}  "
                  f"struct={struct_err.item():.4f}  "
                  f"attr={attr_err.item():.4f}")

def z_score_robust(score):
    med = score.median()
    mad = (score - med).abs().median()
    z_score = (score - med) / mad
    return z_score

def run(data, n_epoch, n_runs = 1):

    edge_index = data.edge_index
    edge_index = to_undirected(edge_index)

    x = data.x
    # x = build_features(data)

    n = x.shape[0]

    A = torch.zeros(n, n)
    A[edge_index[0], edge_index[1]] = 1.0
    n_pos = A.sum()

    pos_weight = (n * n - n_pos) / n_pos

    total_score = total_struct_score = total_attr_score = total_attr_feat = total_probs = 0

    print(f"Baseline attr (prédire la moyenne) : "
          f"{F.mse_loss(torch.zeros_like(x), x).item():.4f}")

    # On fait plusieurs run puis on moyenne les score permet de lisser les erreurs liee a une seed et de faire encore
    # plus ressortir les anomalie trouver a chaque fois
    for i in range(n_runs):
        print(f"run {i + 1}/{n_runs}")

        model = Dominant(in_dim=x.shape[1])

        fit_once(model, x, edge_index, A, n_epoch, 0.005, 1, pos_weight)

        score, struct_score, attr_score, attr_feat, probs = model_eval(model, x, edge_index, A, pos_weight)

        score = z_score_robust(score)
        struct_score = z_score_robust(struct_score)
        attr_score = z_score_robust(attr_score)

        total_score = total_score + score / n_runs
        total_struct_score = total_struct_score + struct_score / n_runs
        total_attr_score = total_attr_score + attr_score / n_runs
        total_attr_feat = total_attr_feat + attr_feat / n_runs
        total_probs = total_probs + probs / n_runs

    return total_score, total_struct_score, total_attr_score, total_attr_feat, total_probs, A

def explain(index, attr_feat, probs, A, names, features_names, nb_attr = 1, nb_voisins = 1):
    print(f"-------\nDetails noeud : {names[index]}")

    ratio = attr_feat[index] / (attr_feat.mean(0) + 1e-8)
    print("Features les plus mal reconstruites (x l'erreur moyenne) :")
    for j in ratio.argsort(descending=True)[:nb_attr]:
        print(f"{features_names[j]:18s} x{ratio[j]:6.1f}")

    nbrs = A[index].nonzero().flatten()
    nbrs = nbrs[nbrs != index]
    if len(nbrs):
        probs_exist = probs[index, nbrs]
        print("Liens existants les moins plausibles :")
        for node in probs_exist.argsort()[:nb_voisins]:
            print(f"{names[index]} -- {names[int(nbrs[node])]:25s} proba={probs_exist[node]:.3f}")
 
    proba_miss = probs[index].clone()
    proba_miss[A[index] > 0] = -1
    proba_miss[index] = -1
    top = proba_miss.topk(nb_voisins)
    print("Liens absents les plus attendus :")
    for proba, node in zip(top.values, top.indices):
        print(f"{names[index]} -- {names[int(node)]:25s} proba={proba:.3f}")

if __name__ == "__main__":
    DATA_PATH = Path(__file__).parent / "data" / "airportsAndCoordAndPop.graphml"
    G = nx.read_graphml(DATA_PATH, node_type=int)
    N_EPOCH = 500
    N_RUNS = 5

    data = from_networkx(
        G,
        group_node_attrs=["lat", "lon", "population"],
    )

    names = [G.nodes[n].get("city_name", str(n)) for n in G.nodes()]

    score, struct_score, attr_score, attr_feat, probs, A = run(data, N_EPOCH, N_RUNS)
    features_names = ["x", "y", "z", "log_pop", "pop_missing", "dist_moy_voisins", "dist_min_voisins"]

    top = score.topk(10)
    print("\nTop 10 anomalies (score, aéroport):")
    for score, idx in zip(top.values, top.indices):
        explain(idx, attr_feat, probs, A, names, features_names, nb_attr=10, nb_voisins=10)