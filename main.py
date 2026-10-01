"""
Projet GNN — Classification du pays des aéroports
Basé sur le TP-GNN (Getting started, avant le GCN).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import torch as th
from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import from_networkx

DATA_PATH = Path(__file__).parent / "data" / "airportsAndCoordAndPop.graphml"

# =============================================================================
# (a) Charger le graphe avec NetworkX + plot rapide
# =============================================================================
print("=" * 50)
print("(a) Chargement du graphe")
print("=" * 50)

G = nx.read_graphml(DATA_PATH, node_type=int)

print("Nombre de nœuds:", G.number_of_nodes())
print("Nombre d'arêtes:", G.number_of_edges())
print("Exemple de nœud:", list(G.nodes(data=True))[0])
print("Exemple d'arête:", list(G.edges(data=True))[0])

# Plot du réseau
plt.figure(figsize=(10, 8))
pos = {n: (float(G.nodes[n]["lon"]), float(G.nodes[n]["lat"])) for n in G.nodes()}
nx.draw(
    G,
    pos,
    with_labels=False,
    node_size=10,
    node_color="lightblue",
    edge_color="gray",
    width=0.2,
    alpha=0.7,
)
plt.title("Airport Network")
plt.show()
print("→ Graphique affiché")

# =============================================================================
# (b) Convertir NetworkX → PyTorch Geometric
# =============================================================================
print("\n" + "=" * 50)
print("(b) Conversion NetworkX → PyTorch Geometric")
print("=" * 50)

data = from_networkx(
    G,
    group_node_attrs=["lat", "lon", "population"],
)
data.x = data.x.float()

print("Conversion OK")
print("Features utilisées: lat, lon, population")
print("data.x shape:", data.x.shape)

# =============================================================================
# (c) Vérifier le contenu de l'objet data
# =============================================================================
print("\n" + "=" * 50)
print("(c) Contenu de l'objet data")
print("=" * 50)

print(data)
print("\nNode features (x):")
print(data.x)
print("\nEdge index:")
print(data.edge_index)
print("\nNumber of nodes:", data.num_nodes)
print("Number of edges:", data.num_edges)
print("Number of node features:", data.num_node_features)

# =============================================================================
# (d) Encoder la classe (pays) avec LabelEncoder
# =============================================================================
print("\n" + "=" * 50)
print("(d) Encodage des pays (labels)")
print("=" * 50)

le = LabelEncoder()
countries = [G.nodes[node]["country"] for node in G.nodes()]
y = le.fit_transform(countries)

data.y = th.tensor(y, dtype=th.long)
data.num_classes = len(le.classes_)

print("Pays (exemples):", countries[:5])
print("Encoded (exemples):", y[:5])
print("Nombre de classes (pays):", data.num_classes)
print("data.y shape:", data.y.shape)
print("data.y:", data.y)

# =============================================================================
# (e) Masques train / test + cacher les features des nœuds test
# =============================================================================
print("\n" + "=" * 50)
print("(e) Masques train / test")
print("=" * 50)

num_nodes = data.num_nodes
train_ratio = 0.80

mask = th.rand(num_nodes) < train_ratio
print("masks shape:", mask.shape)
print("masks (True=train):", mask)
print("Nombre train (True):", int(mask.sum()))

data.train_mask = mask
data.test_mask = ~mask

print("Train:", int(data.train_mask.sum()))
print("Test:", int(data.test_mask.sum()))

# Cacher les features des nœuds test (comme au TP)
x_masked = data.x.clone()
x_masked[~mask] = 0
data.x = x_masked

print("\nFeatures originales (5 premières lignes):")
print(data.x[:5])  # déjà masquées ici
print("Features après masquage — nœuds test mis à 0")
print("Exemple features d'un nœud test (si existe):")
test_idx = (~mask).nonzero(as_tuple=True)[0]
if len(test_idx) > 0:
    print("index test:", int(test_idx[0]), "→", data.x[test_idx[0]])

print("\n" + "=" * 50)
print("Chargement OK — prêt pour le GCN")
print("=" * 50)
print(data)
