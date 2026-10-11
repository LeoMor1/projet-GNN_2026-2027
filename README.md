# Projet GNN 2026–2027 — Classification du pays des aéroports

## Question de recherche

**Peut-on prédire le pays d’un aéroport** à partir de :

- sa position géographique (`lat`, `lon`),
- la population de sa ville (`population`),
- et la structure du réseau aérien (voisins connectés),

en utilisant un **GCN** (Graph Convolutional Network) simple ?

Tâche : **classification multi-classes** (un pays par nœud).  
Métrique : **accuracy** (précision) sur un ensemble de test.

## Données

Fichier : `data/airportsAndCoordAndPop.graphml`

| Attribut nœud | Rôle |
|---------------|------|
| `lat`, `lon`, `population` | Features d’entrée (`data.x`) |
| `country` | Label à prédire (`data.y`) |
| `city_name` | Informatif (non utilisé pour l’entraînement) |

Les arêtes représentent les liaisons aériennes entre aéroports.

Pour le GAT c'est un peu différent : la population n'est pas une entrée, c'est ce qu'on cherche à prédire (à partir de `lat`, `lon` et du degré). Detail important a preciser, la moitier des valeurs sont mises a 10 000 probablement une valeurs par defaut et a prendre donc avec des pincettes.

## Installation

- Python ≥ 3.12
- [conda](https://docs.conda.io/) (recommandé sous Windows) **ou** pip / [uv](https://docs.astral.sh/uv/)

## Installation (reproductibilité)

### Avec conda (recommandé)

```bash
conda create -n gnn312 python=3.12 -y
conda activate gnn312
conda install -y numpy networkx scikit-learn pytorch cpuonly -c pytorch -c conda-forge
pip install torch-geometric
```

### Avec pip

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
# source .venv/bin/activate

pip install -e .
```

### Avec uv

```bash
uv sync
```

## Exécution

```bash
conda activate gnn312
python main.py
```

| Commande | Contenu | Sortie (`results/csv/`) |
|---|---|---|
| `python experiments/run_comparison.py` | GAE contre VGAE, 4 taux × 5 graines | `gae_vs_vgae.csv`, `gae_vs_vgae_summary.csv`, figures dans `results/figures/` |
| `python experiments/run_baselines.py` | Distance, Jaccard, Adamic-Adar (sans PyTorch) | `baselines_liens.csv` |
| `python experiments/run_ablation.py` | Ablation du GAE à 10 % d'anomalies | `ablation_gae.csv` |
| `python experiments/run_top10_suspicious.py` | 10 liens les plus suspects du graphe d'origine | `gae_top10_suspicious_edges.csv`, `vgae_top10_suspicious_edges.csv` |
| `python experiments/run_gat.py` | GAT qui prédit la population, 5 graines + les 20 nœuds les plus suspects | `gat_regression.csv`, `gat_top20_suspicious_nodes.csv` |

Le script (partie **avant** le GCN) :

1. charge le graphe GraphML avec NetworkX ;
2. le convertit en objet PyTorch Geometric (`from_networkx`) ;
3. encode les pays avec `LabelEncoder` ;
4. crée des masques train (80 %) / test (20 %).

Il affiche la taille du graphe, la forme de `data.x` / `data.edge_index`, le nombre de pays, et les tailles train/test.

## Structure du dépôt

```
projet-GNN_2026-2027/
├── data/
│   └── airportsAndCoordAndPop.graphml
├── main.py                  # chargement + préparation des données
├── pyproject.toml
├── README.md
└── rapport/                 # rapport LaTeX
```

Protocole GAT : on coupe les nœuds au hasard (80 % train, 20 % test) et le GAT apprend à prédire `log(1 + population)` à partir de `lat`, `lon` et `log(1 + degré)` (standardisés). La perte (MSE) est calculée seulement sur les nœuds train. Pour chaque nœud, le score d'anomalie est l'écart entre la population prédite et la population observée, en log. On regarde MAE, RMSE et R² (en log).

Paramètres GAT : 2 couches GATConv (3 → 64 × 4 têtes → 64) avec ELU, dropout 0,2, Adam avec un taux de 0,005 et un weight decay de 5e-4, 200 époques, mêmes graines que plus haut.

## Remarques

- `run_baselines.py` tire les liens injectés avec NumPy, `run_comparison.py` avec PyTorch : même loi, pas les mêmes liens.
- `src/metrics.py` calcule AUC et AP sans sklearn (celui-ci pose problème sur certains postes Windows) et traite les égalités de scores par blocs.
- Les paramètres du GAT sont en haut de `src/gat.py` et pas dans `src/config.py`, pour ne pas toucher au fichier de config des autres parties.