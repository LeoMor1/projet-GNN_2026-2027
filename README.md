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

## Prérequis

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




