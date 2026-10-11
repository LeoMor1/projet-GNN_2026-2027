# Projet GNN 2026–2027 — Détection d'anomalies dans un réseau d'aéroports

## Question de recherche

Le réseau aérien mondial relie des milliers d'aéroports dont certaines données
(coordonnées, population) peuvent être erronées ou incohérentes. **Peut-on
détecter automatiquement ces anomalies** — des liaisons aériennes suspectes
entre deux aéroports, ou des aéroports dont les attributs contredisent la
structure du réseau — à l'aide de réseaux de neurones sur graphes (GNN) ?

Le projet explore trois angles complémentaires :

| Partie | Modèle | Cible |
|--------|--------|-------|
| Anomalies de **liens** | GAE et VGAE (autoencodeurs de graphe) | liaisons aériennes suspectes |
| Anomalies de **nœuds** (1) | DOMINANT (GCN autoencodeur, structure + attributs) | aéroports incohérents |
| Anomalies de **nœuds** (2) | GAT (régression) | aéroports dont la population s'écarte de la prédiction |

Métriques : **ROC-AUC**, **Average Precision (AP)**, **Precision@10** pour les
liens ; **AUC / AP** pour DOMINANT ; **MAE, RMSE, R²** (échelle log) pour le GAT.

## Données

Fichier : `data/airportsAndCoordAndPop.graphml` — **3 363 aéroports**,
**13 547 liaisons**, **211 pays**. Graphe non orienté, jamais modifié par le code.

| Attribut nœud | Rôle |
|---------------|------|
| `lat`, `lon`, `population` | Features d'entrée (`data.x`), standardisées |
| `country` | Informatif (exporté dans les CSV de liens suspects) |
| `city_name` | Informatif (affichage et export) |

Détail important : **environ la moitié des nœuds (1 673 sur 3 363) ont une
population de 10 000**, probablement une valeur par défaut du jeu de données.
Ces valeurs sont donc à prendre avec des pincettes ; le GCN (partie DOMINANT)
ajoute explicitement une feature « population manquante » pour l'indiquer au
modèle, et le GAT s'appuie avant tout sur `lat`, `lon` et le degré.

## Installation

- Python ≥ 3.12
- [conda](https://docs.conda.io/) (recommandé sous Windows) **ou** pip / [uv](https://docs.astral.sh/uv/)

### Avec conda (recommandé)

```bash
conda create -n gnn312 python=3.12 -y
conda activate gnn312
conda install -y numpy networkx matplotlib pytorch cpuonly -c pytorch -c conda-forge
pip install torch-geometric
```

### Avec pip

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -e .
```

### Avec uv

```bash
uv sync
```

Remarque : `src/evaluate_gcn.py` importe `scikit-learn` (non listé dans
`pyproject.toml`) ; l'installer à part si besoin (`pip install scikit-learn`).

## Exécution

Toutes les commandes se lancent **à la racine du dépôt** :

```bash
conda activate gnn312

# Partie liens — GAE contre VGAE, 4 taux x 5 graines
python experiments/run_comparison.py

# Partie liens — baselines sans apprentissage (aucune dépendance PyTorch)
python experiments/run_baselines.py

# Partie liens — ablation du GAE à 10 % d'anomalies
python experiments/run_ablation.py

# Partie liens — 10 liaisons les plus suspectes du graphe d'origine
python experiments/run_top10_suspicious.py

# Partie nœuds — DOMINANT : anomalies du graphe d'origine + explications
python src/gcn.py

# Partie nœuds — DOMINANT : évaluation avec anomalies de position injectées
python src/evaluate_gcn.py

# Partie nœuds — GAT : régression de la population, 5 graines + top 20 suspects
python experiments/run_gat.py
```

| Script | Contenu | Sorties |
|---|---|---|
| `experiments/run_comparison.py` | GAE contre VGAE, taux {1, 5, 10, 20 %} × 5 graines | `results/csv/gae_vs_vgae.csv`, `gae_vs_vgae_summary.csv`, figures dans `results/figures/` |
| `experiments/run_baselines.py` | Distance, Jaccard, Adamic-Adar (sans PyTorch) | `results/csv/baselines_liens.csv` |
| `experiments/run_ablation.py` | Ablation du GAE à 10 % d'anomalies | `results/csv/ablation_gae.csv` |
| `experiments/run_top10_suspicious.py` | Top 10 liaisons suspectes du graphe d'origine | `results/csv/gae_top10_suspicious_edges.csv`, `vgae_top10_suspicious_edges.csv` |
| `src/gcn.py` | DOMINANT sur le graphe d'origine, 5 runs moyennés + explication par nœud | console |
| `src/evaluate_gcn.py` | DOMINANT avec nœuds déplacés au hasard, AUC / AP | console |
| `experiments/run_gat.py` | GAT : prédiction de la population, 5 graines + top 20 suspects | `results/csv/gat_regression.csv`, `gat_top20_suspicious_nodes.csv` |

## Reproductibilité et remarques

- Les graines (Python, NumPy, PyTorch) sont fixées par `src/utils.set_seed` ;
  mêmes listes de graines et de taux pour toutes les parties (`src/config.py`).
- `run_baselines.py` tire les liens injectés avec NumPy, `run_comparison.py`
  avec PyTorch : même loi, mais pas les mêmes liens.
- `src/metrics.py` calcule AUC et AP sans sklearn (celui-ci pose problème sur
  certains postes Windows) et traite les égalités de scores par blocs, comme
  sklearn.
- Les identifiants NetworkX ont des trous : `data.nx_node_ids[i]` donne
  l'identifiant d'origine du nœud PyG `i`, utilisé pour les exports CSV.
- `src/gcn.py` et `src/evaluate_gcn.py` construisent leur chemin de données
  relatif à `src/` : les lancer depuis la racine du dépôt peut nécessiter
  d'ajuster `DATA_PATH` (les autres scripts utilisent `src/config.DATA_PATH`).
- Le rapport LaTeX se compile dans `rapport/` (`latexmk main.tex`), le PDF
  compilé est `rapport/main.pdf`.

## Utilisation de l'IA générative

L'IA générative a été utilisée pour mieux comprendre les concepts et le code
présentés dans les articles scientifiques utiliser pour ce projet.