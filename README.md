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

## Structure du projet

```
projet-GNN_2026-2027/
├── main.py                              # lance TOUT (ou une sélection)
├── data/
│   └── airportsAndCoordAndPop.graphml   # graphe original (jamais modifié)
├── src/                                 # bibliothèque réutilisable
│   ├── config.py                        # hyperparamètres, chemins, seeds
│   ├── data.py                          # chargement GraphML + RandomLinkSplit
│   ├── gae.py / vgae.py                 # autoencodeurs (anomalies de liens)
│   ├── link_anomaly.py                  # scores 1-p_ij + injection de fausses arêtes
│   ├── evaluation.py / metrics.py       # AUC, AP, Precision@K
│   ├── io_results.py                    # écriture CSV / figures
│   ├── utils.py                         # seeds, device
│   ├── gcn.py                           # DOMINANT (anomalies de nœuds)
│   ├── evaluate_gcn.py                  # évaluation DOMINANT + anomalies injectées
│   └── gat.py                           # GAT régression population
├── experiments/                         # scripts exécutables séparément
│   ├── run_comparison.py                # GAE vs VGAE (multi-seeds / taux)
│   ├── run_baselines.py                 # baselines liens sans apprentissage
│   ├── run_ablation.py                  # ablation GAE
│   ├── run_top10_suspicious.py          # Top 10 liens suspects
│   └── run_gat.py                       # GAT + export CSV
├── results/
│   ├── csv/                             # tableaux de résultats
│   └── figures/                         # figures générées
├── rapport/                             # rapport LaTeX (Overleaf / latexmk)
│   ├── main.tex
│   ├── sections/
│   ├── figures/
│   ├── bib/
│   └── cls/
├── pyproject.toml
├── uv.lock
└── README.md
```

Séparation voulue :

- **`main.py`** = orchestrateur (tout lancer, ou `--only` / `--skip`) ;
- **`src/`** = code réutilisable (modèles, données, métriques) ;
- **`experiments/`** = points d’entrée des expériences (toujours lançables seuls) ;
- **`results/`** = sorties (CSV, figures), séparées du code ;
- **`data/`** = jeu de données original, non modifié.

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
cd /chemin/vers/projet-GNN_2026-2027
```

### Option A — tout lancer via `main.py`

```bash
# Toutes les expériences (baselines, GAE/VGAE, ablation, top10, GCN, GAT)
python main.py

# Lister les étapes
python main.py --list

# Une ou plusieurs étapes seulement
python main.py --only comparison gat
python main.py --only gcn evaluate_gcn

# Tout sauf certaines étapes
python main.py --skip evaluate_gcn

# Continuer même si une étape échoue
python main.py --continue-on-error
```

Étapes reconnues par `main.py` : `baselines`, `comparison`, `ablation`, `top10`,
`gcn`, `evaluate_gcn`, `gat`  
(alias utiles : `gae`/`vgae` → `comparison`, `dominant` → `gcn`).

### Option B — exécuter chaque modèle / expérience séparément

Les scripts individuels restent disponibles (même comportement qu’avant) :

```bash
# Partie liens — GAE contre VGAE, 4 taux x 5 graines
python experiments/run_comparison.py

# Partie liens — baselines sans apprentissage
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
| `main.py` | Orchestrateur : toutes les étapes ou une sélection | idem aux scripts appelés |
| `experiments/run_comparison.py` | GAE contre VGAE, taux {1, 5, 10, 20 %} × 5 graines | `results/csv/gae_vs_vgae.csv`, `gae_vs_vgae_summary.csv`, figures dans `results/figures/` |
| `experiments/run_baselines.py` | Distance, Jaccard, Adamic-Adar (sans PyTorch Geometric entraîné) | `results/csv/baselines_liens.csv` |
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
- Les chemins de données passent par la **racine du dépôt** (`data/…`), y compris
  pour `src/gcn.py` et `src/evaluate_gcn.py`.
- Le rapport LaTeX se compile dans `rapport/` (`latexmk main.tex`), le PDF
  compilé est `rapport/main.pdf`.

## Utilisation de l'IA générative

L'IA générative a été utilisée pour mieux comprendre les concepts et le code
présentés dans les articles scientifiques utilisés pour ce projet.
