# Projet GNN 2026-2027 : anomalies de liens dans un réseau d'aéroports

Les autoencodeurs de graphe (GAE, VGAE) repèrent-ils les liens aberrants d'un réseau d'aéroports, mieux qu'une heuristique sans apprentissage ? Ce dépôt contient le volet « liens » du projet de groupe. Le rapport est dans `rapport/`.

## Données

`data/airportsAndCoordAndPop.graphml` : 3363 aéroports, 13547 liens non orientés. Attributs d'un nœud : `lat`, `lon`, `population` (entrées des modèles, standardisées), `city_name`, `country` (lecture des résultats uniquement). Le fichier n'est jamais modifié.

## Installation

Python 3.12 ou plus.

```bash
conda create -n gnn312 python=3.12 -y
conda activate gnn312
pip install -e .
```

Versions utilisées : `torch` 2.14.1, `torch-geometric` 2.8.0.

## Expériences

À lancer depuis la racine du dépôt. Les paramètres (taux d'anomalies, graines, dimensions, époques) sont dans `src/config.py`.

| Commande | Contenu | Sortie (`results/csv/`) |
|---|---|---|
| `python experiments/run_comparison.py` | GAE contre VGAE, 4 taux × 5 graines | `gae_vs_vgae.csv`, `gae_vs_vgae_summary.csv`, figures dans `results/figures/` |
| `python experiments/run_baselines.py` | Distance, Jaccard, Adamic-Adar (sans PyTorch) | `baselines_liens.csv` |
| `python experiments/run_ablation.py` | Ablation du GAE à 10 % d'anomalies | `ablation_gae.csv` |
| `python experiments/run_top10_suspicious.py` | 10 liens les plus suspects du graphe d'origine | `gae_top10_suspicious_edges.csv`, `vgae_top10_suspicious_edges.csv` |

Protocole : on ajoute `r × |E|` liens entre des paires non reliées tirées au hasard, avant `RandomLinkSplit` (10 % validation, 20 % test). Les modèles sont entraînés par reconstruction, sans étiquette. Chaque lien du graphe contaminé reçoit le score `1 - σ(z_i · z_j)` ; on mesure AUC et AP contre les liens injectés.

Paramètres : couches 3 → 64 → 32, Adam, taux d'apprentissage 0,01, 200 époques, graines 42, 123, 456, 789, 2026.

## Remarques

- `run_baselines.py` tire les liens injectés avec NumPy, `run_comparison.py` avec PyTorch : même loi, pas les mêmes liens.
- `src/metrics.py` calcule AUC et AP sans sklearn (celui-ci pose problème sur certains postes Windows) et traite les égalités de scores par blocs.

## Structure

```
data/         graphe GraphML
src/          config, données, modèles (gae.py, vgae.py), scores, métriques
experiments/  un script par expérience
results/      csv/ et figures/
rapport/      main.tex + sections/ (un fichier par partie), bibM2.bib, template llncs
```

## Rapport

```bash
cd rapport
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Usage d'IA générative

ChatGPT et Cursor ont servi à comprendre les API de PyTorch Geometric (`GAE`, `VGAE`, `RandomLinkSplit`), à structurer les expériences (découpage, graines, anomalies injectées, CSV) et à déboguer l'import de `sklearn.metrics` sous Windows. Claude (Anthropic) a servi à relire le dépôt et le rapport, à écrire `run_baselines.py`, à nettoyer le code et à rédiger un premier jet de la partie liens du rapport.
