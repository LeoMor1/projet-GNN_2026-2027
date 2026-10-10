"""Paramètres des expériences (modifier ici uniquement)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "airportsAndCoordAndPop.graphml"
RESULTS_CSV_DIR = PROJECT_ROOT / "results" / "csv"
RESULTS_FIG_DIR = PROJECT_ROOT / "results" / "figures"

# Attributs des nœuds utilisés comme entrée
NODE_FEATURES = ["lat", "lon", "population"]

# RandomLinkSplit (le graphe est non orienté)
NUM_VAL = 0.10
NUM_TEST = 0.20
IS_UNDIRECTED = True
ADD_NEGATIVE_TRAIN_SAMPLES = True
SPLIT_LABELS = True

# Modèles
HIDDEN_CHANNELS = 64
LATENT_DIM = 32
LEARNING_RATE = 0.01
EPOCHS = 200

# Protocole d'évaluation
SEEDS = [42, 123, 456, 789, 2026]
ANOMALY_RATES = [0.01, 0.05, 0.10, 0.20]
PRECISION_AT_K = 10
