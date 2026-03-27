"""
Central configuration for the Credit Recourse Engine.
Paths, thresholds, and pipeline constants.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT_DIR / "backend" / "artifacts"

# DATA_DIR: used only during training. Override via DATA_DIR env var.
# On HuggingFace Spaces (inference only) this path is not accessed.
_data_env = os.environ.get("DATA_DIR")
DATA_DIR = Path(_data_env) if _data_env else Path("/Users/abhayjuloori/home-credit-default-risk")

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "xgb_model.pkl"
MAPIE_PATH = ARTIFACTS_DIR / "mapie_clf.pkl"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.pkl"
FEATURE_STATS_PATH = ARTIFACTS_DIR / "feature_stats.pkl"
DICE_DATA_PATH = ARTIFACTS_DIR / "dice_data.pkl"
LABEL_ENCODERS_PATH = ARTIFACTS_DIR / "label_encoders.pkl"
TRAINING_SAMPLE_PATH = ARTIFACTS_DIR / "training_sample.pkl"

# ── Model ────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.15       # 15% test hold-out
CALIBRATION_SIZE = 0.15  # 15% for MAPIE calibration
OPTUNA_TRIALS = 100
OPTUNA_TIMEOUT = None   # No timeout — run all 100 trials

# Optuna SQLite storage — persists study so it can be resumed if interrupted
OPTUNA_STORAGE = str(ARTIFACTS_DIR / "optuna_study.db")

# ── Conformal prediction ─────────────────────────────────────────────────────
CONFORMAL_ALPHA = 0.10   # 90% coverage
DECISION_THRESHOLD = 0.50

# ── Recourse ─────────────────────────────────────────────────────────────────
NUM_COUNTERFACTUALS = 6   # generate 6, rank and return top 3
TOP_K_PATHS = 3
EFFORT_FLIP_WEIGHT = 0.5
EFFORT_COST_WEIGHT = 0.5

# ── Training sample for DiCE (row count) ─────────────────────────────────────
DICE_TRAINING_SAMPLE_SIZE = 2000
