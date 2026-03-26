"""
Resume training from a saved XGBoost model.
Runs stages 5 (MAPIE) and 6 (DiCE) only.

Use this when:
  - The XGBoost model is already saved in backend/artifacts/
  - You need to re-run calibration or DiCE setup without retraining.

Run: python -m backend.scripts.resume_training
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.classifier import CreditClassifier
from backend.ml.config import (
    ARTIFACTS_DIR, CALIBRATION_SIZE, DATA_DIR,
    DICE_DATA_PATH, DICE_TRAINING_SAMPLE_SIZE,
    FEATURE_NAMES_PATH, FEATURE_STATS_PATH,
    LABEL_ENCODERS_PATH, MAPIE_PATH, MODEL_PATH,
    RANDOM_STATE, TEST_SIZE, TRAINING_SAMPLE_PATH,
)
from backend.ml.counterfactuals import CounterfactualGenerator
from backend.ml.feature_engineering import build_features, encode_categoricals
from backend.ml.grey_zone import GreyZonePredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("resume_training")


def main():
    logger.info("Resuming from saved XGBoost model…")

    # ── Load saved model ──────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        logger.error(f"Model not found at {MODEL_PATH}. Run train.py first.")
        sys.exit(1)

    clf = CreditClassifier.load(MODEL_PATH)
    logger.info(f"Model loaded — val_auc={clf.val_auc:.4f}, features={len(clf.feature_names)}")

    feature_names = joblib.load(FEATURE_NAMES_PATH)
    feature_stats = joblib.load(FEATURE_STATS_PATH)
    label_encoders = joblib.load(LABEL_ENCODERS_PATH)

    # ── Rebuild calibration and training splits ───────────────────────────────
    logger.info("Re-building feature matrix for calibration split…")
    df = build_features(DATA_DIR, split="train", use_supplementary=True)

    TARGET_COL, ID_COL = "TARGET", "SK_ID_CURR"
    y = df[TARGET_COL].copy()
    X_raw = df.drop(columns=[c for c in [TARGET_COL, ID_COL] if c in df.columns])
    X_enc, _ = encode_categoricals(X_raw, encoders=label_encoders)
    X_enc = X_enc.replace([np.inf, -np.inf], np.nan)
    X_enc = X_enc[feature_names]   # align columns exactly

    # Reproduce the same splits as train.py
    X_temp, X_cal, y_temp, y_cal = train_test_split(
        X_enc, y, test_size=CALIBRATION_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train, _, y_train, _ = train_test_split(
        X_temp, y_temp,
        test_size=TEST_SIZE / (1 - CALIBRATION_SIZE),
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )
    logger.info(f"Cal: {len(X_cal)} | Train sample available: {len(X_train)}")

    # ── Stage 5: MAPIE calibration ────────────────────────────────────────────
    logger.info("[5/6] Calibrating MAPIE grey zone predictor (method=lac)…")
    gzp = GreyZonePredictor(alpha=0.10, random_state=RANDOM_STATE)
    cal_stats = gzp.calibrate(clf.model, X_cal, y_cal)
    logger.info(f"Grey zone stats: {cal_stats}")
    gzp.save(MAPIE_PATH)
    logger.info(f"MAPIE saved → {MAPIE_PATH}")

    # ── Stage 6: DiCE setup ────────────────────────────────────────────────────
    logger.info("[6/6] Setting up DiCE counterfactual generator…")
    sample_idx = X_train.sample(
        n=min(DICE_TRAINING_SAMPLE_SIZE, len(X_train)),
        random_state=RANDOM_STATE,
    ).index
    training_sample = X_train.loc[sample_idx].copy()
    training_sample["TARGET"] = y_train.loc[sample_idx].values

    # Treat ALL features as continuous for DiCE.
    # After label encoding, every column is numeric (int/float). DiCE's
    # "categorical" mode is designed for original string categories, not
    # label-encoded integers — passing encoded cols as categorical causes
    # string-vs-float type errors at inference time.
    continuous_features = [col for col in X_train.columns]

    gen = CounterfactualGenerator()
    gen.setup(
        model=clf.model,
        training_sample=training_sample,
        continuous_features=continuous_features,
        outcome_col="TARGET",
    )
    gen.save(DICE_DATA_PATH)
    joblib.dump(training_sample, TRAINING_SAMPLE_PATH)
    logger.info(f"DiCE saved → {DICE_DATA_PATH}")

    logger.info("=" * 60)
    logger.info("RESUME COMPLETE")
    logger.info(f"  Model AUC:       {clf.val_auc:.4f}")
    logger.info(f"  Grey zone rate:  {cal_stats['grey_zone_rate']:.1%}")
    logger.info(f"  All artifacts:   {ARTIFACTS_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
