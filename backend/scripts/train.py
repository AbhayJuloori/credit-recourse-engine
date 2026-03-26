"""
Full training pipeline for the Credit Recourse Engine.

Run from the project root:
    python -m backend.scripts.train

Pipeline stages:
    1. Feature engineering (all 6 tables)
    2. Label encoding of categoricals
    3. Train/val/calibration split
    4. XGBoost + Optuna hyperparameter search
    5. MAPIE calibration (grey zone)
    6. DiCE setup (counterfactual generator)
    7. Save all artifacts

Expected runtime: 90–180 minutes on a MacBook M-chip.
Expected AUC:     0.80–0.82 with all supplementary tables.
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Make sure the project root is on the path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.classifier import CreditClassifier
from backend.ml.config import (
    ARTIFACTS_DIR,
    CALIBRATION_SIZE,
    DATA_DIR,
    DICE_DATA_PATH,
    DICE_TRAINING_SAMPLE_SIZE,
    FEATURE_NAMES_PATH,
    FEATURE_STATS_PATH,
    LABEL_ENCODERS_PATH,
    MAPIE_PATH,
    MODEL_PATH,
    OPTUNA_TIMEOUT,
    OPTUNA_TRIALS,
    RANDOM_STATE,
    TEST_SIZE,
    TRAINING_SAMPLE_PATH,
)
from backend.ml.counterfactuals import CounterfactualGenerator
from backend.ml.feature_engineering import build_features, encode_categoricals
from backend.ml.grey_zone import GreyZonePredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def main():
    logger.info("=" * 60)
    logger.info("Credit Recourse Engine — Training Pipeline")
    logger.info("=" * 60)

    # ── Stage 1: Feature engineering ─────────────────────────────────────────
    logger.info("\n[1/6] Building feature matrix…")
    df = build_features(DATA_DIR, split="train", use_supplementary=True)
    logger.info(f"Feature matrix shape: {df.shape}")

    TARGET_COL = "TARGET"
    ID_COL = "SK_ID_CURR"

    y = df[TARGET_COL].copy()
    drop_cols = [TARGET_COL, ID_COL]
    X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ── Stage 2: Encode categoricals ─────────────────────────────────────────
    logger.info("\n[2/6] Encoding categorical features…")
    X_enc, label_encoders = encode_categoricals(X_raw)
    joblib.dump(label_encoders, LABEL_ENCODERS_PATH)
    logger.info(f"Encoded features: {X_enc.shape[1]} columns")

    # Replace inf values
    X_enc = X_enc.replace([np.inf, -np.inf], np.nan)

    feature_names = list(X_enc.columns)
    joblib.dump(feature_names, FEATURE_NAMES_PATH)

    # Feature statistics for effort ranker + imputation
    feature_stats = {
        "mean": X_enc.mean().to_dict(),
        "std": X_enc.std().to_dict(),
        "median": X_enc.median().to_dict(),
        "min": X_enc.min().to_dict(),
        "max": X_enc.max().to_dict(),
    }
    joblib.dump(feature_stats, FEATURE_STATS_PATH)
    logger.info("Feature stats saved.")

    # ── Stage 3: Split ────────────────────────────────────────────────────────
    logger.info("\n[3/6] Splitting data…")

    # First split off calibration set (for MAPIE)
    X_temp, X_cal, y_temp, y_cal = train_test_split(
        X_enc, y,
        test_size=CALIBRATION_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    # Then split train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=TEST_SIZE / (1 - CALIBRATION_SIZE),
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    logger.info(f"Train: {len(X_train)} rows | Val: {len(X_val)} | Cal: {len(X_cal)}")
    logger.info(f"Default rate — train: {y_train.mean():.2%} | val: {y_val.mean():.2%}")

    # ── Stage 4: Train XGBoost + Optuna ──────────────────────────────────────
    logger.info("\n[4/6] Training XGBoost with Optuna…")
    clf = CreditClassifier(random_state=RANDOM_STATE)
    best_auc = clf.train(
        X_train, y_train,
        X_val, y_val,
        n_trials=OPTUNA_TRIALS,
        timeout=OPTUNA_TIMEOUT,
    )
    logger.info(f"Best validation AUC: {best_auc:.4f}")

    clf.save(MODEL_PATH)
    logger.info(f"Model saved → {MODEL_PATH}")

    # ── Stage 5: MAPIE calibration ────────────────────────────────────────────
    logger.info("\n[5/6] Calibrating MAPIE grey zone predictor…")
    gzp = GreyZonePredictor(alpha=0.10, random_state=RANDOM_STATE)
    cal_stats = gzp.calibrate(clf.model, X_cal, y_cal)
    logger.info(f"Grey zone stats: {cal_stats}")
    gzp.save(MAPIE_PATH)
    logger.info(f"MAPIE saved → {MAPIE_PATH}")

    # ── Stage 6: DiCE setup ────────────────────────────────────────────────────
    logger.info("\n[6/6] Setting up DiCE counterfactual generator…")

    # Build a training sample with TARGET for DiCE
    sample_idx = X_train.sample(
        n=min(DICE_TRAINING_SAMPLE_SIZE, len(X_train)),
        random_state=RANDOM_STATE,
    ).index
    training_sample = X_train.loc[sample_idx].copy()
    training_sample["TARGET"] = y_train.loc[sample_idx].values

    # Identify continuous features (numeric with > 20 unique values)
    # All features treated as continuous — label-encoded categoricals are
    # numeric post-encoding; DiCE's categorical mode expects original strings.
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
    logger.info(f"DiCE generator saved → {DICE_DATA_PATH}")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Validation AUC:    {best_auc:.4f}")
    logger.info(f"  Grey zone rate:    {cal_stats['grey_zone_rate']:.1%}")
    logger.info(f"  Features used:     {len(feature_names)}")
    logger.info(f"  Artifacts in:      {ARTIFACTS_DIR}")
    logger.info("=" * 60)

    if best_auc < 0.78:
        logger.warning(
            "AUC below 0.78 — consider increasing OPTUNA_TRIALS "
            "or checking feature engineering."
        )


if __name__ == "__main__":
    main()
