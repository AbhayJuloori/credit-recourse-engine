"""
Grey Zone Wrapper — Conformal Prediction with MAPIE.

Wraps the trained XGBClassifier in a conformal predictor.
Each applicant gets a zone:

  GREEN  (Approve)  — prediction set = {0}          at alpha coverage
  AMBER  (Grey)     — prediction set = {0, 1}        → borderline, human review
  RED    (Deny)     — prediction set = {1}

The grey zone is the key insight: these applicants sit near the decision
boundary where the model's confidence is low. Auto-denying them is wrong.

Method: RAPS (Regularized Adaptive Prediction Sets), which gives better
calibration than plain APS for imbalanced classes.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from mapie.classification import MapieClassifier

logger = logging.getLogger(__name__)

ZONE_APPROVE = "approve"
ZONE_GREY = "grey"
ZONE_DENY = "deny"

ZONE_LABELS = {
    ZONE_APPROVE: "Approve",
    ZONE_GREY: "Grey Zone — Human Review",
    ZONE_DENY: "Deny",
}

ZONE_COLORS = {
    ZONE_APPROVE: "#22c55e",   # green-500
    ZONE_GREY: "#f59e0b",      # amber-500
    ZONE_DENY: "#ef4444",      # red-500
}


class GreyZonePredictor:
    """
    Conformal prediction wrapper.

    Usage:
        predictor = GreyZonePredictor(alpha=0.10)   # 90% coverage
        predictor.calibrate(fitted_classifier, X_cal, y_cal)
        result = predictor.predict(X_new)

        predictor.save(path)
        predictor2 = GreyZonePredictor.load(path)
    """

    def __init__(self, alpha: float = 0.10, random_state: int = 42):
        self.alpha = alpha
        self.random_state = random_state
        self.mapie: Optional[MapieClassifier] = None
        self.calibration_coverage_: Optional[float] = None

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate(
        self,
        fitted_estimator,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
    ) -> dict:
        """
        Fit MAPIE on the calibration set using a pre-fitted estimator.

        Args:
            fitted_estimator: Any sklearn-compatible estimator already fitted.
            X_cal: Calibration features (NOT seen during model training).
            y_cal: Calibration labels.

        Returns:
            dict with coverage and grey-zone rate stats.
        """
        logger.info(f"Calibrating MAPIE with alpha={self.alpha} on {len(X_cal)} samples…")

        # 'lac' (Least Ambiguous Classifier) is the recommended binary method in MAPIE 0.8.x
        # 'raps' is only available for multiclass targets
        # Store the original estimator directly — MAPIE cv='prefit' does not
        # reliably expose estimator_ after fit, so we keep a direct reference.
        self._estimator = fitted_estimator

        self.mapie = MapieClassifier(
            estimator=fitted_estimator,
            method="lac",
            cv="prefit",
            random_state=self.random_state,
        )
        self.mapie.fit(X_cal, y_cal)

        # Evaluate calibration quality
        _, y_psets = self.mapie.predict(X_cal, alpha=self.alpha, include_last_label=True)
        zones = self._psets_to_zones(y_psets)
        grey_rate = (zones == ZONE_GREY).mean()
        approve_rate = (zones == ZONE_APPROVE).mean()
        deny_rate = (zones == ZONE_DENY).mean()

        # Empirical coverage: among predicted single-label sets, how often correct?
        single_label_mask = zones != ZONE_GREY
        if single_label_mask.any():
            y_pred_single = np.where(
                zones[single_label_mask] == ZONE_APPROVE, 0, 1
            )
            self.calibration_coverage_ = (
                y_pred_single == y_cal.values[single_label_mask]
            ).mean()
        else:
            self.calibration_coverage_ = None

        stats = {
            "alpha": self.alpha,
            "grey_zone_rate": float(grey_rate),
            "approve_rate": float(approve_rate),
            "deny_rate": float(deny_rate),
            "calibration_coverage": self.calibration_coverage_,
        }
        logger.info(
            f"MAPIE calibrated: grey={grey_rate:.1%}, "
            f"approve={approve_rate:.1%}, deny={deny_rate:.1%}"
        )
        return stats

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict zone for each applicant.

        Returns:
            DataFrame with columns:
              zone         : 'approve' / 'grey' / 'deny'
              zone_label   : human-readable label
              zone_color   : hex colour for UI
              p_default    : raw default probability (from MAPIE internal model)
              confidence   : 1.0 for single-label zones, 0.5 for grey
        """
        assert self.mapie is not None, "Call .calibrate() first."

        y_pred, y_psets = self.mapie.predict(X, alpha=self.alpha, include_last_label=True)

        # Raw probability from the underlying classifier
        p_default = self._estimator.predict_proba(X)[:, 1]

        zones = self._psets_to_zones(y_psets)
        confidence = np.where(zones == ZONE_GREY, 0.5, 0.9)

        return pd.DataFrame(
            {
                "zone": zones,
                "zone_label": [ZONE_LABELS[z] for z in zones],
                "zone_color": [ZONE_COLORS[z] for z in zones],
                "p_default": p_default.round(4),
                "confidence": confidence,
            }
        )

    def predict_single(self, x: pd.DataFrame) -> dict:
        """Predict a single applicant and return a plain dict."""
        result = self.predict(x)
        return result.iloc[0].to_dict()

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _psets_to_zones(y_psets: np.ndarray) -> np.ndarray:
        """
        Convert MAPIE prediction sets to zone labels.

        y_psets: shape (n_samples, n_classes, n_alpha_levels)
        n_classes: 2 (0=non-default, 1=default)
        """
        # y_psets[:, 0, 0] → True if class 0 (approve) is in prediction set
        # y_psets[:, 1, 0] → True if class 1 (deny)    is in prediction set
        include_0 = y_psets[:, 0, 0]
        include_1 = y_psets[:, 1, 0]

        zones = np.full(len(y_psets), ZONE_GREY, dtype=object)
        zones[include_0 & ~include_1] = ZONE_APPROVE
        zones[~include_0 & include_1] = ZONE_DENY
        # Both False (shouldn't happen at reasonable alpha) → grey
        zones[~include_0 & ~include_1] = ZONE_GREY
        return zones

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        payload = {
            "mapie": self.mapie,
            "estimator": self._estimator,
            "alpha": self.alpha,
            "random_state": self.random_state,
            "calibration_coverage": self.calibration_coverage_,
        }
        joblib.dump(payload, path)
        logger.info(f"GreyZonePredictor saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "GreyZonePredictor":
        payload = joblib.load(path)
        gzp = cls(alpha=payload["alpha"], random_state=payload["random_state"])
        gzp.mapie = payload["mapie"]
        gzp._estimator = payload.get("estimator")
        gzp.calibration_coverage_ = payload["calibration_coverage"]
        logger.info(f"GreyZonePredictor loaded from {path}")
        return gzp
