"""
Prediction endpoint — Layer 1 + Layer 2.

POST /api/predict
  Takes applicant features, returns:
  - P(default)
  - Zone: approve / grey / deny
  - Confidence band
  - Top SHAP contributors
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.ml.feature_engineering import engineer_application_features

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────

class ApplicantFeatures(BaseModel):
    """
    Key applicant features for the prediction form.
    Most features the user fills in; the rest are imputed from training medians.
    """
    # Core financials
    AMT_INCOME_TOTAL: float = Field(..., description="Annual income (USD)", gt=0)
    AMT_CREDIT: float = Field(..., description="Loan amount requested (USD)", gt=0)
    AMT_ANNUITY: float = Field(..., description="Monthly loan annuity (USD)", gt=0)
    AMT_GOODS_PRICE: Optional[float] = Field(None, description="Goods price (USD)")

    # Employment
    DAYS_EMPLOYED: Optional[float] = Field(None, description="Days employed (negative = past days)")
    DAYS_BIRTH: float = Field(..., description="Days since birth (negative number)", lt=0)

    # External credit scores
    EXT_SOURCE_1: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_2: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_3: Optional[float] = Field(None, ge=0, le=1)

    # Demographics
    CODE_GENDER: Optional[str] = Field(None, description="M or F")
    CNT_FAM_MEMBERS: Optional[float] = Field(None, ge=1)
    CNT_CHILDREN: Optional[int] = Field(None, ge=0)
    NAME_EDUCATION_TYPE: Optional[str] = None
    NAME_FAMILY_STATUS: Optional[str] = None
    ORGANIZATION_TYPE: Optional[str] = None

    # Flags
    FLAG_OWN_CAR: Optional[str] = Field(None, description="Y or N")
    FLAG_OWN_REALTY: Optional[str] = Field(None, description="Y or N")
    REGION_RATING_CLIENT: Optional[int] = Field(None, ge=1, le=3)
    REGION_RATING_CLIENT_W_CITY: Optional[int] = Field(None, ge=1, le=3)

    # Social circle
    DEF_30_CNT_SOCIAL_CIRCLE: Optional[float] = Field(None, ge=0)
    DEF_60_CNT_SOCIAL_CIRCLE: Optional[float] = Field(None, ge=0)

    model_config = {"extra": "allow"}


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    zone: str
    zone_label: str
    zone_color: str
    p_default: float
    confidence: float
    shap_top_features: list
    model_loaded: bool


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, features: ApplicantFeatures):
    """
    Predict default probability and zone for an applicant.
    """
    state = request.app.state

    if state.classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training pipeline first.",
        )

    # Build feature row
    X = _build_feature_row(
        features=features.model_dump(exclude_none=False),
        feature_names=state.feature_names,
        feature_stats=state.feature_stats,
        label_encoders=state.label_encoders,
    )

    # Prediction (with or without MAPIE)
    if state.grey_zone is not None:
        result = state.grey_zone.predict_single(X)
        zone = result["zone"]
        zone_label = result["zone_label"]
        zone_color = result["zone_color"]
        p_default = result["p_default"]
        confidence = result["confidence"]
    else:
        # Fallback: direct XGBoost probability
        p_default = float(state.classifier.predict_proba(X)[0])
        zone, zone_label, zone_color, confidence = _simple_zone(p_default)

    # SHAP top features
    shap_top = _get_shap_contributions(state.classifier, X, top_k=8)

    return PredictionResponse(
        zone=zone,
        zone_label=zone_label,
        zone_color=zone_color,
        p_default=round(p_default, 4),
        confidence=float(confidence),
        shap_top_features=shap_top,
        model_loaded=True,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_feature_row(
    features: Dict[str, Any],
    feature_names: list,
    feature_stats: dict,
    label_encoders: dict,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame aligned to the training feature set.
    Missing features are imputed with training medians.
    """
    # Start from training medians (imputation baseline)
    row = {f: feature_stats["median"].get(f, 0.0) for f in feature_names}

    # Apply engineering to raw application features first
    raw_app = {k: v for k, v in features.items() if v is not None}
    raw_df = pd.DataFrame([raw_app])

    # Apply application-level engineering
    try:
        engineered = engineer_application_features(raw_df)
        for col in engineered.columns:
            if col not in feature_names:
                continue
            val = engineered[col].iloc[0]
            if val is None:
                continue
            try:
                fval = float(val)
                if not np.isnan(fval) and not np.isinf(fval):
                    row[col] = fval
            except (ValueError, TypeError):
                pass  # string categoricals — label encoder handles them below
    except Exception as e:
        logger.warning(f"Feature engineering failed: {e}")
        for k, v in raw_app.items():
            if k in feature_names and v is not None:
                row[k] = v

    # Encode any categorical features that were provided
    for col, le in label_encoders.items():
        if col in row and isinstance(row[col], str):
            try:
                row[col] = float(le.transform([str(row[col])])[0])
            except Exception:
                row[col] = float(feature_stats["median"].get(col, 0.0))

    X = pd.DataFrame([row])[feature_names]
    X = X.replace([float("inf"), float("-inf")], float("nan"))
    return X


def _simple_zone(p_default: float) -> tuple:
    """Simple threshold-based zone without MAPIE."""
    from backend.ml.grey_zone import (
        ZONE_APPROVE, ZONE_DENY, ZONE_GREY,
        ZONE_LABELS, ZONE_COLORS,
    )
    if p_default < 0.40:
        z = ZONE_APPROVE
    elif p_default > 0.60:
        z = ZONE_DENY
    else:
        z = ZONE_GREY
    return z, ZONE_LABELS[z], ZONE_COLORS[z], 0.85 if z != ZONE_GREY else 0.5


def _get_shap_contributions(classifier, X: pd.DataFrame, top_k: int = 8) -> list:
    """Compute SHAP values and return top-k feature contributions."""
    try:
        import shap
        explainer = shap.TreeExplainer(classifier.model)
        shap_values = explainer.shap_values(X)
        # For binary classifier, shap_values is either a 2D array or list
        if isinstance(shap_values, list):
            sv = shap_values[1][0]   # class 1 SHAP values for first (only) row
        else:
            sv = shap_values[0]      # single row

        feature_names = list(X.columns)
        contributions = sorted(
            zip(feature_names, sv),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:top_k]

        return [
            {
                "feature": feat,
                "label": feat.replace("_", " ").title(),
                "shap_value": round(float(val), 4),
                "direction": "increases_risk" if val > 0 else "decreases_risk",
                "feature_value": round(float(X[feat].iloc[0]), 4),
            }
            for feat, val in contributions
        ]
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return []
