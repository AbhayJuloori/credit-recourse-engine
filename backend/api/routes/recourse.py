"""
Recourse endpoint — Layer 3 + Layer 4.

POST /api/recourse
  For grey-zone or denied applicants.
  Generates feasibility-constrained counterfactuals via DiCE,
  then ranks them by effort × flip-probability.

  Returns top-3 ranked recourse paths, each with:
  - Concrete feature changes
  - Flip probability
  - Effort score
  - Estimated time
  - Human-readable action steps
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.api.routes.predict import ApplicantFeatures, _build_feature_row
from backend.ml.config import TOP_K_PATHS, NUM_COUNTERFACTUALS
from backend.ml.effort_ranker import EffortRanker, format_ranked_paths_for_api

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────

class RecourseRequest(BaseModel):
    applicant: ApplicantFeatures
    num_paths: Optional[int] = 3


class RecourseStep(BaseModel):
    action: str
    feature: str
    label: str
    original: float
    cf_value: float
    delta: float
    time_estimate: str


class RecoursePathItem(BaseModel):
    rank: int
    flip_probability: float
    effort_score: float
    composite_score: float
    time_estimate: str
    steps: list


class RecourseResponse(BaseModel):
    zone: str
    zone_label: str
    p_default: float
    recourse_available: bool
    paths: list
    message: str


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/recourse", response_model=RecourseResponse)
async def get_recourse(request: Request, body: RecourseRequest):
    """
    Generate ranked recourse paths for a grey-zone or denied applicant.
    """
    state = request.app.state

    if state.classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training pipeline first.",
        )

    features = body.applicant
    top_k = min(body.num_paths or TOP_K_PATHS, 5)

    # ── Build feature row ─────────────────────────────────────────────────────
    X = _build_feature_row(
        features=features.model_dump(exclude_none=False),
        feature_names=state.feature_names,
        feature_stats=state.feature_stats,
        label_encoders=state.label_encoders,
    )

    # ── Current prediction ────────────────────────────────────────────────────
    if state.grey_zone is not None:
        pred = state.grey_zone.predict_single(X)
        zone = pred["zone"]
        zone_label = pred["zone_label"]
        p_default = pred["p_default"]
    else:
        p_default = float(state.classifier.predict_proba(X)[0])
        from backend.api.routes.predict import _simple_zone
        zone, zone_label, _, _ = _simple_zone(p_default)

    # ── Recourse only for grey / deny ─────────────────────────────────────────
    if zone == "approve":
        return RecourseResponse(
            zone=zone,
            zone_label=zone_label,
            p_default=round(p_default, 4),
            recourse_available=False,
            paths=[],
            message="Applicant is already in the Approve zone. No recourse needed.",
        )

    # ── Generate counterfactuals ──────────────────────────────────────────────
    if state.cf_generator is None:
        return RecourseResponse(
            zone=zone,
            zone_label=zone_label,
            p_default=round(p_default, 4),
            recourse_available=False,
            paths=[],
            message="Counterfactual generator not available. Run train.py first.",
        )

    paths = state.cf_generator.generate(
        instance=X,
        num_cfs=NUM_COUNTERFACTUALS,
        desired_class=0,   # flip to Approve
    )

    if not paths:
        return RecourseResponse(
            zone=zone,
            zone_label=zone_label,
            p_default=round(p_default, 4),
            recourse_available=False,
            paths=[],
            message=(
                "No feasible recourse paths found within current constraints. "
                "The applicant's profile may be too far from the approval boundary."
            ),
        )

    # ── Rank paths by effort × flip probability ───────────────────────────────
    feature_stds = state.feature_stats.get("std", {})
    ranker = EffortRanker(feature_stds=feature_stds)
    ranked = ranker.rank(
        paths=paths,
        model=state.classifier.model,
        original_row=X,
        top_k=top_k,
    )

    formatted = format_ranked_paths_for_api(ranked)

    return RecourseResponse(
        zone=zone,
        zone_label=zone_label,
        p_default=round(p_default, 4),
        recourse_available=True,
        paths=formatted,
        message=f"Found {len(formatted)} feasible recourse path(s).",
    )
