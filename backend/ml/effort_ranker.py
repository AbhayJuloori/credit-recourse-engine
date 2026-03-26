"""
Layer 4 — Effort-ranked recourse pathways.

Takes raw DiCE counterfactual paths and ranks them by a composite score:

    score = (flip_probability × w_flip) + (feasibility_score × w_cost)

Where:
    flip_probability  = model P(approve | counterfactual) = 1 - P(default | CF)
    feasibility_score = 1 / (1 + total_effort)
    total_effort      = Σ |Δfeature_i| / σ_i × time_weight_i

This is what separates this project from a basic SHAP chart:
a loan officer gets "Path A — reduce DTI by 8% (~3 months) — 84% flip chance"
instead of a bar chart that tells them nothing actionable.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from backend.ml.constraints import (
    FEATURE_LABELS,
    TIME_WEIGHTS,
    TIME_ESTIMATES,
    get_feature_label,
    get_time_estimate,
    get_time_weight,
)

logger = logging.getLogger(__name__)


class EffortRanker:
    """
    Ranks counterfactual paths by effort × flip-probability.

    Usage:
        ranker = EffortRanker(feature_stds=df_train.std().to_dict())
        ranked = ranker.rank(paths, model, original_row)
    """

    def __init__(
        self,
        feature_stds: dict,
        flip_weight: float = 0.5,
        cost_weight: float = 0.5,
    ):
        """
        Args:
            feature_stds: {feature_name: std_dev} from training set.
                          Used to normalise Δ by variance of each feature.
            flip_weight: weight for flip probability in composite score.
            cost_weight: weight for feasibility (1/effort) in composite score.
        """
        self.feature_stds = feature_stds
        self.flip_weight = flip_weight
        self.cost_weight = cost_weight

    # ── Core ranking ──────────────────────────────────────────────────────────

    def rank(
        self,
        paths: list,
        model,
        original_row: pd.DataFrame,
        top_k: int = 3,
    ) -> list:
        """
        Score and rank counterfactual paths.

        Args:
            paths: output of CounterfactualGenerator.generate()
            model: fitted classifier with predict_proba()
            original_row: single-row DataFrame (the applicant)
            top_k: number of top paths to return

        Returns:
            List of ranked path dicts, sorted by composite score descending.
            Each dict contains:
              rank              : 1-indexed rank
              changes           : list of change dicts (feature, original, cf, delta)
              flip_probability  : P(approve | CF)
              effort_score      : raw effort value
              feasibility_score : 1 / (1 + effort)
              composite_score   : final ranking score
              steps             : list of human-readable action strings
              time_estimate     : estimated total time as string
        """
        if not paths:
            return []

        scored = []
        for path in paths:
            try:
                scored_path = self._score_path(path, model, original_row)
                if scored_path is not None:
                    scored.append(scored_path)
            except Exception as exc:
                logger.debug(f"Skipping path due to scoring error: {exc}")
                continue

        if not scored:
            return []

        # Sort by composite score descending
        scored.sort(key=lambda x: x["composite_score"], reverse=True)

        # Add rank index
        for rank_idx, item in enumerate(scored[:top_k], start=1):
            item["rank"] = rank_idx

        return scored[:top_k]

    # ── Path scoring ──────────────────────────────────────────────────────────

    def _score_path(self, path: dict, model, original_row: pd.DataFrame) -> Optional[dict]:
        changes = path.get("changes", [])
        if not changes:
            return None

        cf_row = path.get("cf_row")
        if cf_row is None:
            return None

        # Build CF dataframe for prediction
        cf_df = original_row.copy()
        for change in changes:
            feat = change["feature"]
            if feat in cf_df.columns:
                cf_df.iloc[0, cf_df.columns.get_loc(feat)] = change["cf"]

        # Flip probability: P(non-default | CF) = 1 - P(default | CF)
        try:
            p_default_cf = model.predict_proba(cf_df)[:, 1][0]
            flip_prob = float(1.0 - p_default_cf)
        except Exception:
            flip_prob = 0.5  # fallback

        # Effort score
        effort = self._compute_effort(changes)
        feasibility = 1.0 / (1.0 + effort)

        composite = (
            self.flip_weight * flip_prob
            + self.cost_weight * feasibility
        )

        # Human-readable steps
        steps = self._generate_steps(changes)
        time_str = self._estimate_total_time(changes)

        return {
            "changes": changes,
            "flip_probability": round(flip_prob, 4),
            "effort_score": round(effort, 4),
            "feasibility_score": round(feasibility, 4),
            "composite_score": round(composite, 4),
            "steps": steps,
            "time_estimate": time_str,
        }

    # ── Effort computation ────────────────────────────────────────────────────

    def _compute_effort(self, changes: list) -> float:
        """
        effort = Σ |Δfeature_i| / max(σ_i, 1e-6) × time_weight_i

        Normalising by σ converts feature changes to standard-deviation units,
        making them comparable across features with very different scales.
        """
        total = 0.0
        for change in changes:
            feat = change["feature"]
            delta = abs(change["delta"])
            sigma = max(self.feature_stds.get(feat, 1.0), 1e-6)
            t_weight = get_time_weight(feat)
            total += (delta / sigma) * t_weight
        return total

    # ── Human-readable output ─────────────────────────────────────────────────

    def _generate_steps(self, changes: list) -> list:
        """Convert raw change dicts to human-readable action strings."""
        steps = []
        for change in changes:
            feat = change["feature"]
            label = get_feature_label(feat)
            orig = change["original"]
            cf = change["cf"]
            delta = change["delta"]

            # DAYS_EMPLOYED is stored as a negative number (days before application).
            # A more negative value means longer tenure — flip the direction label.
            if feat == "DAYS_EMPLOYED":
                tenure_orig_mo = abs(orig) / 30.44
                tenure_cf_mo   = abs(cf)   / 30.44
                delta_mo = abs(tenure_cf_mo - tenure_orig_mo)
                direction_lbl = "Increase" if abs(cf) > abs(orig) else "Reduce"
                if delta_mo >= 12:
                    orig_str = f"{tenure_orig_mo/12:.1f} yrs"
                    cf_str   = f"{tenure_cf_mo/12:.1f} yrs"
                    d_str    = f"Δ {delta_mo/12:.1f} yrs"
                else:
                    orig_str = f"{tenure_orig_mo:.0f} mo"
                    cf_str   = f"{tenure_cf_mo:.0f} mo"
                    d_str    = f"Δ {delta_mo:.0f} mo"
                step = (
                    f"{direction_lbl} employment tenure from "
                    f"{orig_str} to {cf_str} ({d_str})"
                )
                time_est = get_time_estimate(feat)
                steps.append({
                    "action": step,
                    "feature": feat,
                    "label": label,
                    "original": orig,
                    "cf_value": cf,
                    "delta": delta,
                    "time_estimate": time_est,
                })
                continue

            direction = "Increase" if delta > 0 else "Reduce"

            # Format values sensibly
            if abs(orig) > 1000 or abs(cf) > 1000:
                # Large financial values — show absolute change + pct if meaningful
                if abs(orig) > 1e-3:
                    pct_change = (delta / orig) * 100.0
                    step = (
                        f"{direction} {label} from {orig:,.0f} to {cf:,.0f} "
                        f"({pct_change:+.1f}%)"
                    )
                else:
                    step = f"{direction} {label} from {orig:,.0f} to {cf:,.0f}"
            elif abs(orig) <= 1 or abs(cf) <= 1:
                # Ratios / scores (0–1 range) — always show Δ, not %
                step = (
                    f"{direction} {label} from {orig:.3f} to {cf:.3f} "
                    f"(Δ {delta:+.3f})"
                )
            else:
                if abs(orig) > 1e-3:
                    pct_change = (delta / orig) * 100.0
                    step = (
                        f"{direction} {label} from {orig:.2f} to {cf:.2f} "
                        f"({pct_change:+.1f}%)"
                    )
                else:
                    step = f"{direction} {label} from {orig:.2f} to {cf:.2f}"

            time_est = get_time_estimate(feat)
            steps.append(
                {
                    "action": step,
                    "feature": feat,
                    "label": label,
                    "original": orig,
                    "cf_value": cf,
                    "delta": delta,
                    "time_estimate": time_est,
                }
            )
        return steps

    def _estimate_total_time(self, changes: list) -> str:
        """
        Estimate total time as the maximum among changed features.
        (Steps can be done in parallel, so the bottleneck drives the total.)
        """
        max_months = 0.0
        for change in changes:
            feat = change["feature"]
            tw = get_time_weight(feat)
            # Roughly: effort in std-dev units × time_per_std_dev
            sigma = max(self.feature_stds.get(feat, 1.0), 1e-6)
            stds_changed = abs(change["delta"]) / sigma
            months = stds_changed * tw
            max_months = max(max_months, months)

        if max_months < 1:
            return "~2–4 weeks"
        elif max_months < 2:
            return "~1–2 months"
        elif max_months < 4:
            return "~2–4 months"
        elif max_months < 7:
            return "~4–6 months"
        elif max_months < 13:
            return "~6–12 months"
        else:
            return "~12+ months"


# ─────────────────────────────────────────────────────────────────────────────
# Standalone formatting utility
# ─────────────────────────────────────────────────────────────────────────────

def format_ranked_paths_for_api(ranked_paths: list) -> list:
    """
    Convert EffortRanker output to API-friendly format.
    Strips non-serialisable objects.
    """
    result = []
    for path in ranked_paths:
        result.append(
            {
                "rank": path.get("rank"),
                "flip_probability": path["flip_probability"],
                "effort_score": path["effort_score"],
                "composite_score": path["composite_score"],
                "time_estimate": path["time_estimate"],
                "steps": [
                    {
                        "action": s["action"],
                        "feature": s["feature"],
                        "label": s["label"],
                        "original": round(s["original"], 4),
                        "cf_value": round(s["cf_value"], 4),
                        "delta": round(s["delta"], 4),
                        "time_estimate": s["time_estimate"],
                    }
                    for s in path["steps"]
                ],
            }
        )
    return result
