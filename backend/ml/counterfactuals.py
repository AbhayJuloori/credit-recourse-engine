"""
Feasibility-constrained counterfactual generation using DiCE-ML.

Generates recourse paths for denied/grey-zone applicants:
  "If you change X by Y, your application would flip to Approved."

Design choices:
  - Genetic algorithm backend: more diverse paths than gradient-based,
    works natively with XGBoost (no differentiable model needed).
  - Immutable features locked via DiCE's features_to_vary param.
  - Bounded features constrained via permitted_range.
  - Training sample (2k rows) used for DiCE's distribution estimation —
    small enough to load fast, large enough for diversity.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CounterfactualGenerator:
    """
    Wraps DiCE-ML for counterfactual generation with feasibility constraints.

    Usage:
        gen = CounterfactualGenerator()
        gen.setup(
            model=fitted_xgb,
            training_sample=df_sample,
            continuous_features=[...],
            outcome_col='TARGET',
        )
        cfs = gen.generate(instance_df, num_cfs=5)

        gen.save(path)
        gen2 = CounterfactualGenerator.load(path)
    """

    def __init__(self):
        self.explainer = None
        self.dice_data = None
        self.dice_model = None
        self.continuous_features: Optional[list] = None
        self.all_features: Optional[list] = None
        self.outcome_col: str = "TARGET"
        self.feature_min: Optional[dict] = None
        self.feature_max: Optional[dict] = None

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(
        self,
        model,
        training_sample: pd.DataFrame,
        continuous_features: list,
        outcome_col: str = "TARGET",
    ) -> None:
        """
        Initialise DiCE data + model objects from a training sample.

        Args:
            model: fitted XGBClassifier (sklearn API).
            training_sample: small DataFrame (1k-5k rows) with feature cols
                             + outcome_col. Used for distribution estimation.
            continuous_features: list of continuous feature names.
            outcome_col: target column name.
        """
        try:
            import dice_ml
        except ImportError:
            raise ImportError("dice-ml not installed. Run: pip install dice-ml")

        self.outcome_col = outcome_col
        self.continuous_features = continuous_features
        self.all_features = [c for c in training_sample.columns if c != outcome_col]

        # Fill NaN in training sample — DiCE rejects NaNs at generation time.
        # Use 0.0 for supplementary features that are NaN for some applicants.
        training_sample = training_sample.copy()
        training_sample[self.all_features] = training_sample[self.all_features].fillna(0.0)
        training_sample = training_sample.replace([float("inf"), float("-inf")], 0.0)

        # Feature min/max from training sample for bounded constraints
        num_cols = training_sample[self.all_features].select_dtypes("number").columns
        self.feature_min = training_sample[num_cols].min().to_dict()
        self.feature_max = training_sample[num_cols].max().to_dict()

        # DiCE data object
        self.dice_data = dice_ml.Data(
            dataframe=training_sample,
            continuous_features=continuous_features,
            outcome_name=outcome_col,
        )

        # DiCE model wrapper
        self.dice_model = dice_ml.Model(model=model, backend="sklearn")

        # Genetic algorithm explainer
        self.explainer = dice_ml.Dice(self.dice_data, self.dice_model, method="genetic")
        logger.info("DiCE explainer initialised with genetic backend.")

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(
        self,
        instance: pd.DataFrame,
        num_cfs: int = 6,
        desired_class: int = 0,
        mutable_features: Optional[list] = None,
        permitted_range: Optional[dict] = None,
    ) -> list:
        """
        Generate counterfactuals for a single applicant.

        Args:
            instance: single-row DataFrame with all features (no TARGET col).
            num_cfs: number of counterfactuals to attempt.
            desired_class: 0 = flip to Approved (default goal).
            mutable_features: features that can be changed.
                              If None, all non-immutable features are used.
            permitted_range: dict of {feature: [min, max]} constraints.

        Returns:
            List of dicts, each representing one counterfactual path.
            Returns empty list if DiCE fails (graceful degradation).
        """
        assert self.explainer is not None, "Call .setup() first."

        from backend.ml.constraints import (
            RECOURSE_DIRECTION,
            build_permitted_range,
            get_recourse_features,
        )

        if mutable_features is None:
            mutable_features = [
                f for f in get_recourse_features(self.all_features)
                if f in instance.columns
            ]

        if permitted_range is None and self.feature_min:
            instance_dict = instance.iloc[0].to_dict()
            permitted_range = build_permitted_range(
                instance_dict, self.feature_min, self.feature_max
            )

        try:
            # Fill NaN values — some supplementary features may be all-NaN
            # for an applicant; 0.0 is a safe fallback since we're using
            # continuous-only mode in DiCE.
            instance = instance.fillna(0.0)

            # Snap categorical features to nearest valid training value.
            # Median imputation can produce fractional values (e.g. 0.5 for a
            # 0/1 flag) that DiCE rejects as "outside the dataset."
            instance = self._snap_categoricals(instance)

            dice_exp = self.explainer.generate_counterfactuals(
                query_instances=instance,
                total_CFs=num_cfs,
                desired_class=desired_class,
                features_to_vary=mutable_features,
                permitted_range=permitted_range or {},
            )
            return self._parse_dice_output(
                dice_exp, instance, mutable_features, RECOURSE_DIRECTION
            )
        except Exception as exc:
            logger.warning(f"DiCE generation failed: {exc}")
            return []

    # ── Categorical snapping ──────────────────────────────────────────────────

    def _snap_categoricals(self, instance: pd.DataFrame) -> pd.DataFrame:
        """
        For each feature DiCE treats as categorical (not in continuous_features),
        snap the instance value to the nearest allowed value from the training data.

        This is necessary because median imputation can produce fractional values
        (e.g. 0.5 for a binary 0/1 flag) that DiCE rejects as invalid categories.
        """
        if self.dice_data is None:
            return instance

        instance = instance.copy()
        dice_df = self.dice_data.data_df
        continuous = set(self.continuous_features or [])

        for col in instance.columns:
            if col in continuous or col not in dice_df.columns:
                continue
            allowed_raw = dice_df[col].dropna().unique()
            try:
                allowed_floats = [float(v) for v in allowed_raw]
            except (ValueError, TypeError):
                continue
            val = float(instance[col].iloc[0])
            if val not in allowed_floats:
                nearest = min(allowed_floats, key=lambda x: abs(x - val))
                instance.at[instance.index[0], col] = nearest

        return instance

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_dice_output(
        self,
        dice_exp,
        original: pd.DataFrame,
        features_to_report: list = None,
        direction_constraints: dict = None,
    ) -> list:
        """
        Convert DiCE output to a list of change dicts.

        Only reports changes for features in `features_to_report` (the
        recourse-eligible set). DiCE's genetic backend introduces floating-
        point noise in all feature columns; restricting the report list keeps
        paths clean and interpretable.

        `direction_constraints` (from RECOURSE_DIRECTION) filters out changes
        that go in the wrong direction (e.g., dropping a credit score). The
        genetic algorithm doesn't strictly enforce permitted_range, so this
        post-filter is necessary to prevent counterintuitive suggestions.

        Uses a minimum absolute threshold of 1e-3 to filter noise.

        Returns:
            [
              {
                'cf_index': 0,
                'changes': [
                  {'feature': 'EXT_SOURCE_2', 'original': 0.45, 'cf': 0.61, 'delta': 0.16},
                  ...
                ],
                'cf_row': pd.Series  # full counterfactual row
              },
              ...
            ]
        """
        paths = []
        try:
            cf_df = dice_exp.cf_examples_list[0].final_cfs_df
            if cf_df is None or len(cf_df) == 0:
                return []
        except (IndexError, AttributeError):
            return []

        orig_row = original.iloc[0]
        # Restrict reporting to explicitly varied features; fall back to all
        report_set = set(features_to_report) if features_to_report else set(original.columns)
        directions = direction_constraints or {}

        for idx, cf_row in cf_df.iterrows():
            changes = []
            for feat in original.columns:
                if feat not in report_set or feat not in cf_row.index:
                    continue
                orig_val = orig_row[feat]
                cf_val = cf_row[feat]
                if pd.isna(orig_val) or pd.isna(cf_val):
                    continue
                delta = cf_val - orig_val
                # Use 1e-3 threshold to suppress floating-point noise
                if abs(delta) <= 1e-3:
                    continue
                # Drop changes that violate the known beneficial direction
                if feat in directions:
                    if directions[feat] == "increase" and delta < 0:
                        continue  # shouldn't decrease this feature
                    if directions[feat] == "decrease" and delta > 0:
                        continue  # shouldn't increase this feature
                changes.append(
                    {
                        "feature": feat,
                        "original": float(orig_val),
                        "cf": float(cf_val),
                        "delta": float(delta),
                    }
                )
            if changes:
                paths.append(
                    {
                        "cf_index": len(paths),
                        "changes": changes,
                        "cf_row": cf_row,
                    }
                )

        return paths

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        payload = {
            "dice_data": self.dice_data,
            "dice_model": self.dice_model,
            "continuous_features": self.continuous_features,
            "all_features": self.all_features,
            "outcome_col": self.outcome_col,
            "feature_min": self.feature_min,
            "feature_max": self.feature_max,
        }
        joblib.dump(payload, path)
        logger.info(f"CounterfactualGenerator saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "CounterfactualGenerator":
        import dice_ml
        payload = joblib.load(path)
        gen = cls()
        gen.dice_data = payload["dice_data"]
        gen.dice_model = payload["dice_model"]
        gen.continuous_features = payload["continuous_features"]
        gen.all_features = payload["all_features"]
        gen.outcome_col = payload["outcome_col"]
        gen.feature_min = payload["feature_min"]
        gen.feature_max = payload["feature_max"]

        # Rebuild explainer from data + model
        gen.explainer = dice_ml.Dice(gen.dice_data, gen.dice_model, method="genetic")
        logger.info(f"CounterfactualGenerator loaded from {path}")
        return gen
