"""
Feature constraints for counterfactual generation.

Three categories:
  IMMUTABLE  — features the applicant cannot change (age, history, identity)
  BOUNDED    — features that can change but only within business-feasible limits
  MUTABLE    — features that can change freely within [min, max] observed in data

TIME_WEIGHTS — domain-knowledge estimate of months needed to shift each feature
               by 1 standard deviation. Used by the effort ranker.
"""

from typing import Dict, List, Optional, Tuple

# ── Immutable features ────────────────────────────────────────────────────────
# Applicant cannot change these. DiCE will lock them.
IMMUTABLE_FEATURES: List[str] = [
    "DAYS_BIRTH",                  # age — cannot reverse time
    "CODE_GENDER",                 # gender identity
    "NAME_EDUCATION_TYPE",         # completed education (debatable but standard)
    "REGION_RATING_CLIENT",        # regional credit rating — not in applicant's control
    "REGION_RATING_CLIENT_W_CITY", # same
    "ORGANIZATION_TYPE",           # employer type — can change jobs but not same org
    "NAME_FAMILY_STATUS",          # marital status (immutable for now)
    "CNT_CHILDREN",                # children count
    "FLAG_OWN_REALTY",             # property ownership (short term)
    "CNT_FAM_MEMBERS",             # family size — not a recourse lever
    "AGE_YEARS",                   # derived age feature
    "DAYS_EMPLOYED_ANOM",          # anomaly flag — not actionable
    # Aggregated history features — cannot undo past
    "BUREAU_LOAN_COUNT",
    "BUREAU_DAYS_CREDIT_MIN",
    "PREV_APP_COUNT",
]

# ── Bounded features: (min_factor, max_factor) relative to current value ─────
# E.g., (0.7, 1.3) means the feature can change ±30% from current value.
FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "AMT_CREDIT":           (0.70, 1.30),   # loan amount ±30%
    "AMT_ANNUITY":          (0.70, 1.30),   # monthly payment ±30%
    "AMT_GOODS_PRICE":      (0.70, 1.30),   # goods price ±30%
    "AMT_INCOME_TOTAL":     (1.00, 2.00),   # income can only go up realistically
    "DAYS_EMPLOYED":        (1.00, 1.50),   # employment duration can only grow
    "CNT_FAM_MEMBERS":      (0.80, 1.20),   # family size (not a big lever)
}

# ── Absolute range bounds for specific features ───────────────────────────────
FEATURE_RANGE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "EXT_SOURCE_1":         (0.0,  1.0),
    "EXT_SOURCE_2":         (0.0,  1.0),
    "EXT_SOURCE_3":         (0.0,  1.0),
    "CREDIT_INCOME_RATIO":  (0.0, 20.0),
    "ANNUITY_INCOME_RATIO": (0.0,  1.0),
    "DAYS_EMPLOYED_RATIO":  (0.0,  1.0),
    "DEF_30_CNT_SOCIAL_CIRCLE": (0.0, 20.0),
    "DEF_60_CNT_SOCIAL_CIRCLE": (0.0, 20.0),
}

# ── Directional constraints for recourse features ─────────────────────────────
# "increase"  → CF value must be >= current value (good direction is higher)
# "decrease"  → CF value must be <= current value (good direction is lower)
# Features not listed here are free to move in either direction.
RECOURSE_DIRECTION: Dict[str, str] = {
    # Higher credit scores → more likely approved
    "EXT_SOURCE_1":                      "increase",
    "EXT_SOURCE_2":                      "increase",
    "EXT_SOURCE_3":                      "increase",
    # Lower debt burden → better
    "CREDIT_INCOME_RATIO":               "decrease",
    "AMT_CREDIT":                        "decrease",
    "AMT_ANNUITY":                       "decrease",
    # Lower payment-behaviour issues → better
    "INSTAL_INST_DPD_MEAN":              "decrease",
    "INSTAL_INST_DPD_MAX":               "decrease",
    "INSTAL_LATE_RATE":                  "decrease",
    # Lower outstanding overdue → better
    "BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM": "decrease",
    # Lower credit utilization → better
    "CC_CC_LIMIT_USE_RATIO_MEAN":        "decrease",
    # Income and employment can only grow
    "AMT_INCOME_TOTAL":                  "increase",
}

# ── Time weights: months to shift feature by 1 std dev ───────────────────────
# Domain knowledge — loan officers validated these rough estimates.
TIME_WEIGHTS: Dict[str, float] = {
    # Loan parameters — immediate (applicant can just request differently)
    "AMT_CREDIT":                    0.5,
    "AMT_ANNUITY":                   0.5,
    "AMT_GOODS_PRICE":               0.5,

    # Income — slow to increase significantly
    "AMT_INCOME_TOTAL":              9.0,

    # Employment history — grows naturally, can't be rushed
    "DAYS_EMPLOYED":                 4.0,
    "DAYS_EMPLOYED_RATIO":           4.0,

    # External credit scores — takes months of good behaviour
    "EXT_SOURCE_1":                  5.0,
    "EXT_SOURCE_2":                  4.0,
    "EXT_SOURCE_3":                  4.0,
    "EXT_SOURCE_MEAN":               4.5,

    # Derived financial ratios
    "CREDIT_INCOME_RATIO":           6.0,
    "ANNUITY_INCOME_RATIO":          6.0,
    "CREDIT_TERM":                   0.5,
    "INCOME_PER_PERSON":             9.0,

    # Bureau / payment history
    "BUREAU_CREDIT_SUM_DEBT_SUM":    6.0,
    "BUREAU_ACTIVE_COUNT":           3.0,
    "INSTAL_DPD_MEAN":               3.0,
    "CC_LIMIT_USE_RATIO_MEAN":       3.0,
    "CC_PAYMENT_RATIO_MEAN":         2.0,

    # Social circle
    "DEF_30_CNT_SOCIAL_CIRCLE":      6.0,
    "DEF_60_CNT_SOCIAL_CIRCLE":      6.0,

    # Default — if feature not listed
    "_DEFAULT":                       3.0,
}

# ── Human-readable feature descriptions ──────────────────────────────────────
FEATURE_LABELS: Dict[str, str] = {
    "AMT_INCOME_TOTAL":              "Annual income",
    "AMT_CREDIT":                    "Requested loan amount",
    "AMT_ANNUITY":                   "Monthly loan payment",
    "AMT_GOODS_PRICE":               "Goods price",
    "DAYS_EMPLOYED":                 "Employment duration (days)",
    "DAYS_EMPLOYED_RATIO":           "Employment-to-age ratio",
    "EXT_SOURCE_1":                  "External credit score 1",
    "EXT_SOURCE_2":                  "External credit score 2",
    "EXT_SOURCE_3":                  "External credit score 3",
    "EXT_SOURCE_MEAN":               "Mean external credit score",
    "CREDIT_INCOME_RATIO":           "Loan-to-income ratio",
    "ANNUITY_INCOME_RATIO":          "Annuity-to-income ratio",
    "CREDIT_TERM":                   "Loan term (months)",
    "INCOME_PER_PERSON":             "Income per family member",
    "BUREAU_CREDIT_SUM_DEBT_SUM":    "Outstanding bureau debt",
    "BUREAU_ACTIVE_COUNT":           "Number of active credit lines",
    "CC_LIMIT_USE_RATIO_MEAN":       "Credit card utilization",
    "CC_PAYMENT_RATIO_MEAN":         "Credit card payment ratio",
    "INSTAL_DPD_MEAN":               "Avg days past due on installments",
    "DEF_30_CNT_SOCIAL_CIRCLE":      "Defaults in social circle (30d)",
    "DEF_60_CNT_SOCIAL_CIRCLE":      "Defaults in social circle (60d)",
}

# ── Time estimate strings for UI ──────────────────────────────────────────────
TIME_ESTIMATES: Dict[str, str] = {
    "AMT_CREDIT":                    "Can adjust loan request immediately",
    "AMT_ANNUITY":                   "Can adjust repayment terms immediately",
    "AMT_GOODS_PRICE":               "Can adjust immediately",
    "AMT_INCOME_TOTAL":              "~6–12 months",
    "DAYS_EMPLOYED":                 "~3–6 months",
    "DAYS_EMPLOYED_RATIO":           "~3–6 months",
    "EXT_SOURCE_1":                  "~4–6 months of good credit behaviour",
    "EXT_SOURCE_2":                  "~3–5 months of good credit behaviour",
    "EXT_SOURCE_3":                  "~3–5 months of good credit behaviour",
    "EXT_SOURCE_MEAN":               "~3–5 months of good credit behaviour",
    "CREDIT_INCOME_RATIO":           "~3–6 months (income ↑ or loan ↓)",
    "ANNUITY_INCOME_RATIO":          "~3–6 months",
    "CREDIT_TERM":                   "Adjust loan terms immediately",
    "INCOME_PER_PERSON":             "~6–12 months",
    "BUREAU_CREDIT_SUM_DEBT_SUM":    "~6 months of repayments",
    "BUREAU_ACTIVE_COUNT":           "~2–3 months (close unused credit lines)",
    "CC_LIMIT_USE_RATIO_MEAN":       "~2–3 months",
    "CC_PAYMENT_RATIO_MEAN":         "~1–2 months",
    "INSTAL_DPD_MEAN":               "~2–4 months of on-time payments",
    "_DEFAULT":                      "~3 months",
}


# ── Recourse-eligible features ────────────────────────────────────────────────
# Subset of mutable features that are (a) interpretable and (b) actionable.
# DiCE features_to_vary is restricted to this list so paths make sense.
# Expanding this list increases path diversity but risks nonsensical suggestions.
RECOURSE_ELIGIBLE_FEATURES: List[str] = [
    # Core financial levers (immediate: request a different loan)
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_INCOME_TOTAL",
    "CREDIT_INCOME_RATIO",
    "CREDIT_TERM",

    # Employment (grows with time)
    "DAYS_EMPLOYED",

    # External credit scores (highest-impact lever, ~3–6 months)
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",

    # Bureau / credit history (reduce overdue debt, close lines)
    "BUREAU_ACTIVE_COUNT",
    "BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM",

    # Credit card behaviour
    "CC_CC_LIMIT_USE_RATIO_MEAN",

    # Installment payment discipline
    "INSTAL_INST_DPD_MEAN",
    "INSTAL_LATE_RATE",
]


def get_mutable_features(all_features: List[str]) -> List[str]:
    """Return all features that are NOT immutable."""
    return [f for f in all_features if f not in IMMUTABLE_FEATURES]


def get_recourse_features(all_features: List[str]) -> List[str]:
    """Return the interpretable, actionable feature subset for DiCE."""
    return [f for f in RECOURSE_ELIGIBLE_FEATURES if f in all_features]


def get_time_weight(feature: str) -> float:
    """Return months-per-std-dev weight for a feature."""
    return TIME_WEIGHTS.get(feature, TIME_WEIGHTS["_DEFAULT"])


def get_time_estimate(feature: str) -> str:
    """Return human-readable time estimate string for a feature."""
    return TIME_ESTIMATES.get(feature, TIME_ESTIMATES["_DEFAULT"])


def get_feature_label(feature: str) -> str:
    """Return human-readable label for a feature."""
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def build_permitted_range(
    instance: Dict[str, float],
    feature_min: Dict[str, float],
    feature_max: Dict[str, float],
) -> Dict[str, List[float]]:
    """
    Build DiCE-compatible permitted_range dict for bounded features.
    For bounded features: apply relative bounds to current value.
    For range-bounded features: use absolute bounds.
    For mutable features: use training data min/max.
    """
    permitted: Dict[str, List[float]] = {}

    for feat, (lo_factor, hi_factor) in FEATURE_BOUNDS.items():
        if feat in instance and feat in feature_min:
            current = instance[feat]
            lo = max(feature_min[feat], current * lo_factor)
            hi = min(feature_max[feat], current * hi_factor)
            permitted[feat] = [lo, hi]

    for feat, (abs_lo, abs_hi) in FEATURE_RANGE_BOUNDS.items():
        if feat in feature_min:
            permitted[feat] = [abs_lo, abs_hi]

    # Apply directional constraints: lock one end of the range to the
    # current value so DiCE can only move features in their beneficial direction.
    for feat, direction in RECOURSE_DIRECTION.items():
        if feat not in instance or feat not in feature_min:
            continue
        current = instance[feat]
        existing = permitted.get(feat)
        if existing is not None:
            lo, hi = existing
        else:
            lo, hi = feature_min.get(feat, current), feature_max.get(feat, current)
        if direction == "increase":
            lo = max(lo, current)   # can only go up from current value
        elif direction == "decrease":
            hi = min(hi, current)   # can only go down from current value
        # Ensure valid range (lo <= hi); if degenerate, skip
        if lo <= hi:
            permitted[feat] = [lo, hi]

    return permitted
