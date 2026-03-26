"""
Tests for constraints module.
"""

import pytest
from backend.ml.constraints import (
    IMMUTABLE_FEATURES,
    FEATURE_BOUNDS,
    TIME_WEIGHTS,
    build_permitted_range,
    get_feature_label,
    get_mutable_features,
    get_time_estimate,
    get_time_weight,
)


class TestImmutableFeatures:

    def test_immutable_features_is_list(self):
        assert isinstance(IMMUTABLE_FEATURES, list)

    def test_immutable_contains_age(self):
        assert "DAYS_BIRTH" in IMMUTABLE_FEATURES

    def test_immutable_contains_gender(self):
        assert "CODE_GENDER" in IMMUTABLE_FEATURES

    def test_immutable_features_are_strings(self):
        for f in IMMUTABLE_FEATURES:
            assert isinstance(f, str)


class TestMutableFeatures:

    def test_mutable_excludes_immutable(self):
        all_features = ["AMT_CREDIT", "DAYS_BIRTH", "EXT_SOURCE_2", "CODE_GENDER"]
        mutable = get_mutable_features(all_features)
        for imm in IMMUTABLE_FEATURES:
            assert imm not in mutable

    def test_mutable_includes_loan_amount(self):
        all_features = ["AMT_CREDIT", "DAYS_BIRTH", "EXT_SOURCE_2"]
        mutable = get_mutable_features(all_features)
        assert "AMT_CREDIT" in mutable

    def test_mutable_features_not_empty_for_valid_input(self):
        all_features = ["AMT_CREDIT", "EXT_SOURCE_2", "AMT_INCOME_TOTAL"]
        mutable = get_mutable_features(all_features)
        assert len(mutable) > 0


class TestFeatureBounds:

    def test_feature_bounds_are_tuples(self):
        for feat, bounds in FEATURE_BOUNDS.items():
            assert isinstance(bounds, tuple)
            assert len(bounds) == 2

    def test_bounds_lower_less_than_upper(self):
        for feat, (lo, hi) in FEATURE_BOUNDS.items():
            assert lo < hi, f"{feat}: lo ({lo}) should be < hi ({hi})"

    def test_credit_bounds_within_reasonable_range(self):
        lo, hi = FEATURE_BOUNDS["AMT_CREDIT"]
        assert 0 < lo < 1
        assert hi > 1


class TestTimeWeights:

    def test_default_weight_exists(self):
        assert "_DEFAULT" in TIME_WEIGHTS

    def test_weights_are_positive(self):
        for feat, w in TIME_WEIGHTS.items():
            assert w > 0, f"{feat} time weight must be positive"

    def test_income_slower_than_loan_amount(self):
        # Income takes longer to change than adjusting loan amount
        assert TIME_WEIGHTS["AMT_INCOME_TOTAL"] > TIME_WEIGHTS["AMT_CREDIT"]

    def test_ext_source_weight_is_moderate(self):
        # External credit scores take a few months
        assert 2 <= TIME_WEIGHTS["EXT_SOURCE_2"] <= 10

    def test_get_time_weight_returns_default_for_unknown(self):
        w = get_time_weight("SOME_UNKNOWN_FEATURE_XYZ")
        assert w == TIME_WEIGHTS["_DEFAULT"]

    def test_get_time_weight_returns_correct_for_known(self):
        w = get_time_weight("EXT_SOURCE_2")
        assert w == TIME_WEIGHTS["EXT_SOURCE_2"]


class TestGetTimeEstimate:

    def test_returns_string(self):
        est = get_time_estimate("AMT_CREDIT")
        assert isinstance(est, str)

    def test_returns_default_for_unknown(self):
        est = get_time_estimate("UNKNOWN_XYZ")
        assert isinstance(est, str)
        assert len(est) > 0


class TestGetFeatureLabel:

    def test_returns_string(self):
        label = get_feature_label("AMT_INCOME_TOTAL")
        assert isinstance(label, str)

    def test_unknown_feature_returns_title_case(self):
        label = get_feature_label("SOME_FEATURE_NAME")
        assert label == "Some Feature Name"

    def test_known_feature_returns_human_label(self):
        label = get_feature_label("AMT_INCOME_TOTAL")
        assert label == "Annual income"


class TestBuildPermittedRange:

    def test_returns_dict(self):
        instance = {"AMT_CREDIT": 400000.0, "AMT_INCOME_TOTAL": 120000.0}
        feature_min = {"AMT_CREDIT": 100000.0, "AMT_INCOME_TOTAL": 50000.0}
        feature_max = {"AMT_CREDIT": 1000000.0, "AMT_INCOME_TOTAL": 500000.0}
        result = build_permitted_range(instance, feature_min, feature_max)
        assert isinstance(result, dict)

    def test_range_respects_lower_bound_factor(self):
        instance = {"AMT_CREDIT": 400000.0}
        feature_min = {"AMT_CREDIT": 0.0}
        feature_max = {"AMT_CREDIT": 1000000.0}
        result = build_permitted_range(instance, feature_min, feature_max)
        lo, hi = result["AMT_CREDIT"]
        lo_factor, hi_factor = FEATURE_BOUNDS["AMT_CREDIT"]
        assert abs(lo - 400000.0 * lo_factor) < 1.0

    def test_range_values_lo_less_than_hi(self):
        instance = {"AMT_CREDIT": 400000.0}
        feature_min = {"AMT_CREDIT": 0.0}
        feature_max = {"AMT_CREDIT": 1000000.0}
        result = build_permitted_range(instance, feature_min, feature_max)
        if "AMT_CREDIT" in result:
            lo, hi = result["AMT_CREDIT"]
            assert lo <= hi
