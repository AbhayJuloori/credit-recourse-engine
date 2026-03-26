"""
Tests for the effort ranker — Layer 4.

These test the core ranking logic independently of any ML model.
"""

import numpy as np
import pandas as pd
import pytest

from backend.ml.effort_ranker import EffortRanker, format_ranked_paths_for_api


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def feature_stds():
    return {
        "EXT_SOURCE_2": 0.1,
        "AMT_CREDIT": 50000.0,
        "AMT_INCOME_TOTAL": 80000.0,
        "CREDIT_INCOME_RATIO": 2.0,
        "DAYS_EMPLOYED": 1000.0,
    }


@pytest.fixture
def ranker(feature_stds):
    return EffortRanker(feature_stds=feature_stds, flip_weight=0.5, cost_weight=0.5)


@pytest.fixture
def mock_model():
    """Fake model that returns fixed probabilities."""
    class _FakeModel:
        def predict_proba(self, X):
            # Return low default probability for counterfactuals
            return np.array([[0.80, 0.20]] * len(X))
    return _FakeModel()


@pytest.fixture
def original_row():
    return pd.DataFrame([{
        "EXT_SOURCE_2": 0.45,
        "AMT_CREDIT": 400000.0,
        "AMT_INCOME_TOTAL": 120000.0,
        "CREDIT_INCOME_RATIO": 3.33,
        "DAYS_EMPLOYED": -1000.0,
    }])


@pytest.fixture
def sample_paths(original_row):
    """Three candidate counterfactual paths."""
    # Path A: small change — high flip chance
    path_a = {
        "cf_index": 0,
        "changes": [
            {"feature": "EXT_SOURCE_2", "original": 0.45, "cf": 0.62, "delta": 0.17},
        ],
        "cf_row": original_row.iloc[0].copy(),
    }
    path_a["cf_row"]["EXT_SOURCE_2"] = 0.62

    # Path B: multiple large changes — lower composite score
    path_b = {
        "cf_index": 1,
        "changes": [
            {"feature": "AMT_INCOME_TOTAL", "original": 120000, "cf": 210000, "delta": 90000},
            {"feature": "AMT_CREDIT", "original": 400000, "cf": 300000, "delta": -100000},
        ],
        "cf_row": original_row.iloc[0].copy(),
    }
    path_b["cf_row"]["AMT_INCOME_TOTAL"] = 210000
    path_b["cf_row"]["AMT_CREDIT"] = 300000

    # Path C: moderate change
    path_c = {
        "cf_index": 2,
        "changes": [
            {"feature": "CREDIT_INCOME_RATIO", "original": 3.33, "cf": 2.5, "delta": -0.83},
        ],
        "cf_row": original_row.iloc[0].copy(),
    }
    path_c["cf_row"]["CREDIT_INCOME_RATIO"] = 2.5

    return [path_a, path_b, path_c]


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestEffortRanker:

    def test_rank_returns_list(self, ranker, sample_paths, mock_model, original_row):
        result = ranker.rank(sample_paths, mock_model, original_row, top_k=3)
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_rank_top_k_limit(self, ranker, sample_paths, mock_model, original_row):
        result = ranker.rank(sample_paths, mock_model, original_row, top_k=2)
        assert len(result) <= 2

    def test_rank_indices_are_sequential(self, ranker, sample_paths, mock_model, original_row):
        result = ranker.rank(sample_paths, mock_model, original_row, top_k=3)
        for i, path in enumerate(result, start=1):
            assert path["rank"] == i

    def test_composite_score_is_between_0_and_1(self, ranker, sample_paths, mock_model, original_row):
        result = ranker.rank(sample_paths, mock_model, original_row, top_k=3)
        for path in result:
            assert 0.0 <= path["composite_score"] <= 1.0

    def test_paths_sorted_by_composite_score_descending(self, ranker, sample_paths, mock_model, original_row):
        result = ranker.rank(sample_paths, mock_model, original_row, top_k=3)
        scores = [p["composite_score"] for p in result]
        assert scores == sorted(scores, reverse=True)

    def test_small_change_ranks_higher_than_large_change(self, ranker, mock_model, original_row, feature_stds):
        """A small credit score improvement should beat a huge income change."""
        easy_path = {
            "cf_index": 0,
            "changes": [{"feature": "EXT_SOURCE_2", "original": 0.45, "cf": 0.50, "delta": 0.05}],
            "cf_row": original_row.iloc[0].copy(),
        }
        hard_path = {
            "cf_index": 1,
            "changes": [{"feature": "AMT_INCOME_TOTAL", "original": 120000, "cf": 300000, "delta": 180000}],
            "cf_row": original_row.iloc[0].copy(),
        }
        result = ranker.rank([easy_path, hard_path], mock_model, original_row, top_k=2)
        # easy_path should be rank 1 (smaller effort for same flip prob)
        assert result[0]["effort_score"] <= result[1]["effort_score"]

    def test_empty_paths_returns_empty(self, ranker, mock_model, original_row):
        result = ranker.rank([], mock_model, original_row)
        assert result == []

    def test_path_has_required_keys(self, ranker, sample_paths, mock_model, original_row):
        result = ranker.rank(sample_paths, mock_model, original_row, top_k=1)
        assert len(result) > 0
        required = {"rank", "flip_probability", "effort_score", "composite_score",
                    "time_estimate", "steps"}
        assert required.issubset(result[0].keys())

    def test_steps_have_action_strings(self, ranker, sample_paths, mock_model, original_row):
        result = ranker.rank(sample_paths, mock_model, original_row, top_k=3)
        for path in result:
            assert len(path["steps"]) > 0
            for step in path["steps"]:
                assert isinstance(step["action"], str)
                assert len(step["action"]) > 10

    def test_compute_effort_zero_delta(self, ranker):
        effort = ranker._compute_effort([
            {"feature": "EXT_SOURCE_2", "original": 0.45, "cf": 0.45, "delta": 0.0}
        ])
        assert effort == pytest.approx(0.0, abs=1e-6)

    def test_compute_effort_scales_with_delta(self, ranker):
        effort_small = ranker._compute_effort([
            {"feature": "EXT_SOURCE_2", "original": 0.45, "cf": 0.50, "delta": 0.05}
        ])
        effort_large = ranker._compute_effort([
            {"feature": "EXT_SOURCE_2", "original": 0.45, "cf": 0.65, "delta": 0.20}
        ])
        assert effort_large > effort_small

    def test_time_estimate_string_format(self, ranker, sample_paths, mock_model, original_row):
        result = ranker.rank(sample_paths, mock_model, original_row)
        for path in result:
            assert isinstance(path["time_estimate"], str)
            assert len(path["time_estimate"]) > 0


class TestFormatForApi:

    def test_format_removes_non_serialisable(self, ranker, sample_paths, mock_model, original_row):
        ranked = ranker.rank(sample_paths, mock_model, original_row)
        formatted = format_ranked_paths_for_api(ranked)
        import json
        # Should serialise without error
        json.dumps(formatted)

    def test_format_preserves_rank(self, ranker, sample_paths, mock_model, original_row):
        ranked = ranker.rank(sample_paths, mock_model, original_row)
        formatted = format_ranked_paths_for_api(ranked)
        for i, item in enumerate(formatted, start=1):
            assert item["rank"] == i
