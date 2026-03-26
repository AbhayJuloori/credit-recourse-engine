"""
API endpoint tests using TestClient.
These run without real ML artifacts (model_loaded=False scenario).
"""

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app


@pytest.fixture
def client():
    """TestClient with lifespan — models will be None (artifacts not trained yet)."""
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        res = client.get("/health")
        assert res.status_code == 200

    def test_health_has_status_key(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_health_has_model_loaded_key(self, client):
        data = client.get("/health").json()
        assert "model_loaded" in data

    def test_health_has_grey_zone_key(self, client):
        data = client.get("/health").json()
        assert "grey_zone_loaded" in data


class TestPredictEndpoint:

    def test_predict_returns_503_without_model(self, client):
        """When model artifacts are not present, should return 503."""
        payload = {
            "AMT_INCOME_TOTAL": 100000,
            "AMT_CREDIT": 400000,
            "AMT_ANNUITY": 20000,
            "DAYS_BIRTH": -12000,
        }
        res = client.post("/api/predict", json=payload)
        # 503 if not trained, 200 if artifacts exist
        assert res.status_code in [200, 503]

    def test_predict_missing_required_fields_returns_422(self, client):
        """Missing required fields should return 422 (validation error)."""
        payload = {"AMT_INCOME_TOTAL": 100000}   # missing AMT_CREDIT, etc.
        res = client.post("/api/predict", json=payload)
        assert res.status_code == 422

    def test_predict_negative_income_returns_422(self, client):
        payload = {
            "AMT_INCOME_TOTAL": -100,   # invalid: gt=0
            "AMT_CREDIT": 400000,
            "AMT_ANNUITY": 20000,
            "DAYS_BIRTH": -12000,
        }
        res = client.post("/api/predict", json=payload)
        assert res.status_code == 422


class TestRecourseEndpoint:

    def test_recourse_returns_503_or_200(self, client):
        payload = {
            "applicant": {
                "AMT_INCOME_TOTAL": 100000,
                "AMT_CREDIT": 400000,
                "AMT_ANNUITY": 20000,
                "DAYS_BIRTH": -12000,
            }
        }
        res = client.post("/api/recourse", json=payload)
        assert res.status_code in [200, 503]

    def test_recourse_invalid_ext_source_returns_422(self, client):
        payload = {
            "applicant": {
                "AMT_INCOME_TOTAL": 100000,
                "AMT_CREDIT": 400000,
                "AMT_ANNUITY": 20000,
                "DAYS_BIRTH": -12000,
                "EXT_SOURCE_1": 5.0,  # out of [0, 1] range
            }
        }
        res = client.post("/api/recourse", json=payload)
        assert res.status_code == 422


class TestFrontend:

    def test_root_returns_html(self, client):
        res = client.get("/")
        assert res.status_code in [200, 404]   # 404 if templates not built yet
        if res.status_code == 200:
            assert "text/html" in res.headers.get("content-type", "")
