"""
Phase 4 Integration Tests.
Uses real models from models/registry/ and real yFinance data.
FastAPI TestClient — no server process needed.
"""
import pytest
import numpy as np
from unittest.mock import patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.serving.app import app
    with TestClient(app) as c:
        yield c


@pytest.mark.integration
def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert len(data["tickers_available"]) >= 1
    assert "AAPL" in data["tickers_available"]


@pytest.mark.integration
def test_list_models_endpoint(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    models = resp.json()
    assert len(models) == 5
    tickers = {m["ticker"] for m in models}
    assert tickers == {"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"}
    for m in models:
        assert m["best_model"] in {"random_forest", "xgboost", "lstm"}
        assert m["val_metrics"]["rmse"] > 0


@pytest.mark.integration
def test_get_model_aapl(client):
    resp = client.get("/models/AAPL")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"]     == "AAPL"
    assert data["best_model"] == "random_forest"
    assert data["val_metrics"]["rmse"] > 0


@pytest.mark.integration
def test_get_model_lowercase(client):
    """Ticker normalization to uppercase."""
    resp = client.get("/models/aapl")
    assert resp.status_code == 200
    assert resp.json()["ticker"] == "AAPL"


@pytest.mark.integration
def test_get_model_not_found(client):
    resp = client.get("/models/FAKE")
    assert resp.status_code == 404


@pytest.mark.integration
def test_predict_aapl(client):
    """Full inference pipeline with real model and real yFinance data."""
    resp = client.post("/predict/AAPL", json={"lookback_days": 90})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"]       == "AAPL"
    assert data["model_used"]   == "random_forest"
    assert 50.0 < data["predicted_close"] < 2000.0   # sanity range for AAPL
    assert data["features_used"] == 40
    assert data["inference_time_ms"] > 0
    assert data["val_rmse"] > 0


@pytest.mark.integration
def test_predict_all_tickers(client):
    """All 5 tickers should return valid predictions."""
    for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]:
        resp = client.post(f"/predict/{ticker}", json={"lookback_days": 90})
        assert resp.status_code == 200, f"{ticker} failed: {resp.text}"
        data = resp.json()
        assert np.isfinite(data["predicted_close"]), f"{ticker} predicted nan/inf"
        assert data["predicted_close"] > 0
