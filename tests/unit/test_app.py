"""Unit tests for FastAPI app — uses TestClient, no real server."""
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def _mock_model_info(ticker: str = "AAPL") -> dict:
    return {
        "ticker":       ticker,
        "best_model":   "random_forest",
        "model_path":   f"models/registry/{ticker}/random_forest.joblib",
        "val_metrics":  {"rmse": 27.74, "r2": -0.25, "mae": 20.09, "mape": 8.87},
        "test_metrics": {"rmse": 55.13, "r2": -3.15, "mae": 48.30, "mape": 19.25},
        "all_results":  [],
    }


@pytest.fixture
def client():
    with patch("src.serving.predictor.list_available_tickers", return_value=["AAPL", "MSFT"]):
        from src.serving.app import app
        with TestClient(app) as c:
            yield c


def test_health_returns_ok(client):
    with patch("src.serving.app.list_available_tickers", return_value=["AAPL", "MSFT"]):
        resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"]  == "ok"
    assert data["version"] == "1.0.0"
    assert "AAPL" in data["tickers_available"]


def test_list_models_returns_all(client):
    with patch("src.serving.app.list_available_tickers", return_value=["AAPL", "MSFT"]), \
         patch("src.serving.app.load_best_model_info", side_effect=_mock_model_info):
        resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[0]["best_model"] == "random_forest"


def test_get_model_found(client):
    with patch("src.serving.app.load_best_model_info", return_value=_mock_model_info("AAPL")):
        resp = client.get("/models/AAPL")
    assert resp.status_code == 200
    assert resp.json()["ticker"] == "AAPL"


def test_get_model_not_found(client):
    with patch("src.serving.app.load_best_model_info", side_effect=FileNotFoundError("not found")):
        resp = client.get("/models/FAKE")
    assert resp.status_code == 404


def test_predict_success(client):
    mock_result = {
        "ticker":            "AAPL",
        "predicted_close":   185.42,
        "model_used":        "random_forest",
        "prediction_date":   "2026-04-17",
        "val_rmse":          27.74,
        "val_r2":            -0.25,
        "features_used":     40,
        "inference_time_ms": 12.5,
    }
    with patch("src.serving.app.list_available_tickers", return_value=["AAPL"]), \
         patch("src.serving.app.run_inference", return_value=mock_result):
        resp = client.post("/predict/AAPL", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"]          == "AAPL"
    assert data["predicted_close"] == 185.42
    assert data["model_used"]      == "random_forest"


def test_predict_ticker_not_found(client):
    with patch("src.serving.app.list_available_tickers", return_value=["AAPL"]):
        resp = client.post("/predict/FAKE", json={})
    assert resp.status_code == 404


def test_predict_uppercase_normalization(client):
    mock_result = {
        "ticker":            "AAPL",
        "predicted_close":   185.42,
        "model_used":        "random_forest",
        "prediction_date":   "2026-04-17",
        "val_rmse":          27.74,
        "val_r2":            -0.25,
        "features_used":     40,
        "inference_time_ms": 12.5,
    }
    with patch("src.serving.app.list_available_tickers", return_value=["AAPL"]), \
         patch("src.serving.app.run_inference", return_value=mock_result):
        resp = client.post("/predict/aapl", json={})   # lowercase
    assert resp.status_code == 200
    assert resp.json()["ticker"] == "AAPL"
