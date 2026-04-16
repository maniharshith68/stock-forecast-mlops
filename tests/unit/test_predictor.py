"""Unit tests for predictor — fully mocked, zero network, zero disk."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_feature_df(rows: int = 100) -> pd.DataFrame:
    np.random.seed(0)
    idx  = pd.date_range("2024-01-01", periods=rows, freq="B")
    cols = {f"f{i}": np.random.randn(rows) for i in range(40)}
    cols.update({
        "Ticker":             "AAPL",
        "target_next_close":  np.random.rand(rows) * 100 + 150,
        "target_next_return": np.random.randn(rows) * 0.01,
    })
    return pd.DataFrame(cols, index=idx)


def _make_ohlcv_df(rows: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="B")
    return pd.DataFrame({
        "Open":   np.random.rand(rows) * 10 + 150,
        "High":   np.random.rand(rows) * 10 + 155,
        "Low":    np.random.rand(rows) * 10 + 145,
        "Close":  np.random.rand(rows) * 10 + 150,
        "Volume": np.random.randint(1_000_000, 5_000_000, rows).astype(float),
    }, index=idx)


# ── list_available_tickers ────────────────────────────────────────────────────

def test_list_available_tickers_empty(tmp_path):
    with patch("src.serving.predictor.REGISTRY_DIR", tmp_path):
        from src.serving.predictor import list_available_tickers
        assert list_available_tickers() == []


def test_list_available_tickers_finds_tickers(tmp_path):
    for ticker in ["AAPL", "MSFT"]:
        d = tmp_path / ticker
        d.mkdir()
        (d / "best_model.json").write_text("{}")
    with patch("src.serving.predictor.REGISTRY_DIR", tmp_path):
        from src.serving.predictor import list_available_tickers
        result = list_available_tickers()
    assert set(result) == {"AAPL", "MSFT"}


# ── load_best_model_info ──────────────────────────────────────────────────────

def test_load_best_model_info_success(tmp_path):
    import json
    info = {"best_model": "random_forest", "val_metrics": {"rmse": 1.0}}
    d    = tmp_path / "AAPL"
    d.mkdir()
    (d / "best_model.json").write_text(json.dumps(info))
    with patch("src.serving.predictor.REGISTRY_DIR", tmp_path):
        from src.serving.predictor import load_best_model_info
        result = load_best_model_info("AAPL")
    assert result["best_model"] == "random_forest"


def test_load_best_model_info_missing_raises(tmp_path):
    with patch("src.serving.predictor.REGISTRY_DIR", tmp_path):
        from src.serving.predictor import load_best_model_info
        with pytest.raises(FileNotFoundError):
            load_best_model_info("NONEXISTENT")


# ── fetch_and_engineer_features ──────────────────────────────────────────────

@patch("src.serving.predictor.joblib.load")
@patch("src.serving.predictor.compute_features")
@patch("src.serving.predictor.yf.download")
def test_fetch_and_engineer_features(mock_yf_download, mock_compute, mock_joblib, tmp_path):
    # Mock yfinance download
    mock_yf_download.return_value = _make_ohlcv_df()

    # Mock feature engineering
    feat_df = _make_feature_df()
    mock_compute.return_value = feat_df

    # Mock scaler
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.random.randn(1, 40)
    mock_joblib.return_value = mock_scaler

    # Create fake scaler file
    scaler_dir = tmp_path / "AAPL"
    scaler_dir.mkdir()
    (scaler_dir / "scaler.joblib").write_bytes(b"fake")

    with patch("src.serving.predictor.PROCESSED_DIR", tmp_path):
        from src.serving.predictor import fetch_and_engineer_features
        features, n_features = fetch_and_engineer_features("AAPL", lookback_days=90)

    assert features.shape == (1, 40)
    assert n_features == 40


# ── run_inference ─────────────────────────────────────────────────────────────

@patch("src.serving.predictor.fetch_and_engineer_features")
@patch("src.serving.predictor.load_best_model_info")
@patch("src.serving.predictor.MODEL_LOADERS")
def test_run_inference_random_forest(mock_loaders, mock_info, mock_features):
    mock_info.return_value = {
        "best_model": "random_forest",
        "model_path": "/tmp/rf.joblib",
        "val_metrics": {"rmse": 27.7, "r2": -0.25},
    }
    mock_features.return_value = (np.random.randn(1, 40), 40)

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([185.42])
    mock_loaders.__getitem__.return_value = lambda path: mock_model
    mock_loaders.__contains__ = lambda self, key: True

    from src.serving.predictor import run_inference
    result = run_inference("AAPL")

    assert result["ticker"]          == "AAPL"
    assert result["model_used"]      == "random_forest"
    assert isinstance(result["predicted_close"],   float)
    assert isinstance(result["inference_time_ms"], float)
    assert result["features_used"]   == 40
