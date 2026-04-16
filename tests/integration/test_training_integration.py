"""
Phase 3 Integration Tests.

Strategy:
  - RF and XGBoost: real AAPL data, MLflow fully mocked (no DB writes)
  - LSTM: tiny synthetic data (fast, tests full loop end-to-end)
  - Pipeline: RF + XGBoost on AAPL, MLflow mocked, verifies best_model.json

All tests complete in under 60 seconds.
No MLflow server required. No real MLflow DB writes.
"""
import json
import numpy as np
import pandas as pd
import torch
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Helper: fully mocked MLflow ──────────────────────────────────────────────

def _make_mock_mlflow():
    """
    Returns a MagicMock that fully replaces the mlflow module.
    - start_run() returns a context manager that yields a fake run object
    - All log_* calls are no-ops
    - No DB, no HTTP, no filesystem writes
    """
    mock_mf  = MagicMock()
    fake_run = MagicMock()
    fake_run.__enter__ = MagicMock(
        return_value=MagicMock(info=MagicMock(run_id="test-run-id"))
    )
    fake_run.__exit__ = MagicMock(return_value=False)
    mock_mf.start_run.return_value = fake_run
    return mock_mf


# ── RF ───────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_rf_trains_on_real_aapl_data(tmp_path):
    with patch("src.training.random_forest.mlflow", _make_mock_mlflow()), \
         patch("src.training.random_forest.REGISTRY_DIR", tmp_path):
        from src.training.random_forest import train_random_forest
        result = train_random_forest("AAPL")

    assert result["model"] == "random_forest"
    assert Path(result["model_path"]).exists()
    for v in result["val_metrics"].values():
        assert np.isfinite(v), f"Non-finite metric: {v}"
    assert result["val_metrics"]["rmse"] > 0


# ── XGBoost ──────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_xgb_trains_on_real_aapl_data(tmp_path):
    with patch("src.training.xgboost_model.mlflow", _make_mock_mlflow()), \
         patch("src.training.xgboost_model.REGISTRY_DIR", tmp_path):
        from src.training.xgboost_model import train_xgboost
        result = train_xgboost("AAPL")

    assert result["model"] == "xgboost"
    assert Path(result["model_path"]).exists()
    for v in result["val_metrics"].values():
        assert np.isfinite(v), f"Non-finite metric: {v}"
    assert result["val_metrics"]["rmse"] > 0


# ── LSTM ─────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_lstm_full_loop_synthetic(tmp_path):
    """
    Trains LSTM for 3 epochs on 30 synthetic samples.
    Verifies: batching, forward pass, backward pass, evaluation,
    model save, and model reload all work correctly.
    Completes in under 5 seconds.
    """
    from src.training.lstm_model import (
        StockLSTM, _make_batches, _evaluate, load_lstm_model,
    )
    from src.training.evaluator import compute_metrics

    np.random.seed(42)
    input_size, seq_len, n_samples = 8, 5, 30
    X = np.random.randn(n_samples, seq_len, input_size).astype(np.float32)
    y = (np.random.rand(n_samples) * 100 + 50).astype(np.float32)

    model     = StockLSTM(input_size=input_size, hidden_size=16, num_layers=1, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    # Train 3 epochs
    for epoch in range(3):
        model.train()
        for X_b, y_b in _make_batches(X, y, batch_size=8, shuffle=True):
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()

    # Evaluate
    loss_val, preds, targets = _evaluate(model, X, y, batch_size=8, criterion=criterion)
    metrics = compute_metrics(targets, preds, "LSTM/synthetic/test")

    assert np.isfinite(metrics.mae)
    assert np.isfinite(metrics.rmse)
    assert preds.shape == (n_samples,)

    # Save
    model_path = tmp_path / "lstm.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_size":  input_size, "hidden_size": 16,
            "num_layers":  1,          "dropout":     0.0,
        },
    }, model_path)
    assert model_path.exists()

    # Reload and verify
    loaded, device = load_lstm_model(model_path)
    assert loaded is not None
    test_input = torch.randn(1, seq_len, input_size)
    with torch.no_grad():
        out = loaded(test_input)
    assert out.shape == (1,) and torch.isfinite(out).all()


# ── Pipeline ─────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_pipeline_rf_xgb_writes_best_model_json(tmp_path):
    """
    Runs the full pipeline with RF + XGBoost on AAPL.
    Verifies best_model.json is written with correct structure.
    MLflow fully mocked — no DB writes.
    """
    mock_mf = _make_mock_mlflow()

    with patch("src.training.random_forest.mlflow",     mock_mf), \
         patch("src.training.xgboost_model.mlflow",     mock_mf), \
         patch("src.training.training_pipeline.mlflow", mock_mf), \
         patch("src.training.random_forest.REGISTRY_DIR",     tmp_path), \
         patch("src.training.xgboost_model.REGISTRY_DIR",     tmp_path), \
         patch("src.training.training_pipeline.REGISTRY_DIR", tmp_path):

        from src.training.training_pipeline import run_training_pipeline
        summary = run_training_pipeline(
            tickers=["AAPL"],
            models=["random_forest", "xgboost"],
            mlflow_tracking_uri="sqlite:///:memory:",
        )

    # Summary assertions
    assert "AAPL" in summary["results"], "AAPL missing from results"
    assert summary["failed"] == [],      "Unexpected failures"

    best = summary["results"]["AAPL"]
    assert best["best_model"] in {"random_forest", "xgboost"}
    assert best["val_metrics"]["rmse"] > 0

    # best_model.json assertions
    best_json = tmp_path / "AAPL" / "best_model.json"
    assert best_json.exists(), "best_model.json not written"

    data = json.loads(best_json.read_text())
    assert "best_model"   in data
    assert "val_metrics"  in data
    assert "test_metrics" in data
    assert "all_results"  in data
    assert len(data["all_results"]) == 2   # RF + XGBoost
