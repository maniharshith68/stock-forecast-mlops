import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def _make_df(rows: int = 150, n_feats: int = 40) -> pd.DataFrame:
    np.random.seed(1)
    idx  = pd.date_range("2020-01-01", periods=rows, freq="B")
    data = {f"f{i}": np.random.randn(rows) for i in range(n_feats)}
    data.update({
        "Ticker":             "TEST",
        "target_next_close":  np.random.rand(rows) * 100 + 50,
        "target_next_return": np.random.randn(rows) * 0.01,
    })
    return pd.DataFrame(data, index=idx)


def _mock_mlflow():
    m   = MagicMock()
    run = MagicMock()
    run.__enter__ = MagicMock(
        return_value=MagicMock(info=MagicMock(run_id="unit-test-run"))
    )
    run.__exit__ = MagicMock(return_value=False)
    m.start_run.return_value = run
    return m


@patch("src.training.xgboost_model.mlflow", _mock_mlflow())
@patch("src.training.xgboost_model.load_tabular_data")
def test_returns_expected_keys(mock_load):
    df = _make_df()
    mock_load.return_value = (df.iloc[:100], df.iloc[100:125], df.iloc[125:])
    with patch("src.training.xgboost_model.REGISTRY_DIR", Path("/tmp/test_xgb")):
        from src.training.xgboost_model import train_xgboost
        result = train_xgboost("TEST")
    assert result["model"] == "xgboost"
    assert {"model", "val_metrics", "test_metrics", "model_path", "run_id"}.issubset(
        result.keys()
    )


@patch("src.training.xgboost_model.mlflow", _mock_mlflow())
@patch("src.training.xgboost_model.load_tabular_data")
def test_val_metrics_finite(mock_load):
    df = _make_df()
    mock_load.return_value = (df.iloc[:100], df.iloc[100:125], df.iloc[125:])
    with patch("src.training.xgboost_model.REGISTRY_DIR", Path("/tmp/test_xgb")):
        from src.training.xgboost_model import train_xgboost
        result = train_xgboost("TEST")
    for v in result["val_metrics"].values():
        assert np.isfinite(v), f"Non-finite metric: {v}"
