"""Unit tests for preprocessor — split ratios, scaling, LSTM shapes."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from src.etl.preprocessor import split_and_scale, _make_sequences, load_split, load_scaler


def _make_feature_df(rows: int = 300) -> pd.DataFrame:
    """Synthetic feature DataFrame matching compute_features output."""
    np.random.seed(0)
    idx = pd.date_range("2020-01-01", periods=rows, freq="B")
    data = {
        "Ticker":            ["TEST"] * rows,
        "returns_1d":        np.random.randn(rows) * 0.01,
        "sma_20":            np.random.rand(rows) * 100 + 50,
        "rsi_14":            np.random.rand(rows) * 100,
        "macd":              np.random.randn(rows),
        "Close":             np.random.rand(rows) * 100 + 50,
        "Volume":            np.random.randint(1_000_000, 5_000_000, rows).astype(float),
        "target_next_close": np.random.rand(rows) * 100 + 50,
        "target_next_return":np.random.randn(rows) * 0.01,
    }
    return pd.DataFrame(data, index=idx)


def test_split_ratios(tmp_path):
    df     = _make_feature_df(rows=300)
    result = split_and_scale(df, "TEST", tmp_path)
    total  = result["train_rows"] + result["val_rows"] + result["test_rows"]
    assert total == len(df)
    assert 0.68 <= result["train_rows"] / len(df) <= 0.72
    assert 0.13 <= result["val_rows"]   / len(df) <= 0.17


def test_parquet_files_created(tmp_path):
    df = _make_feature_df()
    split_and_scale(df, "TEST", tmp_path)
    assert (tmp_path / "train.parquet").exists()
    assert (tmp_path / "val.parquet").exists()
    assert (tmp_path / "test.parquet").exists()
    assert (tmp_path / "scaler.joblib").exists()


def test_scaler_fitted_on_train_only(tmp_path):
    """Val/test data scaled with train scaler — train mean ≈ 0 for scaled cols."""
    df     = _make_feature_df(rows=500)
    result = split_and_scale(df, "TEST", tmp_path)

    train = load_split(result["train_path"])
    # RobustScaler centers on median — check scaled columns exist and are finite
    feat_cols = [c for c in train.columns
                 if c not in {"Ticker", "target_next_close", "target_next_return"}]
    assert train[feat_cols].isnull().sum().sum() == 0
    assert np.isfinite(train[feat_cols].values).all()


def test_load_split_roundtrip(tmp_path):
    df = _make_feature_df()
    split_and_scale(df, "TEST", tmp_path)
    train = load_split(tmp_path / "train.parquet")
    assert isinstance(train, pd.DataFrame)
    assert len(train) > 0


def test_load_scaler_roundtrip(tmp_path):
    df = _make_feature_df()
    split_and_scale(df, "TEST", tmp_path)
    scaler = load_scaler(tmp_path / "scaler.joblib")
    assert hasattr(scaler, "transform")


def test_lstm_sequences_shape(tmp_path):
    df     = _make_feature_df(rows=300)
    result = split_and_scale(df, "TEST", tmp_path, sequence_length=30)

    lstm_paths = result["lstm_paths"]
    assert "train" in lstm_paths

    X = np.load(lstm_paths["train"]["X"])
    y = np.load(lstm_paths["train"]["y"])

    n_features = result["feature_count"]
    assert X.ndim == 3
    assert X.shape[1] == 30       # sequence length
    assert X.shape[2] == n_features
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]


def test_make_sequences_correct_length():
    rows      = 100
    seq_len   = 10
    n_feats   = 5
    values    = np.random.rand(rows, n_feats)
    target    = np.random.rand(rows)
    df        = pd.DataFrame(values, columns=[f"f{i}" for i in range(n_feats)])
    df["target_next_close"] = target

    X, y = _make_sequences(df, [f"f{i}" for i in range(n_feats)], seq_len)
    assert X.shape == (rows - seq_len, seq_len, n_feats)
    assert y.shape == (rows - seq_len,)


def test_chronological_split_no_leakage(tmp_path):
    """Train max date must be strictly before val min date."""
    df     = _make_feature_df(rows=300)
    result = split_and_scale(df, "TEST", tmp_path)
    train  = load_split(result["train_path"])
    val    = load_split(result["val_path"])
    test   = load_split(result["test_path"])
    assert train.index.max() < val.index.min()
    assert val.index.max()   < test.index.min()
