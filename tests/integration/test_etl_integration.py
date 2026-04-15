"""
Integration tests for ETL — reads real parquet files from data/raw/.
Requires ingestion to have been run first.
Run with: pytest tests/integration/test_etl_integration.py -v -m integration
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.etl.etl_pipeline import run_etl_pipeline, load_raw_parquets
from src.etl.feature_engineer import compute_features
from src.etl.preprocessor import split_and_scale, load_split


@pytest.mark.integration
def test_load_raw_parquets_aapl():
    df = load_raw_parquets("AAPL")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 500          # 2018 → today is ~1800 rows
    assert "Close" in df.columns
    assert df.index.is_monotonic_increasing


@pytest.mark.integration
def test_feature_engineering_on_real_data():
    df      = load_raw_parquets("AAPL")
    feat_df = compute_features(df, "AAPL")
    assert len(feat_df) > 400
    assert feat_df.isnull().sum().sum() == 0
    assert "rsi_14" in feat_df.columns
    assert "target_next_close" in feat_df.columns


@pytest.mark.integration
def test_split_and_scale_real_data(tmp_path):
    df      = load_raw_parquets("MSFT")
    feat_df = compute_features(df, "MSFT")
    result  = split_and_scale(feat_df, "MSFT", tmp_path, sequence_length=30)

    assert result["train_rows"] > 0
    assert result["val_rows"]   > 0
    assert result["test_rows"]  > 0
    assert Path(result["scaler_path"]).exists()

    # Verify LSTM sequences were created
    assert "train" in result["lstm_paths"]
    X = np.load(result["lstm_paths"]["train"]["X"])
    assert X.ndim == 3
    assert X.shape[1] == 30


@pytest.mark.integration
def test_full_etl_pipeline_two_tickers(tmp_path, monkeypatch):
    # Redirect output to tmp_path
    import src.etl.etl_pipeline as etl_mod
    monkeypatch.setattr(etl_mod, "PROCESSED_DATA_DIR", tmp_path)

    summary = run_etl_pipeline(tickers=["AAPL", "MSFT"])

    assert summary["success"] == 2
    assert summary["failed"]  == []
    for ticker in ["AAPL", "MSFT"]:
        outcome = summary["outcomes"][ticker]
        assert outcome["train_rows"] > 0
        assert Path(outcome["scaler_path"]).exists()


@pytest.mark.integration
def test_no_data_leakage_across_splits(tmp_path):
    df      = load_raw_parquets("GOOGL")
    feat_df = compute_features(df, "GOOGL")
    result  = split_and_scale(feat_df, "GOOGL", tmp_path)

    train = load_split(result["train_path"])
    val   = load_split(result["val_path"])
    test  = load_split(result["test_path"])

    assert train.index.max() < val.index.min(),  "Train/val date overlap!"
    assert val.index.max()   < test.index.min(), "Val/test date overlap!"
