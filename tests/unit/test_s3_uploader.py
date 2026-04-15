"""Unit tests for S3 uploader — uses moto to mock AWS, zero real AWS calls."""
import os
import pytest
import pandas as pd
import boto3
from moto import mock_aws
from unittest.mock import patch
from src.ingestion.s3_uploader import save_locally, upload_to_s3, store_ohlcv


BUCKET = "test-bucket"
PREFIX = "raw/ohlcv"


def _make_df(rows: int = 10) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=rows, freq="B")
    return pd.DataFrame(
        {"Open": [100.0]*rows, "High": [105.0]*rows,
         "Low": [98.0]*rows, "Close": [102.0]*rows,
         "Volume": [500_000]*rows, "Ticker": ["TEST"]*rows},
        index=idx,
    )


def test_save_locally(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = _make_df()
    path = save_locally(df, "TEST", "2024-01-01")
    assert path.exists()
    loaded = pd.read_parquet(path)
    assert len(loaded) == 10


@mock_aws
def test_upload_to_s3_success():
    # Create mock bucket
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)

    df = _make_df()
    result = upload_to_s3(df, "TEST", "2024-01-01", BUCKET, PREFIX)

    assert result is True

    # Verify object actually exists
    objects = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX)
    assert objects["KeyCount"] == 1
    assert "TEST/2024-01-01.parquet" in objects["Contents"][0]["Key"]


@mock_aws
def test_upload_to_s3_wrong_bucket_fails():
    # Don't create bucket — should return False gracefully
    df = _make_df()
    result = upload_to_s3(df, "TEST", "2024-01-01", "nonexistent-bucket", PREFIX)
    assert result is False


def test_upload_to_s3_no_credentials_returns_false():
    """Without moto, real boto3 with bad creds returns False gracefully."""
    df = _make_df()
    with patch.dict(os.environ, {
        "AWS_ACCESS_KEY_ID": "",
        "AWS_SECRET_ACCESS_KEY": "",
        "AWS_DEFAULT_REGION": "us-east-1",
    }):
        result = upload_to_s3(df, "TEST", "2024-01-01", BUCKET, PREFIX)
    assert result is False


def test_store_ohlcv_local_only(tmp_path, monkeypatch):
    """store_ohlcv saves locally and skips S3 when no AWS key set."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)

    df = _make_df()
    outcome = store_ohlcv(df, "TEST", "2024-01-01")

    assert outcome["local"] is not None
    assert outcome["s3"] is False
