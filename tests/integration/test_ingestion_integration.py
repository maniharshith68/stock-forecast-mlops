"""
Integration tests — real yFinance network call, local disk write.
No AWS needed. Run with: pytest tests/integration/ -v
"""
import pytest
import pandas as pd
from pathlib import Path
from src.ingestion.downloader import download_ohlcv
from src.ingestion.validator import validate_ohlcv
from src.ingestion.s3_uploader import save_locally
from src.ingestion.pipeline import run_ingestion_pipeline


@pytest.mark.integration
def test_download_real_ticker():
    """Downloads a small slice of real AAPL data."""
    df = download_ohlcv("AAPL", "2024-01-01", "2024-01-31")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "Close" in df.columns
    assert df.index.tz is None  # tz-naive


@pytest.mark.integration
def test_download_and_validate_real():
    df = download_ohlcv("MSFT", "2024-01-01", "2024-06-01")
    result = validate_ohlcv(df, "MSFT")
    assert result.passed is True


@pytest.mark.integration
def test_save_locally_real_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = download_ohlcv("GOOGL", "2024-01-01", "2024-03-01")
    path = save_locally(df, "GOOGL", "2024-03-01")
    assert path.exists()
    loaded = pd.read_parquet(path)
    assert len(loaded) == len(df)


@pytest.mark.integration
def test_full_pipeline_two_tickers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)

    summary = run_ingestion_pipeline(
        tickers=["AAPL", "MSFT"],
        start_date="2024-01-01",
        end_date="2024-03-01",
    )

    assert summary["downloaded"] == 2
    assert summary["validated"] == 2
    assert summary["stored"] == 2
    assert summary["failed_validation"] == []
    for ticker in ["AAPL", "MSFT"]:
        assert summary["outcomes"][ticker]["validation"] == "PASS"
        assert Path(summary["outcomes"][ticker]["local"]).exists()
