"""Unit tests for downloader — zero network calls."""
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from src.ingestion.downloader import download_ohlcv, _clean_dataframe, download_multiple_tickers


def _make_fake_df(rows: int = 30, ticker: str = "AAPL") -> pd.DataFrame:
    """Helper: build a minimal valid OHLCV DataFrame."""
    idx = pd.date_range("2023-01-01", periods=rows, freq="B")
    return pd.DataFrame(
        {
            "Open":   [150.0 + i for i in range(rows)],
            "High":   [155.0 + i for i in range(rows)],
            "Low":    [148.0 + i for i in range(rows)],
            "Close":  [152.0 + i for i in range(rows)],
            "Volume": [1_000_000 + i * 1000 for i in range(rows)],
        },
        index=idx,
    )


@patch("src.ingestion.downloader.yf.download")
def test_download_ohlcv_success(mock_download):
    fake_df = _make_fake_df()
    mock_download.return_value = fake_df

    result = download_ohlcv("AAPL", "2023-01-01")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 30
    mock_download.assert_called_once()



@patch("src.ingestion.downloader.yf.Ticker")
def test_download_ohlcv_empty_raises(mock_ticker_cls):
    mock_ticker_cls.return_value.history.return_value = pd.DataFrame()

    with pytest.raises(Exception):
        download_ohlcv("INVALID", "2023-01-01")


@patch("src.ingestion.downloader.yf.download")
def test_download_ohlcv_retries_then_raises(mock_download):
    mock_download.side_effect = RuntimeError("network error")

    with patch("src.ingestion.downloader.time.sleep"):
        with pytest.raises(RuntimeError):
            download_ohlcv("AAPL", "2023-01-01")


def test_clean_dataframe_removes_timezone():
    import pytz
    idx = pd.date_range("2023-01-01", periods=5, freq="B", tz="America/New_York")
    df = pd.DataFrame(
        {"Open": [1.0]*5, "High": [2.0]*5, "Low": [0.5]*5,
         "Close": [1.5]*5, "Volume": [1000]*5},
        index=idx,
    )
    result = _clean_dataframe(df, "TEST")
    assert result.index.tz is None


def test_clean_dataframe_drops_duplicates():
    idx = pd.DatetimeIndex(["2023-01-02", "2023-01-02", "2023-01-03"])
    df = pd.DataFrame(
        {"Open": [1.0]*3, "High": [2.0]*3, "Low": [0.5]*3,
         "Close": [1.5]*3, "Volume": [1000]*3},
        index=idx,
    )
    result = _clean_dataframe(df, "TEST")
    assert len(result) == 2


@patch("src.ingestion.downloader.yf.Ticker")
def test_download_multiple_tickers_skips_failures(mock_ticker_cls):
    good_df = _make_fake_df()

    def side_effect(ticker):
        mock = MagicMock()
        if ticker == "AAPL":
            mock.history.return_value = good_df
        else:
            mock.history.return_value = pd.DataFrame()  # empty → will raise
        return mock

    mock_ticker_cls.side_effect = side_effect

    with patch("src.ingestion.downloader.time.sleep"):
        results = download_multiple_tickers(["AAPL", "BADTICKER"], "2023-01-01")

    assert "AAPL" in results
    assert "BADTICKER" not in results
