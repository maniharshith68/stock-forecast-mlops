"""Unit tests for validator — pure logic, no I/O."""
import pandas as pd
import pytest
from src.ingestion.validator import validate_ohlcv


def _make_valid_df(rows: int = 30) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=rows, freq="B")
    return pd.DataFrame(
        {
            "Open":   [150.0] * rows,
            "High":   [155.0] * rows,
            "Low":    [148.0] * rows,
            "Close":  [152.0] * rows,
            "Volume": [1_000_000] * rows,
            "Ticker": ["AAPL"] * rows,
        },
        index=idx,
    )


def test_valid_dataframe_passes():
    df = _make_valid_df()
    result = validate_ohlcv(df, "AAPL")
    assert result.passed is True
    assert result.errors == []


def test_empty_dataframe_fails():
    result = validate_ohlcv(pd.DataFrame(), "AAPL")
    assert result.passed is False
    assert any("empty" in e.lower() for e in result.errors)


def test_too_few_rows_fails():
    df = _make_valid_df(rows=5)
    result = validate_ohlcv(df, "AAPL")
    assert result.passed is False
    assert any("few rows" in e for e in result.errors)


def test_missing_column_fails():
    df = _make_valid_df().drop(columns=["Close"])
    result = validate_ohlcv(df, "AAPL")
    assert result.passed is False
    assert any("Close" in e for e in result.errors)


def test_negative_price_fails():
    df = _make_valid_df()
    df.loc[df.index[0], "Close"] = -1.0
    result = validate_ohlcv(df, "AAPL")
    assert result.passed is False
    assert any("Close" in e for e in result.errors)


def test_high_lower_than_low_fails():
    df = _make_valid_df()
    df.loc[df.index[0], "High"] = 100.0   # lower than Low=148
    df.loc[df.index[0], "Low"] = 200.0    # higher than High
    result = validate_ohlcv(df, "AAPL")
    assert result.passed is False


def test_excessive_nulls_fails():
    df = _make_valid_df()
    # Set 50% of Close to NaN → exceeds 5% threshold
    null_idx = df.index[: len(df) // 2]
    df.loc[null_idx, "Close"] = None
    result = validate_ohlcv(df, "AAPL")
    assert result.passed is False


def test_non_datetime_index_fails():
    df = _make_valid_df()
    df.index = list(range(len(df)))
    result = validate_ohlcv(df, "AAPL")
    assert result.passed is False


def test_negative_volume_fails():
    df = _make_valid_df()
    df.loc[df.index[0], "Volume"] = -100
    result = validate_ohlcv(df, "AAPL")
    assert result.passed is False
