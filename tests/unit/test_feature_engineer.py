"""Unit tests for feature engineer — pure logic, no I/O."""
import pandas as pd
import numpy as np
import pytest
from src.etl.feature_engineer import compute_features, _compute_rsi, _compute_atr


def _make_ohlcv(rows: int = 120) -> pd.DataFrame:
    """Synthetic OHLCV with realistic structure."""
    np.random.seed(42)
    idx    = pd.date_range("2020-01-01", periods=rows, freq="B")
    close  = 100 + np.cumsum(np.random.randn(rows) * 0.5)
    open_  = close * (1 + np.random.randn(rows) * 0.002)
    high   = np.maximum(close, open_) * (1 + np.abs(np.random.randn(rows) * 0.003))
    low    = np.minimum(close, open_) * (1 - np.abs(np.random.randn(rows) * 0.003))
    volume = np.random.randint(500_000, 2_000_000, rows).astype(float)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low,
         "Close": close, "Volume": volume},
        index=idx,
    )


def test_compute_features_returns_dataframe():
    df     = _make_ohlcv()
    result = compute_features(df, "TEST")
    assert isinstance(result, pd.DataFrame)


def test_compute_features_has_expected_columns():
    df     = _make_ohlcv()
    result = compute_features(df, "TEST")
    expected = [
        "returns_1d", "sma_10", "sma_20", "sma_50",
        "ema_12", "ema_26", "rsi_14", "macd", "macd_signal",
        "bb_width", "atr_14", "volume_ratio",
        "close_lag_1", "target_next_close", "target_next_return",
    ]
    for col in expected:
        assert col in result.columns, f"Missing column: {col}"


def test_compute_features_no_nulls_in_output():
    df     = _make_ohlcv(rows=150)
    result = compute_features(df, "TEST")
    null_counts = result.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    assert len(cols_with_nulls) == 0, f"Nulls found: {cols_with_nulls.to_dict()}"


def test_compute_features_drops_warmup_rows():
    df     = _make_ohlcv(rows=120)
    result = compute_features(df, "TEST")
    # sma_50 needs 50 rows + lag_10 needs 10 + 1 target shift = ~61 dropped
    assert len(result) < len(df)
    assert len(result) > 0


def test_compute_features_ticker_column():
    df     = _make_ohlcv()
    result = compute_features(df, "AAPL")
    assert (result["Ticker"] == "AAPL").all()


def test_target_is_next_close():
    df     = _make_ohlcv(rows=120)
    result = compute_features(df, "TEST")
    # target_next_close at row i should equal Close at row i+1
    # since we dropped warmup rows, verify alignment on a subset
    close_series  = df["Close"]
    target_series = result["target_next_close"]

    for ts in target_series.index[:10]:
        loc = close_series.index.get_loc(ts)
        if loc + 1 < len(close_series):
            expected = close_series.iloc[loc + 1]
            assert abs(target_series[ts] - expected) < 1e-6, (
                f"Target mismatch at {ts}: got {target_series[ts]}, expected {expected}"
            )


def test_rsi_bounds():
    series = _make_ohlcv()["Close"]
    rsi    = _compute_rsi(series)
    valid  = rsi.dropna()
    assert (valid >= 0).all() and (valid <= 100).all(), "RSI out of [0, 100]"


def test_atr_non_negative():
    df  = _make_ohlcv()
    atr = _compute_atr(df)
    assert (atr.dropna() >= 0).all(), "ATR should be non-negative"


def test_bb_width_non_negative():
    df     = _make_ohlcv(rows=150)
    result = compute_features(df, "TEST")
    assert (result["bb_width"] >= 0).all(), "BB width should be non-negative"


def test_volume_ratio_positive():
    df     = _make_ohlcv(rows=150)
    result = compute_features(df, "TEST")
    assert (result["volume_ratio"] > 0).all()


def test_too_few_rows_returns_empty_or_raises():
    """With fewer rows than warm-up period, output should be empty."""
    df     = _make_ohlcv(rows=30)
    result = compute_features(df, "TEST")
    assert len(result) == 0
