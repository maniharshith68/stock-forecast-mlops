import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("etl.feature_engineer")


def compute_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Takes a clean OHLCV DataFrame and returns a DataFrame
    with all engineered features + target column.
    Input must have columns: Open, High, Low, Close, Volume
    and a DatetimeIndex sorted ascending.
    """
    logger.info(f"Engineering features for {ticker} | input rows: {len(df)}")

    df = df.copy().sort_index()
    out = pd.DataFrame(index=df.index)
    out["Ticker"] = ticker

    # ------------------------------------------------------------------ #
    # 1. Price-based features
    # ------------------------------------------------------------------ #
    out["returns_1d"]        = df["Close"].pct_change(1)
    out["returns_5d"]        = df["Close"].pct_change(5)
    out["returns_10d"]       = df["Close"].pct_change(10)
    out["high_low_range"]    = (df["High"] - df["Low"]) / df["Close"]
    out["close_open_spread"] = (df["Close"] - df["Open"]) / df["Open"]

    # ------------------------------------------------------------------ #
    # 2. Moving averages
    # ------------------------------------------------------------------ #
    out["sma_10"] = df["Close"].rolling(10).mean()
    out["sma_20"] = df["Close"].rolling(20).mean()
    out["sma_50"] = df["Close"].rolling(50).mean()
    out["ema_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    out["ema_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # Price relative to moving averages (dimensionless ratio)
    out["close_to_sma20"] = df["Close"] / out["sma_20"] - 1
    out["close_to_sma50"] = df["Close"] / out["sma_50"] - 1
    out["sma10_sma20_cross"] = out["sma_10"] / out["sma_20"] - 1

    # ------------------------------------------------------------------ #
    # 3. Momentum: RSI
    # ------------------------------------------------------------------ #
    out["rsi_14"] = _compute_rsi(df["Close"], period=14)

    # ------------------------------------------------------------------ #
    # 4. Momentum: MACD
    # ------------------------------------------------------------------ #
    out["macd"]        = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"]   = out["macd"] - out["macd_signal"]

    # ------------------------------------------------------------------ #
    # 5. Volatility: Bollinger Bands
    # ------------------------------------------------------------------ #
    bb_std          = df["Close"].rolling(20).std()
    out["bb_upper"] = out["sma_20"] + 2 * bb_std
    out["bb_lower"] = out["sma_20"] - 2 * bb_std
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["sma_20"]
    out["bb_pos"]   = (df["Close"] - out["bb_lower"]) / (
        out["bb_upper"] - out["bb_lower"] + 1e-9
    )

    # ------------------------------------------------------------------ #
    # 6. Volatility: ATR (Average True Range)
    # ------------------------------------------------------------------ #
    out["atr_14"] = _compute_atr(df, period=14)

    # ------------------------------------------------------------------ #
    # 7. Volume features
    # ------------------------------------------------------------------ #
    out["volume_sma_10"] = df["Volume"].rolling(10).mean()
    out["volume_ratio"]  = df["Volume"] / (out["volume_sma_10"] + 1e-9)
    out["log_volume"]    = np.log1p(df["Volume"])

    # ------------------------------------------------------------------ #
    # 8. Lag features (previous close prices, normalized)
    # ------------------------------------------------------------------ #
    for lag in [1, 2, 3, 5, 10]:
        out[f"close_lag_{lag}"] = df["Close"].shift(lag)
        out[f"return_lag_{lag}"] = df["Close"].pct_change(lag).shift(1)

    # ------------------------------------------------------------------ #
    # 9. Raw OHLCV passthrough (kept for LSTM context)
    # ------------------------------------------------------------------ #
    out["Open"]   = df["Open"]
    out["High"]   = df["High"]
    out["Low"]    = df["Low"]
    out["Close"]  = df["Close"]
    out["Volume"] = df["Volume"]

    # ------------------------------------------------------------------ #
    # 10. Target: next day's close price (what we predict)
    # ------------------------------------------------------------------ #
    out["target_next_close"]  = df["Close"].shift(-1)
    out["target_next_return"] = df["Close"].pct_change(1).shift(-1)

    # Drop rows where we can't compute features (warm-up period)
    # sma_50 needs 50 rows; lag_10 needs 10; combined ~60 rows dropped
    before = len(out)
    out = out.dropna(subset=["sma_50", "rsi_14", "close_lag_10", "target_next_close"])
    dropped = before - len(out)

    logger.info(
        f"Features engineered for {ticker} | "
        f"output rows: {len(out)} (dropped {dropped} warm-up rows) | "
        f"features: {len(out.columns)}"
    )
    return out


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI implementation."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs  = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high      = df["High"]
    low       = df["Low"]
    prev_close = df["Close"].shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(com=period - 1, min_periods=period).mean()
