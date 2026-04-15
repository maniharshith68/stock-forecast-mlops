import time
from datetime import datetime, date
from typing import Optional
import pandas as pd
import yfinance as yf
from src.utils.logger import get_logger
from src.utils.config import get

logger = get_logger("ingestion.downloader")

REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5


def download_ohlcv(
    ticker: str,
    start_date: str,
    end_date: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker from yFinance.
    Retries up to MAX_RETRIES times on failure.
    Returns a clean DataFrame with a DatetimeIndex.
    """
    if end_date is None:
        end_date = date.today().isoformat()

    logger.info(
        f"Downloading {ticker} | {start_date} → {end_date} | interval={interval}"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                actions=False,
            )

            if df is None or df.empty:
                raise ValueError(f"yFinance returned empty data for {ticker}")

            df = _clean_dataframe(df, ticker)
            logger.info(
                f"Downloaded {len(df)} rows for {ticker} "
                f"({df.index.min().date()} to {df.index.max().date()})"
            )
            return df

        except Exception as e:
            logger.warning(
                f"Attempt {attempt}/{MAX_RETRIES} failed for {ticker}: {e}"
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logger.error(f"All {MAX_RETRIES} attempts failed for {ticker}")
                raise

    # unreachable but satisfies type checker
    raise RuntimeError(f"Failed to download {ticker}")


def _clean_dataframe(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Normalize columns, ensure DatetimeIndex is tz-naive UTC,
    add ticker column, drop duplicates.
    """
    # Flatten MultiIndex columns if present (yfinance sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only OHLCV columns that exist
    cols_to_keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols_to_keep].copy()

    # Normalize index to tz-naive
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    df.index.name = "Date"
    df["Ticker"] = ticker

    # Drop duplicate dates
    df = df[~df.index.duplicated(keep="first")]

    # Sort chronologically
    df = df.sort_index()

    return df


def download_multiple_tickers(
    tickers: list[str],
    start_date: str,
    end_date: Optional[str] = None,
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV for multiple tickers.
    Returns dict of {ticker: DataFrame}.
    Failed tickers are logged and skipped.
    """
    results = {}
    failed = []

    for ticker in tickers:
        try:
            df = download_ohlcv(ticker, start_date, end_date, interval)
            results[ticker] = df
        except Exception as e:
            logger.error(f"Skipping {ticker} after all retries: {e}")
            failed.append(ticker)

    logger.info(
        f"Download complete: {len(results)} succeeded, {len(failed)} failed"
        + (f" | Failed: {failed}" if failed else "")
    )
    return results
