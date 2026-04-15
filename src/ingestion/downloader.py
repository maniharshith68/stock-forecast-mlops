import time
from datetime import date
from typing import Optional
import pandas as pd
import yfinance as yf
from src.utils.logger import get_logger

logger = get_logger("ingestion.downloader")

REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 15   # longer delay — Yahoo rate limits need breathing room
INTER_TICKER_DELAY  = 8    # wait between tickers to avoid triggering rate limit


def download_ohlcv(
    ticker: str,
    start_date: str,
    end_date: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker using yf.download().
    Uses period="max" for long date ranges (more reliable than start/end).
    Retries up to MAX_RETRIES times on failure.
    """
    if end_date is None:
        end_date = date.today().isoformat()

    logger.info(
        f"Downloading {ticker} | {start_date} → {end_date} | interval={interval}"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = yf.download(
                tickers=ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                actions=False,
                progress=False,
                threads=False,   # single-threaded — avoids triggering rate limits
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
                wait = RETRY_DELAY_SECONDS * attempt   # back-off: 15s, 30s
                logger.info(f"Waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                logger.error(f"All {MAX_RETRIES} attempts failed for {ticker}")
                raise

    raise RuntimeError(f"Failed to download {ticker}")


def _clean_dataframe(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Normalize columns, ensure DatetimeIndex is tz-naive,
    add ticker column, drop duplicates.
    yf.download() returns MultiIndex columns when given a single ticker —
    this handles both cases.
    """
    # yf.download returns MultiIndex columns like ("Close", "AAPL")
    # Flatten to single level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only OHLCV columns
    cols_to_keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols_to_keep].copy()

    # Normalize index to tz-naive
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    df.index.name = "Date"
    df["Ticker"] = ticker

    # Drop duplicate dates, sort
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    # Drop rows where all price columns are NaN
    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    df = df.dropna(subset=price_cols, how="all")

    return df


def download_multiple_tickers(
    tickers: list[str],
    start_date: str,
    end_date: Optional[str] = None,
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV for multiple tickers one at a time with delays.
    Failed tickers are logged and skipped.
    """
    results = {}
    failed  = []

    for i, ticker in enumerate(tickers):
        # Polite delay between tickers (skip before first)
        if i > 0:
            logger.info(f"Waiting {INTER_TICKER_DELAY}s before next ticker...")
            time.sleep(INTER_TICKER_DELAY)

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
