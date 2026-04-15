from pathlib import Path
import pandas as pd
from src.utils.logger import get_logger
from src.utils.config import load_config
from src.etl.feature_engineer import compute_features
from src.etl.preprocessor import split_and_scale

logger = get_logger("etl.pipeline")

RAW_DATA_DIR       = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


def load_raw_parquets(ticker: str) -> pd.DataFrame:
    """
    Load all parquet files for a ticker from data/raw/{ticker}/.
    Merges them, deduplicates by date, sorts chronologically.
    """
    ticker_dir = RAW_DATA_DIR / ticker
    if not ticker_dir.exists():
        raise FileNotFoundError(
            f"No raw data found for {ticker} at {ticker_dir}. "
            "Run the ingestion pipeline first."
        )

    parquet_files = sorted(ticker_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {ticker_dir}")

    frames = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(frames, axis=0)

    # Deduplicate rows (same date from multiple ingestion runs)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    logger.info(
        f"Loaded raw data for {ticker}: {len(df)} rows "
        f"({df.index.min().date()} → {df.index.max().date()}) "
        f"from {len(parquet_files)} file(s)"
    )
    return df


def run_etl_pipeline(
    tickers: list[str] | None = None,
    sequence_length: int | None = None,
) -> dict:
    """
    Full ETL pipeline for all tickers:
      1. Load raw parquets from data/raw/
      2. Engineer features
      3. Split and scale
      4. Save to data/processed/

    Returns summary dict.
    """
    config       = load_config()
    tickers      = tickers or config["data"]["tickers"]
    seq_len      = sequence_length or config["models"]["lstm"]["sequence_length"]

    logger.info(
        f"Starting ETL pipeline | tickers={tickers} | sequence_length={seq_len}"
    )

    summary = {
        "total":   len(tickers),
        "success": 0,
        "failed":  [],
        "outcomes": {},
    }

    for ticker in tickers:
        try:
            logger.info(f"--- ETL: {ticker} ---")

            # 1. Load
            raw_df = load_raw_parquets(ticker)

            # 2. Feature engineering
            features_df = compute_features(raw_df, ticker)

            # Save intermediate features (pre-scaling) — useful for EDA
            feat_dir = PROCESSED_DATA_DIR / ticker
            feat_dir.mkdir(parents=True, exist_ok=True)
            features_path = feat_dir / "features.parquet"
            features_df.to_parquet(features_path)
            logger.info(f"Saved raw features: {features_path}")

            # 3. Split and scale
            result = split_and_scale(
                features_df, ticker,
                output_dir=feat_dir,
                sequence_length=seq_len,
            )
            result["features_path"] = str(features_path)
            summary["outcomes"][ticker] = result
            summary["success"] += 1

        except Exception as e:
            logger.error(f"ETL failed for {ticker}: {e}", exc_info=True)
            summary["failed"].append(ticker)
            summary["outcomes"][ticker] = {"error": str(e)}

    logger.info(
        f"ETL pipeline finished | "
        f"success={summary['success']} | failed={summary['failed']}"
    )
    return summary
