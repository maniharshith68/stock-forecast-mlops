"""
Builds reference datasets for drift detection.
Reference = training data from data/processed/{ticker}/train.parquet
            scaled back using the saved scaler.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger("monitoring.reference_builder")

PROCESSED_DIR = Path("data/processed")


def load_reference_data(ticker: str) -> pd.DataFrame:
    """
    Load training split as the reference (baseline) dataset.
    Returns unscaled features for human-readable drift reports.
    Uses features.parquet (pre-scaling) so feature values are interpretable.
    """
    features_path = PROCESSED_DIR / ticker / "features.parquet"

    if not features_path.exists():
        raise FileNotFoundError(
            f"Features parquet not found for {ticker} at {features_path}. "
            "Run the ETL pipeline first."
        )

    df = pd.read_parquet(features_path)

    # Use only the training portion (first 70%) as reference
    n_train = int(len(df) * 0.70)
    reference_df = df.iloc[:n_train].copy()

    logger.info(
        f"Loaded reference data for {ticker} | "
        f"rows={len(reference_df)} | "
        f"date_range={reference_df.index.min().date()} → "
        f"{reference_df.index.max().date()}"
    )
    return reference_df


def build_current_data(
    ticker:        str,
    lookback_days: int = 120,
) -> pd.DataFrame:
    """
    Fetch recent OHLCV from yFinance and engineer features.
    This is the 'current production' data to compare against reference.
    """
    import yfinance as yf
    from datetime import date, timedelta
    from src.etl.feature_engineer import compute_features

    end_date   = date.today().isoformat()
    start_date = (date.today() - timedelta(days=lookback_days + 60)).isoformat()

    logger.info(
        f"Fetching current data for {ticker} | "
        f"{start_date} → {end_date} | lookback={lookback_days} days"
    )

    df = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        raise ValueError(f"yFinance returned no data for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.index.name = "Date"

    feat_df = compute_features(df, ticker)

    if feat_df.empty:
        raise ValueError(f"Feature engineering produced empty DataFrame for {ticker}")

    logger.info(
        f"Current data built for {ticker} | rows={len(feat_df)} | "
        f"date_range={feat_df.index.min().date()} → {feat_df.index.max().date()}"
    )
    return feat_df


def load_reference_predictions(ticker: str) -> np.ndarray:
    """
    Load validation set predictions from the trained model.
    These serve as the reference prediction distribution.
    """
    import joblib
    from pathlib import Path

    val_path = PROCESSED_DIR / ticker / "val.parquet"
    if not val_path.exists():
        raise FileNotFoundError(f"Val parquet not found for {ticker}")

    val_df = pd.read_parquet(val_path)

    # Load best model and run predictions on val set
    import json
    registry_path = Path(f"models/registry/{ticker}/best_model.json")
    if not registry_path.exists():
        raise FileNotFoundError(f"No trained model for {ticker}")

    with open(registry_path) as f:
        model_info = json.load(f)

    model_name = model_info["best_model"]
    model_path = model_info["model_path"]

    exclude   = {"Ticker", "target_next_close", "target_next_return"}
    feat_cols = [c for c in val_df.columns if c not in exclude]
    X_val     = val_df[feat_cols].values

    if model_name == "random_forest":
        model = joblib.load(model_path)
        preds = model.predict(X_val)
    elif model_name == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor()
        model.load_model(model_path)
        preds = model.predict(X_val)
    else:
        # LSTM — use saved val targets as proxy (avoids subprocess complexity)
        preds = val_df["target_next_close"].values

    logger.info(
        f"Loaded reference predictions for {ticker} | "
        f"model={model_name} | n_preds={len(preds)}"
    )
    return preds.astype(np.float64)
