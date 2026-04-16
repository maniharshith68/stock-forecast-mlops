import time
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from typing import Any
import torch

import yfinance as yf
from src.utils.logger import get_logger
from src.utils.config import load_config
from src.etl.feature_engineer import compute_features

logger = get_logger("serving.predictor")

REGISTRY_DIR  = Path("models/registry")
PROCESSED_DIR = Path("data/processed")


# ── Registry helpers ──────────────────────────────────────────────────────────

def list_available_tickers() -> list[str]:
    """Return all tickers that have a complete model registry entry."""
    if not REGISTRY_DIR.exists():
        return []
    return sorted(
        t.name for t in REGISTRY_DIR.iterdir()
        if t.is_dir() and (t / "best_model.json").exists()
    )


def load_best_model_info(ticker: str) -> dict:
    """Load best_model.json for a ticker. Raises FileNotFoundError if missing."""
    path = REGISTRY_DIR / ticker / "best_model.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found for ticker '{ticker}'. "
            f"Run the training pipeline first."
        )
    with open(path) as f:
        return json.load(f)


# ── Model loaders ─────────────────────────────────────────────────────────────

def _load_random_forest(model_path: str) -> Any:
    return joblib.load(model_path)


def _load_xgboost(model_path: str) -> Any:
    from xgboost import XGBRegressor
    model = XGBRegressor()
    model.load_model(model_path)
    return model


def _load_lstm(model_path: str) -> Any:
    from src.training.lstm_model import StockLSTM, load_lstm_model
    model, device = load_lstm_model(model_path)
    return (model, device)


MODEL_LOADERS = {
    "random_forest": _load_random_forest,
    "xgboost":       _load_xgboost,
    "lstm":          _load_lstm,
}


# ── Feature pipeline ──────────────────────────────────────────────────────────

def fetch_and_engineer_features(
    ticker:       str,
    lookback_days: int = 90,
) -> tuple[np.ndarray, int]:
    """
    Fetches recent OHLCV from yFinance, runs feature engineering,
    loads the saved scaler, and returns the latest feature row
    ready for model inference.

    Returns:
        (features_2d, n_features) where features_2d shape is (1, n_features)
    """

    end_date   = date.today().isoformat()
    start_date = (date.today() - timedelta(days=lookback_days + 60)).isoformat()

    logger.info(f"Fetching {ticker} OHLCV | {start_date} → {end_date}")

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

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normalize index timezone
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.index.name = "Date"

    # Engineer features
    feat_df = compute_features(df, ticker)
    if feat_df.empty:
        raise ValueError(
            f"Feature engineering produced empty DataFrame for {ticker}. "
            "Not enough historical data."
        )

    # Load scaler fitted during training
    scaler_path = PROCESSED_DIR / ticker / "scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found for {ticker} at {scaler_path}. "
            "Run the ETL pipeline first."
        )
    scaler = joblib.load(scaler_path)

    # Select and scale feature columns (same exclusion logic as training)
    exclude    = {"Ticker", "target_next_close", "target_next_return"}
    feat_cols  = [c for c in feat_df.columns if c not in exclude]
    latest_row = feat_df[feat_cols].iloc[[-1]].values   # shape (1, n_features)

    # Convert to DataFrame with feature names to suppress sklearn warning
    latest_df = pd.DataFrame(latest_row, columns=feat_cols)
    scaled = scaler.transform(latest_df)
    return scaled, len(feat_cols)


# ── Inference ─────────────────────────────────────────────────────────────────

def _predict_tabular(model: Any, features: np.ndarray) -> float:
    """RF and XGBoost both use sklearn-compatible predict."""
    return float(model.predict(features)[0])


def _predict_lstm(model_tuple: tuple, features: np.ndarray, seq_len: int = 30) -> float:
    """
    LSTM expects (1, seq_len, n_features).
    We only have one row at inference time, so we tile it across seq_len.
    In production, pass the full sequence for best results.
    """
    model, device = model_tuple
    n_features    = features.shape[1]
    # Tile single feature row across sequence length
    seq = np.tile(features, (seq_len, 1))           # (seq_len, n_features)
    X   = torch.from_numpy(seq).float().unsqueeze(0)  # (1, seq_len, n_features)
    with torch.no_grad():
        pred = model(X.to(device))
    return float(pred.cpu().numpy()[0])


def run_inference(
    ticker:        str,
    lookback_days: int = 90,
) -> dict:
    """
    Full inference pipeline for one ticker:
      1. Load best model info
      2. Fetch and engineer features
      3. Load model
      4. Predict
      5. Return structured result

    Returns dict matching PredictionResponse schema.
    """
    t0 = time.time()

    # Load registry info
    info       = load_best_model_info(ticker)
    model_name = info["best_model"]
    model_path = info["model_path"]

    logger.info(f"Running inference | ticker={ticker} | model={model_name}")

    # Load model
    if model_name not in MODEL_LOADERS:
        raise ValueError(f"Unknown model type '{model_name}' for {ticker}")
    model = MODEL_LOADERS[model_name](model_path)

    # Feature engineering
    features, n_features = fetch_and_engineer_features(ticker, lookback_days)

    # Predict
    if model_name == "lstm":
        config   = load_config()
        seq_len  = config["models"]["lstm"]["sequence_length"]
        pred_val = _predict_lstm(model, features, seq_len)
    else:
        pred_val = _predict_tabular(model, features)

    elapsed_ms = (time.time() - t0) * 1000

    # Prediction date is next trading day
    prediction_date = (date.today() + timedelta(days=1)).isoformat()

    result = {
        "ticker":            ticker,
        "predicted_close":   round(pred_val, 4),
        "model_used":        model_name,
        "prediction_date":   prediction_date,
        "val_rmse":          info["val_metrics"]["rmse"],
        "val_r2":            info["val_metrics"]["r2"],
        "features_used":     n_features,
        "inference_time_ms": round(elapsed_ms, 2),
    }

    logger.info(
        f"Prediction | {ticker} | {model_name} | "
        f"predicted_close={pred_val:.4f} | {elapsed_ms:.0f}ms"
    )
    return result
