import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from src.utils.logger import get_logger

logger = get_logger("etl.preprocessor")

# Columns never fed into the scaler
NON_FEATURE_COLS = {"Ticker", "target_next_close", "target_next_return"}

# Train / val / test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test = remaining 0.15


def split_and_scale(
    df: pd.DataFrame,
    ticker: str,
    output_dir: Path,
    sequence_length: int = 30,
) -> dict:
    """
    1. Chronological train/val/test split (no shuffle — time series)
    2. Fit RobustScaler on train only, apply to val and test
    3. Save splits as parquet + scaler as joblib
    4. Also produce LSTM sequences (numpy arrays) and save as .npy

    Returns dict with split sizes and file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.sort_index()
    n  = len(df)

    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    logger.info(
        f"{ticker} split: train={len(train_df)} | val={len(val_df)} | test={len(test_df)}"
    )

    # Feature columns (everything except meta and targets)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    # Fit scaler on train only
    scaler = RobustScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols]   = scaler.transform(val_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

    # Save tabular splits
    train_path  = output_dir / "train.parquet"
    val_path    = output_dir / "val.parquet"
    test_path   = output_dir / "test.parquet"
    scaler_path = output_dir / "scaler.joblib"

    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    test_df.to_parquet(test_path)
    joblib.dump(scaler, scaler_path)

    logger.info(f"Saved tabular splits and scaler to {output_dir}")

    # LSTM sequences
    lstm_paths = _save_lstm_sequences(
        train_df, val_df, test_df,
        feature_cols, sequence_length, output_dir, ticker
    )

    return {
        "ticker":         ticker,
        "train_rows":     len(train_df),
        "val_rows":       len(val_df),
        "test_rows":      len(test_df),
        "feature_count":  len(feature_cols),
        "train_path":     str(train_path),
        "val_path":       str(val_path),
        "test_path":      str(test_path),
        "scaler_path":    str(scaler_path),
        "lstm_paths":     lstm_paths,
    }


def _save_lstm_sequences(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    sequence_length: int,
    output_dir: Path,
    ticker: str,
) -> dict:
    """
    Build sliding-window sequences for LSTM:
      X shape: (samples, sequence_length, n_features)
      y shape: (samples,)
    """
    lstm_dir = output_dir / "lstm"
    lstm_dir.mkdir(exist_ok=True)
    paths = {}

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        X, y = _make_sequences(split_df, feature_cols, sequence_length)
        if X is None:
            logger.warning(
                f"{ticker} {split_name}: not enough rows for sequences "
                f"(need >{sequence_length}, have {len(split_df)})"
            )
            continue

        x_path = lstm_dir / f"{split_name}_X.npy"
        y_path = lstm_dir / f"{split_name}_y.npy"
        np.save(x_path, X)
        np.save(y_path, y)
        paths[split_name] = {"X": str(x_path), "y": str(y_path)}
        logger.info(
            f"LSTM sequences [{ticker}] {split_name}: X={X.shape}, y={y.shape}"
        )

    return paths


def _make_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Sliding window: each sample is seq_len consecutive rows."""
    values = df[feature_cols].values
    target = df["target_next_close"].values

    if len(values) <= seq_len:
        return None, None

    X, y = [], []
    for i in range(seq_len, len(values)):
        X.append(values[i - seq_len : i])
        y.append(target[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def load_split(path: str | Path) -> pd.DataFrame:
    """Convenience loader for any split parquet."""
    return pd.read_parquet(path)


def load_scaler(path: str | Path) -> RobustScaler:
    """Load a saved scaler."""
    return joblib.load(path)
