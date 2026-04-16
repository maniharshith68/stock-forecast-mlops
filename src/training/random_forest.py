import time
import joblib
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from src.utils.logger import get_logger
from src.utils.config import load_config
from src.training.evaluator import compute_metrics

logger = get_logger("training.random_forest")

PROCESSED_DIR = Path("data/processed")
REGISTRY_DIR  = Path("models/registry")


def load_tabular_data(
    ticker: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = PROCESSED_DIR / ticker
    return (
        pd.read_parquet(base / "train.parquet"),
        pd.read_parquet(base / "val.parquet"),
        pd.read_parquet(base / "test.parquet"),
    )


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"Ticker", "target_next_close", "target_next_return"}
    return [c for c in df.columns if c not in exclude]


def train_random_forest(ticker: str) -> dict:
    config  = load_config()
    rf_cfg  = config["models"]["random_forest"]
    tgt_col = config["training"]["target_col"]

    logger.info(f"Training Random Forest | ticker={ticker}")

    train_df, val_df, test_df = load_tabular_data(ticker)
    feat_cols = get_feature_cols(train_df)

    X_train, y_train = train_df[feat_cols].values, train_df[tgt_col].values
    X_val,   y_val   = val_df[feat_cols].values,   val_df[tgt_col].values
    X_test,  y_test  = test_df[feat_cols].values,  test_df[tgt_col].values

    with mlflow.start_run(run_name=f"rf_{ticker}") as run:
        mlflow.log_params({
            "model":        "RandomForest",
            "ticker":       ticker,
            "n_estimators": rf_cfg["n_estimators"],
            "max_depth":    rf_cfg["max_depth"],
            "n_features":   len(feat_cols),
            "train_rows":   len(X_train),
        })

        t0    = time.time()
        model = RandomForestRegressor(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            n_jobs=rf_cfg.get("n_jobs", -1),
            random_state=42,
        )
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        train_m = compute_metrics(y_train, model.predict(X_train), f"RF/{ticker}/train")
        val_m   = compute_metrics(y_val,   model.predict(X_val),   f"RF/{ticker}/val")
        test_m  = compute_metrics(y_test,  model.predict(X_test),  f"RF/{ticker}/test")

        mlflow.log_metrics({
            "train_mae":  train_m.mae,  "train_rmse": train_m.rmse,  "train_r2": train_m.r2,
            "val_mae":    val_m.mae,    "val_rmse":   val_m.rmse,    "val_r2":   val_m.r2,
            "test_mae":   test_m.mae,   "test_rmse":  test_m.rmse,   "test_r2":  test_m.r2,
            "train_time_s": elapsed,
        })

        out_dir    = REGISTRY_DIR / ticker
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "random_forest.joblib"
        joblib.dump(model, model_path)
        mlflow.log_param("model_path", str(model_path))

        run_id = run.info.run_id
        logger.info(
            f"RF/{ticker} done | val_rmse={val_m.rmse:.4f} | "
            f"val_r2={val_m.r2:.4f} | time={elapsed:.1f}s"
        )

    return {
        "model":        "random_forest",
        "ticker":       ticker,
        "run_id":       run_id,
        "model_path":   str(model_path),
        "val_metrics":  val_m.to_dict(),
        "test_metrics": test_m.to_dict(),
    }
