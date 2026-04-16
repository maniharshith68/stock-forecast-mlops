import time
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
from xgboost import XGBRegressor
from src.utils.logger import get_logger
from src.utils.config import load_config
from src.training.evaluator import compute_metrics
from src.training.random_forest import load_tabular_data, get_feature_cols

logger = get_logger("training.xgboost_model")

REGISTRY_DIR = Path("models/registry")


def train_xgboost(ticker: str) -> dict:
    config  = load_config()
    xgb_cfg = config["models"]["xgboost"]
    tgt_col = config["training"]["target_col"]

    logger.info(f"Training XGBoost | ticker={ticker}")

    train_df, val_df, test_df = load_tabular_data(ticker)
    feat_cols = get_feature_cols(train_df)

    X_train, y_train = train_df[feat_cols].values, train_df[tgt_col].values
    X_val,   y_val   = val_df[feat_cols].values,   val_df[tgt_col].values
    X_test,  y_test  = test_df[feat_cols].values,  test_df[tgt_col].values

    with mlflow.start_run(run_name=f"xgb_{ticker}") as run:
        mlflow.log_params({
            "model":            "XGBoost",
            "ticker":           ticker,
            "n_estimators":     xgb_cfg["n_estimators"],
            "learning_rate":    xgb_cfg["learning_rate"],
            "max_depth":        xgb_cfg["max_depth"],
            "subsample":        xgb_cfg.get("subsample", 0.8),
            "colsample_bytree": xgb_cfg.get("colsample_bytree", 0.8),
            "n_features":       len(feat_cols),
            "train_rows":       len(X_train),
        })

        t0    = time.time()
        model = XGBRegressor(
            n_estimators=xgb_cfg["n_estimators"],
            learning_rate=xgb_cfg["learning_rate"],
            max_depth=xgb_cfg["max_depth"],
            subsample=xgb_cfg.get("subsample", 0.8),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
            random_state=42,
            verbosity=0,
            eval_metric="rmse",
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        elapsed = time.time() - t0

        train_m = compute_metrics(y_train, model.predict(X_train), f"XGB/{ticker}/train")
        val_m   = compute_metrics(y_val,   model.predict(X_val),   f"XGB/{ticker}/val")
        test_m  = compute_metrics(y_test,  model.predict(X_test),  f"XGB/{ticker}/test")

        mlflow.log_metrics({
            "train_mae":  train_m.mae,  "train_rmse": train_m.rmse,  "train_r2": train_m.r2,
            "val_mae":    val_m.mae,    "val_rmse":   val_m.rmse,    "val_r2":   val_m.r2,
            "test_mae":   test_m.mae,   "test_rmse":  test_m.rmse,   "test_r2":  test_m.r2,
            "train_time_s": elapsed,
        })

        out_dir    = REGISTRY_DIR / ticker
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "xgboost.json"
        model.save_model(str(model_path))
        mlflow.log_param("model_path", str(model_path))

        run_id = run.info.run_id
        logger.info(
            f"XGB/{ticker} done | val_rmse={val_m.rmse:.4f} | "
            f"val_r2={val_m.r2:.4f} | time={elapsed:.1f}s"
        )

    return {
        "model":        "xgboost",
        "ticker":       ticker,
        "run_id":       run_id,
        "model_path":   str(model_path),
        "val_metrics":  val_m.to_dict(),
        "test_metrics": test_m.to_dict(),
    }
