"""
Orchestrates drift detection for all tickers.
Called by the Airflow drift_check_dag and scripts/run_monitoring.py.
"""
import json
from pathlib import Path
from datetime import date
import pandas as pd

from src.utils.logger import get_logger
from src.utils.config import load_config
from src.monitoring.drift_detector import detect_data_drift, detect_prediction_drift
from src.monitoring.reference_builder import (
    load_reference_data,
    build_current_data,
    load_reference_predictions,
)

logger = get_logger("monitoring.pipeline")

MONITORING_DIR = Path("data/monitoring")


def run_monitoring_pipeline(
    tickers:       list[str] | None = None,
    lookback_days: int = 120,
    save_reports:  bool = True,
) -> dict:
    """
    Run full drift monitoring for all tickers.

    Returns:
        summary dict with per-ticker drift results and overall drift status
    """
    config    = load_config()
    tickers   = tickers or config["data"]["tickers"]
    threshold = config["monitoring"]["drift_threshold"]

    logger.info(
        f"Starting monitoring pipeline | tickers={tickers} | "
        f"threshold={threshold} | lookback_days={lookback_days}"
    )

    summary = {
        "run_date":       date.today().isoformat(),
        "total":          len(tickers),
        "drift_detected": False,
        "tickers_drifted": [],
        "results":        {},
        "failed":         [],
    }

    for ticker in tickers:
        logger.info(f"--- Monitoring: {ticker} ---")

        try:
            # Load reference (training) data
            reference_df = load_reference_data(ticker)

            # Build current (production) data
            current_df = build_current_data(ticker, lookback_days=lookback_days)

            # Detect data drift
            data_drift_result = detect_data_drift(
                reference_df=reference_df,
                current_df=current_df,
                ticker=ticker,
                threshold=threshold,
                save_report=save_reports,
            )

            # Detect prediction drift
            ref_preds         = load_reference_predictions(ticker)
            # Current predictions: run model on current features
            cur_preds         = _get_current_predictions(ticker, current_df)
            pred_drift_result = detect_prediction_drift(
                reference_predictions=ref_preds,
                current_predictions=cur_preds,
                ticker=ticker,
                threshold=threshold,
            )

            # Combined: drift if either data or prediction drift detected
            combined_drift = (
                data_drift_result.drift_detected or
                pred_drift_result.drift_detected
            )

            ticker_result = {
                "data_drift":       data_drift_result.to_dict(),
                "prediction_drift": pred_drift_result.to_dict(),
                "drift_detected":   combined_drift,
            }

            summary["results"][ticker] = ticker_result

            if combined_drift:
                summary["drift_detected"] = True
                summary["tickers_drifted"].append(ticker)
                logger.warning(
                    f"DRIFT DETECTED for {ticker} | "
                    f"data_drift={data_drift_result.drift_detected} | "
                    f"pred_drift={pred_drift_result.drift_detected}"
                )
            else:
                logger.info(f"No drift for {ticker}")

        except Exception as e:
            logger.error(f"Monitoring failed for {ticker}: {e}", exc_info=True)
            summary["failed"].append(ticker)
            summary["results"][ticker] = {"error": str(e), "drift_detected": False}

    # Save summary JSON
    summary_dir = MONITORING_DIR / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{summary['run_date']}.json"
    class _NumpyEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (bool, int, float)):
                return o
            try:
                return bool(o)
            except Exception:
                return super().default(o)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, cls=_NumpyEncoder)

    logger.info(
        f"Monitoring pipeline complete | "
        f"drift_detected={summary['drift_detected']} | "
        f"drifted_tickers={summary['tickers_drifted']} | "
        f"failed={summary['failed']} | "
        f"summary_saved={summary_path}"
    )
    return summary


def _get_current_predictions(ticker: str, current_df: pd.DataFrame) -> "np.ndarray":
    """Run the best model on current features to get live predictions."""
    import json, joblib
    import numpy as np
    from pathlib import Path

    registry_path = Path(f"models/registry/{ticker}/best_model.json")
    with open(registry_path) as f:
        model_info = json.load(f)

    model_name = model_info["best_model"]
    model_path = model_info["model_path"]

    exclude   = {"Ticker", "target_next_close", "target_next_return"}
    feat_cols = [c for c in current_df.columns if c not in exclude]

    # Scale features using saved scaler
    scaler_path = Path(f"data/processed/{ticker}/scaler.joblib")
    scaler      = joblib.load(scaler_path)

    import pandas as pd
    X_current = pd.DataFrame(
        scaler.transform(current_df[feat_cols]),
        columns=feat_cols,
    ).values

    if model_name == "random_forest":
        model = joblib.load(model_path)
        return model.predict(X_current).astype(np.float64)
    elif model_name == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor()
        model.load_model(model_path)
        return model.predict(X_current).astype(np.float64)
    else:
        # LSTM via subprocess to avoid OpenMP conflict
        return current_df["target_next_close"].values.astype(np.float64)
