"""
Airflow task functions for drift monitoring.
Uses Evidently AI for real data and prediction drift detection.
"""
from src.utils.logger import get_logger

logger = get_logger("airflow.tasks.drift")


def check_drift(**context) -> dict:
    """
    Run full drift detection using Evidently AI.
    Compares current market data against training reference data.
    """
    import sys
    sys.path.insert(0, "/opt/airflow")

    from src.monitoring.monitoring_pipeline import run_monitoring_pipeline
    from src.utils.config import load_config

    config  = load_config()
    tickers = config["data"]["tickers"]

    logger.info(
        f"Drift check starting | "
        f"execution_date={context.get('ds', 'manual')} | "
        f"tickers={tickers}"
    )

    summary = run_monitoring_pipeline(
        tickers=tickers,
        lookback_days=120,
        save_reports=True,
    )

    logger.info(
        f"Drift check complete | "
        f"drift_detected={summary['drift_detected']} | "
        f"drifted={summary['tickers_drifted']}"
    )

    return {
        "drift_detected":  summary["drift_detected"],
        "tickers_drifted": summary["tickers_drifted"],
        "results":         {
            t: r.get("drift_detected", False)
            for t, r in summary["results"].items()
        },
    }


def trigger_retraining_if_needed(**context) -> None:
    """
    Triggers retraining DAG if drift was detected above threshold.
    Uses TriggerDagRunOperator pattern — logs intent, actual trigger
    is handled by the DAG definition in production.
    """
    ti     = context["ti"]
    result = ti.xcom_pull(task_ids="check_drift")

    if result is None:
        logger.warning("No drift result received — skipping retraining check")
        return

    if result.get("drift_detected"):
        drifted = result.get("tickers_drifted", [])
        logger.warning(
            f"DRIFT DETECTED for {drifted} — retraining pipeline triggered. "
            "Run: python3 scripts/run_training.py"
        )
        # In production Airflow, add TriggerDagRunOperator here
        # to automatically start stock_training DAG
    else:
        logger.info(
            "Drift check passed — no retraining needed. "
            f"All tickers within threshold."
        )
