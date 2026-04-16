"""Pure Python task functions for the drift check DAG."""
from src.utils.logger import get_logger

logger = get_logger("airflow.tasks.drift")


def check_drift(**context) -> dict:
    from src.utils.config import load_config

    config    = load_config()
    tickers   = config["data"]["tickers"]
    threshold = config["monitoring"]["drift_threshold"]

    logger.info(
        f"Drift check | execution_date={context.get('ds', 'manual')} | "
        f"tickers={tickers} | threshold={threshold}"
    )

    # Phase 7 replaces this stub with real Evidently AI drift scores
    drift_results = {
        ticker: {"drift_detected": False, "drift_score": 0.0}
        for ticker in tickers
    }

    any_drift = any(v["drift_detected"] for v in drift_results.values())
    logger.info(f"Drift check complete | drift_detected={any_drift}")

    return {"drift_detected": any_drift, "results": drift_results}


def trigger_retraining_if_needed(**context) -> None:
    ti     = context["ti"]
    result = ti.xcom_pull(task_ids="check_drift")

    if result is None:
        logger.warning("No drift result — skipping retraining check")
        return

    if result.get("drift_detected"):
        logger.warning(
            "Drift detected — retraining triggered. "
            "Phase 7 will add TriggerDagRunOperator."
        )
    else:
        logger.info("No drift detected — no retraining needed")
