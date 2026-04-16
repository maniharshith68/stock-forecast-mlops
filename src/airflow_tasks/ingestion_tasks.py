"""
Pure Python task functions for the ingestion DAG.
Zero Airflow imports — fully testable on any Python version.
"""
from src.utils.logger import get_logger

logger = get_logger("airflow.tasks.ingestion")


def run_ingestion(**context) -> dict:
    import sys
    sys.path.insert(0, "/opt/airflow")
    from src.ingestion.pipeline import run_ingestion_pipeline

    logger.info(f"Starting ingestion | execution_date={context.get('ds', 'manual')}")
    summary = run_ingestion_pipeline()
    logger.info(
        f"Ingestion complete | downloaded={summary['downloaded']} | "
        f"stored={summary['stored']}"
    )
    return summary


def validate_ingestion(**context) -> None:
    ti      = context["ti"]
    summary = ti.xcom_pull(task_ids="run_ingestion")

    if summary is None:
        raise ValueError("No summary returned from ingestion task")

    if summary["stored"] == 0:
        raise ValueError(
            f"Ingestion stored 0 tickers. "
            f"Failed: {summary['failed_validation']}"
        )

    logger.info(
        f"Ingestion validation passed: "
        f"{summary['stored']}/{summary['total']} tickers stored"
    )
