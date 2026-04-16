"""Pure Python task functions for the ETL DAG."""
from src.utils.logger import get_logger

logger = get_logger("airflow.tasks.etl")


def run_etl(**context) -> dict:
    import sys
    sys.path.insert(0, "/opt/airflow")
    from src.etl.etl_pipeline import run_etl_pipeline

    logger.info(f"Starting ETL | execution_date={context.get('ds', 'manual')}")
    summary = run_etl_pipeline()
    logger.info(f"ETL complete | success={summary['success']} | failed={summary['failed']}")
    return summary


def validate_etl(**context) -> None:
    from pathlib import Path
    from src.utils.config import load_config

    config  = load_config()
    tickers = config["data"]["tickers"]
    missing = []

    for ticker in tickers:
        base = Path(f"data/processed/{ticker}")
        for fname in ["train.parquet", "val.parquet", "test.parquet", "scaler.joblib"]:
            p = base / fname
            if not p.exists():
                missing.append(str(p))
        lstm_X = base / "lstm" / "train_X.npy"
        if not lstm_X.exists():
            missing.append(str(lstm_X))

    if missing:
        raise FileNotFoundError(
            "ETL validation failed. Missing:\n" + "\n".join(missing)
        )

    logger.info(f"ETL validation passed for all {len(tickers)} tickers")
