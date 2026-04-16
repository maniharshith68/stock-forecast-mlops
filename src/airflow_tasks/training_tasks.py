"""Pure Python task functions for the training DAG."""
import json
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger("airflow.tasks.training")


def run_training(**context) -> dict:
    import sys
    sys.path.insert(0, "/opt/airflow")
    from src.training.training_pipeline import run_training_pipeline

    logger.info(f"Starting training | execution_date={context.get('ds', 'manual')}")
    summary = run_training_pipeline()
    logger.info(
        f"Training complete | success={len(summary['results'])} | "
        f"failed={summary['failed']}"
    )
    return summary


def validate_training(**context) -> None:
    from src.utils.config import load_config

    config  = load_config()
    tickers = config["data"]["tickers"]
    missing = []

    for ticker in tickers:
        registry = Path(f"models/registry/{ticker}")
        for fname in ["best_model.json", "random_forest.joblib",
                      "xgboost.json", "lstm.pt"]:
            p = registry / fname
            if not p.exists():
                missing.append(str(p))

    if missing:
        raise FileNotFoundError(
            "Training validation failed. Missing:\n" + "\n".join(missing)
        )

    for ticker in tickers:
        p = Path(f"models/registry/{ticker}/best_model.json")
        if p.exists():
            info = json.loads(p.read_text())
            logger.info(
                f"  {ticker}: best={info['best_model']} | "
                f"val_rmse={info['val_metrics']['rmse']:.4f}"
            )

    logger.info(f"Training validation passed for all {len(tickers)} tickers")
