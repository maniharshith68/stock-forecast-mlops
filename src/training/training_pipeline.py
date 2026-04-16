import json
from pathlib import Path
import mlflow
from src.utils.logger import get_logger
from src.utils.config import load_config
from src.training.random_forest import train_random_forest
from src.training.xgboost_model import train_xgboost
from src.training.lstm_model import train_lstm, train_lstm_subprocess

logger = get_logger("training.pipeline")

REGISTRY_DIR = Path("models/registry")


def select_best_model(results: list[dict]) -> dict:
    """Pick the model with the lowest validation RMSE."""
    return min(results, key=lambda r: r["val_metrics"]["rmse"])


def run_training_pipeline(
    tickers:             list[str] | None = None,
    models:              list[str] | None = None,
    mlflow_tracking_uri: str | None       = None,
) -> dict:
    """
    Train all models for all tickers.
    Writes best_model.json per ticker to models/registry/{ticker}/.

    Args:
        tickers:             ticker symbols (default: from config)
        models:              model names to train (default: all three)
        mlflow_tracking_uri: override tracking URI (tests pass sqlite:///:memory:)
    """
    config  = load_config()
    tickers = tickers or config["data"]["tickers"]
    models  = models  or ["random_forest", "xgboost", "lstm"]
    uri     = mlflow_tracking_uri or config["mlflow"]["tracking_uri"]

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    logger.info(
        f"Training pipeline start | tickers={tickers} | "
        f"models={models} | mlflow_uri={uri}"
    )

    summary = {
        "total_tickers":  len(tickers),
        "models_trained": models,
        "results":        {},
        "failed":         [],
    }

    trainer_map = {
        "random_forest": train_random_forest,
        "xgboost":       train_xgboost,
        "lstm":          train_lstm_subprocess,
    }

    for ticker in tickers:
        logger.info(f"=== {ticker} ===")
        ticker_results = []

        for model_name in models:
            try:
                result = trainer_map[model_name](ticker)
                ticker_results.append(result)
                logger.info(
                    f"  {model_name}/{ticker} OK | "
                    f"val_rmse={result['val_metrics']['rmse']:.4f}"
                )
            except Exception as e:
                logger.error(
                    f"  {model_name}/{ticker} FAILED: {e}", exc_info=True
                )
                summary["failed"].append(f"{model_name}/{ticker}")

        if not ticker_results:
            logger.error(f"All models failed for {ticker}")
            continue

        best = select_best_model(ticker_results)
        logger.info(
            f"Best for {ticker}: {best['model']} "
            f"(val_rmse={best['val_metrics']['rmse']:.4f})"
        )

        best_info = {
            "ticker":       ticker,
            "best_model":   best["model"],
            "model_path":   best["model_path"],
            "run_id":       best["run_id"],
            "val_metrics":  best["val_metrics"],
            "test_metrics": best["test_metrics"],
            "all_results": [
                {
                    "model":       r["model"],
                    "val_metrics": r["val_metrics"],
                    "model_path":  r["model_path"],
                }
                for r in ticker_results
            ],
        }

        out_path = REGISTRY_DIR / ticker / "best_model.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(best_info, f, indent=2)

        summary["results"][ticker] = best_info

    logger.info(
        f"Pipeline done | success={len(summary['results'])} | "
        f"failed={summary['failed']}"
    )
    return summary
