#!/usr/bin/env python3
"""
Run the model training pipeline.

Usage:
    python3 scripts/run_training.py
    python3 scripts/run_training.py --tickers AAPL MSFT
    python3 scripts/run_training.py --models random_forest xgboost
    python3 scripts/run_training.py --models lstm --tickers AAPL
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.training.training_pipeline import run_training_pipeline
from src.utils.config import get
from src.utils.logger import get_logger

logger = get_logger("scripts.run_training")


def parse_args():
    parser = argparse.ArgumentParser(description="Train stock forecasting models")
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Ticker symbols to train (default: all from config.yaml)",
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["random_forest", "xgboost", "lstm"],
        choices=["random_forest", "xgboost", "lstm"],
        help="Models to train (default: all three)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    uri  = get("mlflow.tracking_uri", "sqlite:///mlflow.db")

    logger.info(
        f"Training started | tickers={args.tickers} | "
        f"models={args.models} | mlflow={uri}"
    )

    summary = run_training_pipeline(
        tickers=args.tickers,
        models=args.models,
        mlflow_tracking_uri=uri,
    )

    print("\n" + "=" * 65)
    print("TRAINING SUMMARY")
    print("=" * 65)
    print(f"Tickers trained : {list(summary['results'].keys())}")
    print(f"Models trained  : {summary['models_trained']}")
    if summary["failed"]:
        print(f"Failed          : {summary['failed']}")
    print()
    for ticker, info in summary["results"].items():
        print(f"  {ticker}:")
        print(f"    Best model : {info['best_model']}")
        print(f"    Val  RMSE  : {info['val_metrics']['rmse']:.4f}")
        print(f"    Val  R²    : {info['val_metrics']['r2']:.4f}")
        print(f"    Test RMSE  : {info['test_metrics']['rmse']:.4f}")
        print(f"    Test R²    : {info['test_metrics']['r2']:.4f}")
    print("=" * 65)
    print("MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("=" * 65)


if __name__ == "__main__":
    main()
