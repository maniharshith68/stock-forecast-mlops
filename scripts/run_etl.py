#!/usr/bin/env python3
"""
CLI entrypoint to run the ETL pipeline manually.

Usage:
    python3 scripts/run_etl.py
    python3 scripts/run_etl.py --tickers AAPL MSFT
    python3 scripts/run_etl.py --tickers AAPL --seq-len 30
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.etl.etl_pipeline import run_etl_pipeline
from src.utils.logger import get_logger

logger = get_logger("scripts.run_etl")


def parse_args():
    parser = argparse.ArgumentParser(description="Run ETL and feature engineering pipeline")
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Tickers to process (default: all from config.yaml)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=None,
        help="LSTM sequence length (default: from config.yaml)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Manual ETL run started")

    summary = run_etl_pipeline(
        tickers=args.tickers,
        sequence_length=args.seq_len,
    )

    print("\n" + "=" * 60)
    print("ETL SUMMARY")
    print("=" * 60)
    print(f"Total tickers:  {summary['total']}")
    print(f"Succeeded:      {summary['success']}")
    print(f"Failed:         {summary['failed'] or 'none'}")
    print()

    for ticker, outcome in summary["outcomes"].items():
        if "error" in outcome:
            print(f"  {ticker}: FAILED — {outcome['error']}")
        else:
            print(f"  {ticker}:")
            print(f"    Rows  → train={outcome['train_rows']} | val={outcome['val_rows']} | test={outcome['test_rows']}")
            print(f"    Features: {outcome['feature_count']}")
            print(f"    Scaler: {outcome['scaler_path']}")
            if outcome.get("lstm_paths"):
                for split, paths in outcome["lstm_paths"].items():
                    print(f"    LSTM [{split}]: X={paths['X']}, y={paths['y']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
