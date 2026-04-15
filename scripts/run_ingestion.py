#!/usr/bin/env python3
"""
CLI entrypoint to run the ingestion pipeline manually.

Usage:
    python3 scripts/run_ingestion.py
    python3 scripts/run_ingestion.py --tickers AAPL MSFT --start 2023-01-01
"""
import sys
import argparse
from pathlib import Path

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.ingestion.pipeline import run_ingestion_pipeline
from src.utils.logger import get_logger

logger = get_logger("scripts.run_ingestion")


def parse_args():
    parser = argparse.ArgumentParser(description="Run stock data ingestion pipeline")
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Space-separated ticker symbols (default: from config.yaml)"
    )
    parser.add_argument(
        "--start", default=None,
        help="Start date YYYY-MM-DD (default: from config.yaml)"
    )
    parser.add_argument(
        "--end", default=None,
        help="End date YYYY-MM-DD (default: today)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Manual ingestion run started")

    summary = run_ingestion_pipeline(
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
    )

    print("\n" + "=" * 50)
    print("INGESTION SUMMARY")
    print("=" * 50)
    print(f"Total tickers:       {summary['total']}")
    print(f"Downloaded:          {summary['downloaded']}")
    print(f"Validated:           {summary['validated']}")
    print(f"Stored:              {summary['stored']}")
    if summary["failed_validation"]:
        print(f"Failed validation:   {summary['failed_validation']}")
    print("\nPer-ticker outcomes:")
    for ticker, outcome in summary["outcomes"].items():
        print(f"  {ticker}: {outcome}")
    print("=" * 50)


if __name__ == "__main__":
    main()
