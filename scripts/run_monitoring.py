#!/usr/bin/env python3
"""
CLI entrypoint to run the drift monitoring pipeline manually.

Usage:
    python3 scripts/run_monitoring.py
    python3 scripts/run_monitoring.py --tickers AAPL MSFT
    python3 scripts/run_monitoring.py --lookback 90 --no-reports
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.monitoring.monitoring_pipeline import run_monitoring_pipeline
from src.utils.logger import get_logger

logger = get_logger("scripts.run_monitoring")


def parse_args():
    parser = argparse.ArgumentParser(description="Run drift monitoring pipeline")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument(
        "--lookback", type=int, default=120,
        help="Days of recent data to use as current dataset (default: 120)"
    )
    parser.add_argument(
        "--no-reports", action="store_true",
        help="Skip saving HTML drift reports"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Monitoring pipeline started")

    summary = run_monitoring_pipeline(
        tickers=args.tickers,
        lookback_days=args.lookback,
        save_reports=not args.no_reports,
    )

    print("\n" + "=" * 65)
    print("MONITORING SUMMARY")
    print("=" * 65)
    print(f"Run date:         {summary['run_date']}")
    print(f"Drift detected:   {summary['drift_detected']}")
    print(f"Tickers drifted:  {summary['tickers_drifted'] or 'none'}")
    print(f"Failed:           {summary['failed'] or 'none'}")
    print()

    for ticker, result in summary["results"].items():
        if "error" in result:
            print(f"  {ticker}: ERROR — {result['error']}")
            continue

        dd = result["data_drift"]
        pd_ = result["prediction_drift"]
        print(f"  {ticker}:")
        print(f"    Data drift:       {dd['drift_detected']} "
              f"(score={dd['drift_score']:.3f}, "
              f"features={len(dd['drifted_features'])}/{dd['total_features']})")
        print(f"    Prediction drift: {pd_['drift_detected']} "
              f"(KS={pd_['drift_score']:.3f})")
        if dd.get("report_path"):
            print(f"    Report:           {dd['report_path']}")

    print("=" * 65)
    if summary["drift_detected"]:
        print("⚠ DRIFT DETECTED — consider retraining:")
        print("  python3 scripts/run_training.py")
    else:
        print("✓ No significant drift detected")
    print("=" * 65)


if __name__ == "__main__":
    main()
