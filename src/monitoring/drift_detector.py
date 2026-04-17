"""
Drift detection using Evidently AI.
Evidently imports are inside functions — runs in Docker (Python 3.11).
Module-level imports are Python 3.14 safe.
"""
from pathlib import Path
from datetime import date
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger("monitoring.drift_detector")

REPORTS_DIR = Path("data/monitoring/reports")


@dataclass
class DriftResult:
    ticker:           str
    run_date:         str
    drift_detected:   bool
    drift_score:      float
    drifted_features: list[str]  = field(default_factory=list)
    total_features:   int        = 0
    report_path:      str | None = None
    error:            str | None = None

    def to_dict(self) -> dict:
        return {
            "ticker":           self.ticker,
            "run_date":         self.run_date,
            "drift_detected":   self.drift_detected,
            "drift_score":      round(self.drift_score, 4),
            "drifted_features": self.drifted_features,
            "total_features":   self.total_features,
            "report_path":      self.report_path,
            "error":            self.error,
        }


def detect_data_drift(
    reference_df: pd.DataFrame,
    current_df:   pd.DataFrame,
    ticker:       str,
    threshold:    float = 0.1,
    save_report:  bool  = True,
) -> DriftResult:
    run_date = date.today().isoformat()
    logger.info(
        f"Running drift detection | ticker={ticker} | "
        f"reference_rows={len(reference_df)} | current_rows={len(current_df)}"
    )

    exclude   = {"Ticker", "target_next_close", "target_next_return"}
    feat_cols = [
        c for c in reference_df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(reference_df[c])
    ]

    ref_features = reference_df[feat_cols].copy()

    # Handle empty current DataFrame — early return BEFORE Evidently import
    if current_df.empty:
        return DriftResult(
            ticker=ticker, run_date=run_date,
            drift_detected=False, drift_score=0.0,
            error="Empty current DataFrame",
        )

    cur_features = current_df[feat_cols].copy() if all(
        c in current_df.columns for c in feat_cols
    ) else pd.DataFrame()

    valid_cols = [
        c for c in feat_cols
        if not ref_features[c].isna().all()
        and len(cur_features.columns) > 0
        and c in cur_features.columns
        and not cur_features[c].isna().all()
    ]
    ref_features = ref_features[valid_cols]
    cur_features = cur_features[valid_cols] if valid_cols else pd.DataFrame()

    if ref_features.empty or cur_features.empty:
        return DriftResult(
            ticker=ticker, run_date=run_date,
            drift_detected=False, drift_score=0.0,
            error="Empty feature DataFrame after filtering",
        )

    # Evidently import inside function — Docker/Python 3.11 only
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    try:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_features, current_data=cur_features)

        report_dict    = report.as_dict()
        drift_metric   = report_dict["metrics"][0]["result"]
        n_drifted      = drift_metric.get("number_of_drifted_columns", 0)
        n_total        = drift_metric.get("number_of_columns", len(valid_cols))
        drift_score    = n_drifted / max(n_total, 1)
        drift_detected = drift_score > threshold

        drifted_features = []
        for col_name, col_info in drift_metric.get("drift_by_columns", {}).items():
            if col_info.get("drift_detected", False):
                drifted_features.append(col_name)

        report_path = None
        if save_report:
            report_dir = REPORTS_DIR / ticker
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = str(report_dir / f"{run_date}.html")
            report.save_html(report_path)
            logger.info(f"Drift report saved: {report_path}")

        result = DriftResult(
            ticker=ticker, run_date=run_date,
            drift_detected=drift_detected,
            drift_score=round(drift_score, 4),
            drifted_features=drifted_features,
            total_features=n_total,
            report_path=report_path,
        )

        if drift_detected:
            logger.warning(
                f"Drift DETECTED | ticker={ticker} | "
                f"score={drift_score:.3f} | {n_drifted}/{n_total} features"
            )
        else:
            logger.info(
                f"Drift OK | ticker={ticker} | "
                f"score={drift_score:.3f} | {n_drifted}/{n_total} features"
            )

        return result

    except Exception as e:
        logger.error(f"Drift detection failed for {ticker}: {e}", exc_info=True)
        return DriftResult(
            ticker=ticker, run_date=run_date,
            drift_detected=False, drift_score=0.0,
            error=str(e),
        )


def detect_prediction_drift(
    reference_predictions: np.ndarray,
    current_predictions:   np.ndarray,
    ticker:                str,
    threshold:             float = 0.1,
) -> DriftResult:
    """
    KS-test on prediction distributions.
    scipy is Python 3.14 compatible — no Evidently needed here.
    """
    from scipy import stats

    run_date = date.today().isoformat()
    logger.info(
        f"Prediction drift check | ticker={ticker} | "
        f"ref={len(reference_predictions)} | cur={len(current_predictions)}"
    )

    try:
        ks_stat, p_value = stats.ks_2samp(
            reference_predictions, current_predictions
        )
        drift_detected = ks_stat > threshold

        result = DriftResult(
            ticker=ticker, run_date=run_date,
            drift_detected=drift_detected,
            drift_score=round(float(ks_stat), 4),
            drifted_features=["predictions"] if drift_detected else [],
            total_features=1,
        )

        msg = (
            f"Prediction drift {'DETECTED' if drift_detected else 'OK'} | "
            f"ticker={ticker} | KS={ks_stat:.4f} | p={p_value:.4f}"
        )
        if drift_detected:
            logger.warning(msg)
        else:
            logger.info(msg)

        return result

    except Exception as e:
        logger.error(f"Prediction drift failed for {ticker}: {e}", exc_info=True)
        return DriftResult(
            ticker=ticker, run_date=run_date,
            drift_detected=False, drift_score=0.0,
            error=str(e),
        )
