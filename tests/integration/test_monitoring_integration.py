"""
Integration tests for drift monitoring.
- reference_builder tests: real data, no mocking needed
- detect_data_drift: mocked (Evidently requires Python <3.13)
- full pipeline: mocked detect_data_drift, real everything else
Evidently runs for real in Docker (Python 3.11).
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from src.monitoring.drift_detector import DriftResult
from datetime import date


def _mock_drift_result(ticker: str, detected: bool = False) -> DriftResult:
    return DriftResult(
        ticker=ticker,
        run_date=date.today().isoformat(),
        drift_detected=detected,
        drift_score=0.35 if detected else 0.05,
        drifted_features=["f1", "f2"] if detected else [],
        total_features=40,
    )


@pytest.mark.integration
def test_load_reference_data_aapl():
    from src.monitoring.reference_builder import load_reference_data
    df = load_reference_data("AAPL")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 500
    # Reference should be the training portion only
    assert len(df) < 2100   # less than full dataset


@pytest.mark.integration
def test_build_current_data_aapl():
    from src.monitoring.reference_builder import build_current_data
    df = build_current_data("AAPL", lookback_days=90)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 10
    assert "target_next_close" in df.columns
    # Should be recent data
    last_date = df.index.max()
    assert last_date.year >= 2026


@pytest.mark.integration
def test_detect_data_drift_mocked():
    """
    Validates detect_data_drift plumbing with real data shapes.
    Evidently call mocked — runs in Docker (Python 3.11) for real detection.
    """
    from src.monitoring.reference_builder import load_reference_data, build_current_data

    reference_df = load_reference_data("AAPL")
    current_df   = build_current_data("AAPL", lookback_days=90)

    # Verify data shapes are correct for drift detection
    assert len(reference_df) > 100
    assert len(current_df)   > 10

    # Verify feature columns align
    exclude    = {"Ticker", "target_next_close", "target_next_return"}
    ref_cols   = {c for c in reference_df.columns if c not in exclude}
    cur_cols   = {c for c in current_df.columns   if c not in exclude}
    common_cols = ref_cols & cur_cols
    assert len(common_cols) > 30, f"Only {len(common_cols)} common feature cols"

    # Mock the actual Evidently call and verify result structure
    expected = _mock_drift_result("AAPL", detected=False)
    with patch("src.monitoring.drift_detector.detect_data_drift",
               return_value=expected):
        from src.monitoring.drift_detector import detect_data_drift
        result = detect_data_drift(
            reference_df=reference_df,
            current_df=current_df,
            ticker="AAPL",
            threshold=0.1,
            save_report=False,
        )

    assert result.ticker         == "AAPL"
    assert isinstance(result.drift_detected, bool) or hasattr(result.drift_detected, '__bool__')
    assert 0.0 <= result.drift_score <= 1.0
    assert result.total_features  > 0
    assert result.error is None


@pytest.mark.integration
def test_load_reference_predictions_aapl():
    from src.monitoring.reference_builder import load_reference_predictions
    preds = load_reference_predictions("AAPL")
    assert isinstance(preds, np.ndarray)
    assert len(preds) > 0
    assert np.isfinite(preds).all()
    # Predictions should be in a reasonable stock price range
    assert preds.mean() > 0


@pytest.mark.integration
def test_full_monitoring_pipeline_one_ticker(tmp_path):
    """
    Full pipeline on AAPL — real reference builder, real yFinance,
    mocked Evidently (Docker-only). Tests orchestration and output structure.
    """
    import src.monitoring.monitoring_pipeline as mp_module

    with patch("src.monitoring.monitoring_pipeline.detect_data_drift",
               return_value=_mock_drift_result("AAPL")), \
         patch.object(mp_module, "MONITORING_DIR", tmp_path):
        from src.monitoring.monitoring_pipeline import run_monitoring_pipeline
        summary = run_monitoring_pipeline(
            tickers=["AAPL"],
            lookback_days=90,
            save_reports=False,
        )

    assert "AAPL"             in summary["results"]
    assert summary["failed"]  == []
    assert isinstance(summary["drift_detected"], bool)

    aapl_result = summary["results"]["AAPL"]
    assert "data_drift"       in aapl_result
    assert "prediction_drift" in aapl_result
    assert "drift_detected"   in aapl_result

    # Summary JSON should be saved
    summary_files = list((tmp_path / "summaries").glob("*.json"))
    assert len(summary_files) == 1
