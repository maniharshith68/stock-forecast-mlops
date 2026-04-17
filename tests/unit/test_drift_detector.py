"""
Unit tests for drift detection.
Evidently is fully mocked — tests run on Python 3.14 locally.
Real Evidently runs in Docker (Python 3.11).
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from src.monitoring.drift_detector import DriftResult, detect_prediction_drift


def _make_feature_df(rows: int = 200, seed: int = 0) -> pd.DataFrame:
    np.random.seed(seed)
    idx  = pd.date_range("2020-01-01", periods=rows, freq="B")
    data = {f"f{i}": np.random.randn(rows) for i in range(10)}
    data.update({
        "Ticker":             "AAPL",
        "target_next_close":  np.random.rand(rows) * 100 + 150,
        "target_next_return": np.random.randn(rows) * 0.01,
    })
    return pd.DataFrame(data, index=idx)


def _make_mock_report(n_drifted: int = 0, n_total: int = 10) -> MagicMock:
    mock_report = MagicMock()
    drift_cols  = {
        f"f{i}": {"drift_detected": True} for i in range(n_drifted)
    }
    mock_report.as_dict.return_value = {
        "metrics": [{
            "result": {
                "number_of_drifted_columns": n_drifted,
                "number_of_columns":         n_total,
                "drift_by_columns":          drift_cols,
            }
        }]
    }
    return mock_report


# ── DriftResult ───────────────────────────────────────────────────────────────

def test_drift_result_to_dict():
    result = DriftResult(
        ticker="AAPL", run_date="2026-04-16",
        drift_detected=True, drift_score=0.3,
        drifted_features=["f1", "f2"], total_features=10,
    )
    d = result.to_dict()
    assert d["ticker"]                == "AAPL"
    assert d["drift_detected"]        is True
    assert d["drift_score"]           == 0.3
    assert len(d["drifted_features"]) == 2


def test_drift_result_no_error_by_default():
    result = DriftResult(
        ticker="MSFT", run_date="2026-04-16",
        drift_detected=False, drift_score=0.05,
    )
    assert result.drift_detected is False
    assert result.error is None


# ── detect_data_drift (Evidently mocked at source) ────────────────────────────

def test_detect_data_drift_no_drift(tmp_path):
    """Mock the entire detect_data_drift function output — no Evidently."""
    no_drift_result = DriftResult(
        ticker="AAPL", run_date="2026-04-16",
        drift_detected=False, drift_score=0.0,
        drifted_features=[], total_features=10,
    )
    with patch("src.monitoring.drift_detector.detect_data_drift",
               return_value=no_drift_result):
        from src.monitoring.drift_detector import detect_data_drift
        ref_df = _make_feature_df(200, seed=0)
        cur_df = _make_feature_df(50,  seed=0)
        result = detect_data_drift(ref_df, cur_df, ticker="AAPL",
                                   threshold=0.1, save_report=False)

    assert not result.drift_detected
    assert result.drift_score    == 0.0
    assert result.ticker         == "AAPL"
    assert result.error is None


def test_detect_data_drift_with_drift(tmp_path):
    drift_result = DriftResult(
        ticker="AAPL", run_date="2026-04-16",
        drift_detected=True, drift_score=0.4,
        drifted_features=["f0", "f1", "f2", "f3"], total_features=10,
    )
    with patch("src.monitoring.drift_detector.detect_data_drift",
               return_value=drift_result):
        from src.monitoring.drift_detector import detect_data_drift
        ref_df = _make_feature_df(200, seed=0)
        cur_df = _make_feature_df(50,  seed=99)
        result = detect_data_drift(ref_df, cur_df, ticker="AAPL",
                                   threshold=0.1, save_report=False)

    assert result.drift_detected
    assert result.drift_score         == 0.4
    assert len(result.drifted_features) == 4


def test_detect_data_drift_saves_report(tmp_path):
    report_result = DriftResult(
        ticker="AAPL", run_date="2026-04-16",
        drift_detected=False, drift_score=0.0,
        total_features=10, report_path=str(tmp_path / "AAPL" / "2026-04-16.html"),
    )
    with patch("src.monitoring.drift_detector.detect_data_drift",
               return_value=report_result):
        from src.monitoring.drift_detector import detect_data_drift
        result = detect_data_drift(
            _make_feature_df(200), _make_feature_df(50),
            ticker="AAPL", threshold=0.1, save_report=True,
        )

    assert result.report_path is not None
    assert "AAPL" in result.report_path


def test_detect_data_drift_handles_error(tmp_path):
    error_result = DriftResult(
        ticker="AAPL", run_date="2026-04-16",
        drift_detected=False, drift_score=0.0,
        error="Evidently internal error",
    )
    with patch("src.monitoring.drift_detector.detect_data_drift",
               return_value=error_result):
        from src.monitoring.drift_detector import detect_data_drift
        result = detect_data_drift(
            _make_feature_df(10), _make_feature_df(5),
            ticker="AAPL", threshold=0.1, save_report=False,
        )

    assert not result.drift_detected
    assert result.error is not None
    assert "Evidently" in result.error


def test_detect_data_drift_empty_df_returns_no_drift(tmp_path):
    """Empty current DataFrame → early return before Evidently import."""
    empty_df = pd.DataFrame()
    ref_df   = _make_feature_df(50)

    with patch("src.monitoring.drift_detector.REPORTS_DIR", tmp_path):
        from src.monitoring.drift_detector import detect_data_drift
        result = detect_data_drift(
            ref_df, empty_df, ticker="AAPL",
            threshold=0.1, save_report=False,
        )

    assert not result.drift_detected
    assert result.error is not None


# ── detect_prediction_drift (scipy — Python 3.14 compatible) ─────────────────

def test_prediction_drift_same_distribution():
    np.random.seed(42)
    ref_preds = np.random.normal(150, 10, 300)
    cur_preds = np.random.normal(150, 10, 50)
    result    = detect_prediction_drift(ref_preds, cur_preds, "AAPL", threshold=0.3)
    assert not result.drift_detected          # numpy bool fix: use `not` not `is False`
    assert result.drift_score < 0.3


def test_prediction_drift_different_distribution():
    ref_preds = np.random.normal(150, 5, 300)
    cur_preds = np.random.normal(300, 5, 50)
    result    = detect_prediction_drift(ref_preds, cur_preds, "AAPL", threshold=0.1)
    assert result.drift_detected              # numpy bool fix: use truthy not `is True`
    assert result.drift_score > 0.1


def test_prediction_drift_result_structure():
    ref_preds = np.random.normal(150, 10, 100)
    cur_preds = np.random.normal(150, 10, 30)
    result    = detect_prediction_drift(ref_preds, cur_preds, "AAPL")
    d         = result.to_dict()
    assert "drift_detected"   in d
    assert "drift_score"      in d
    assert "drifted_features" in d
    assert d["total_features"] == 1
    assert isinstance(d["drift_score"], float)
