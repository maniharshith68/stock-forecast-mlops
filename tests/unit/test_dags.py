"""
Unit tests for Airflow task callables.
Imports directly from src.airflow_tasks — zero Airflow dependency.
All tests run on Python 3.14 locally.
Airflow itself runs in Docker (Python 3.11).
"""
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, ".")


# ── Ingestion tasks ───────────────────────────────────────────────────────────

def test_run_ingestion_success():
    mock_summary = {
        "total": 5, "downloaded": 5, "validated": 5,
        "stored": 5, "failed_validation": [],
    }
    with patch("src.ingestion.pipeline.run_ingestion_pipeline",
               return_value=mock_summary):
        from src.airflow_tasks.ingestion_tasks import run_ingestion
        result = run_ingestion(ds="2026-04-16", ti=MagicMock())
    assert result["stored"] == 5
    assert result["failed_validation"] == []


def test_validate_ingestion_raises_on_zero_stored():
    mock_ti = MagicMock()
    mock_ti.xcom_pull.return_value = {
        "total": 5, "downloaded": 0, "validated": 0,
        "stored": 0, "failed_validation": ["AAPL"],
    }
    from src.airflow_tasks.ingestion_tasks import validate_ingestion
    with pytest.raises(ValueError, match="stored 0 tickers"):
        validate_ingestion(ds="2026-04-16", ti=mock_ti)


def test_validate_ingestion_passes_on_full_store():
    mock_ti = MagicMock()
    mock_ti.xcom_pull.return_value = {
        "total": 5, "downloaded": 5, "validated": 5,
        "stored": 5, "failed_validation": [],
    }
    from src.airflow_tasks.ingestion_tasks import validate_ingestion
    validate_ingestion(ds="2026-04-16", ti=mock_ti)  # should not raise


# ── ETL tasks ─────────────────────────────────────────────────────────────────

def test_run_etl_success():
    mock_summary = {"success": 5, "failed": [], "outcomes": {}}
    with patch("src.etl.etl_pipeline.run_etl_pipeline",
               return_value=mock_summary):
        from src.airflow_tasks.etl_tasks import run_etl
        result = run_etl(ds="2026-04-16", ti=MagicMock())
    assert result["success"] == 5
    assert result["failed"]  == []


def test_validate_etl_passes_when_files_exist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tickers = ["AAPL", "MSFT"]

    # Create required files
    for ticker in tickers:
        base = tmp_path / "data" / "processed" / ticker
        (base / "lstm").mkdir(parents=True)
        for fname in ["train.parquet", "val.parquet", "test.parquet", "scaler.joblib"]:
            (base / fname).write_bytes(b"fake")
        (base / "lstm" / "train_X.npy").write_bytes(b"fake")

    with patch("src.utils.config.load_config",
               return_value={"data": {"tickers": tickers}}):
        from src.airflow_tasks.etl_tasks import validate_etl
        validate_etl(ds="2026-04-16", ti=MagicMock())


# ── Training tasks ────────────────────────────────────────────────────────────

def test_run_training_success():
    mock_summary = {
        "total_tickers": 5,
        "models_trained": ["random_forest", "xgboost", "lstm"],
        "results": {
            t: {"best_model": "random_forest",
                "val_metrics": {"rmse": 20.0, "r2": 0.5},
                "test_metrics": {"rmse": 25.0, "r2": 0.4}}
            for t in ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        },
        "failed": [],
    }
    with patch("src.training.training_pipeline.run_training_pipeline",
               return_value=mock_summary):
        from src.airflow_tasks.training_tasks import run_training
        result = run_training(ds="2026-04-16", ti=MagicMock())
    assert len(result["results"]) == 5
    assert result["failed"] == []


# ── Drift tasks ───────────────────────────────────────────────────────────────

def test_check_drift_no_drift_by_default():
    from src.airflow_tasks.drift_tasks import check_drift
    result = check_drift(ds="2026-04-16", ti=MagicMock())
    assert result["drift_detected"] is False
    assert len(result["results"]) == 5
    for ticker_result in result["results"].values():
        assert ticker_result["drift_detected"] is False
        assert ticker_result["drift_score"]    == 0.0


def test_trigger_retraining_no_action_when_no_drift():
    mock_ti = MagicMock()
    mock_ti.xcom_pull.return_value = {"drift_detected": False, "results": {}}
    from src.airflow_tasks.drift_tasks import trigger_retraining_if_needed
    trigger_retraining_if_needed(ds="2026-04-16", ti=mock_ti)  # should not raise


# ── Structural tests ──────────────────────────────────────────────────────────

def test_all_dag_files_exist():
    dag_files = [
        "airflow/dags/ingestion_dag.py",
        "airflow/dags/etl_dag.py",
        "airflow/dags/training_dag.py",
        "airflow/dags/drift_check_dag.py",
    ]
    for f in dag_files:
        assert Path(f).exists(), f"Missing DAG file: {f}"


def test_all_task_modules_exist():
    task_files = [
        "src/airflow_tasks/__init__.py",
        "src/airflow_tasks/ingestion_tasks.py",
        "src/airflow_tasks/etl_tasks.py",
        "src/airflow_tasks/training_tasks.py",
        "src/airflow_tasks/drift_tasks.py",
    ]
    for f in task_files:
        assert Path(f).exists(), f"Missing task module: {f}"


def test_dag_files_have_correct_schedules():
    schedules = {
        "airflow/dags/ingestion_dag.py":   "0 6 * * *",
        "airflow/dags/etl_dag.py":         "30 6 * * *",
        "airflow/dags/training_dag.py":    "0 2 * * 0",
        "airflow/dags/drift_check_dag.py": "0 7 * * *",
    }
    for filepath, expected in schedules.items():
        content = Path(filepath).read_text()
        assert expected in content, f"{filepath} missing schedule '{expected}'"


def test_dag_files_import_from_airflow_tasks():
    """Verify DAGs import from src.airflow_tasks, not inline."""
    for dag_file in ["ingestion_dag", "etl_dag", "training_dag", "drift_check_dag"]:
        content = Path(f"airflow/dags/{dag_file}.py").read_text()
        assert "src.airflow_tasks" in content, (
            f"{dag_file}.py should import from src.airflow_tasks"
        )
