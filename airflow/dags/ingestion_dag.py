from __future__ import annotations
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.airflow_tasks.ingestion_tasks import run_ingestion, validate_ingestion

DEFAULT_ARGS = {
    "owner": "mlops", "depends_on_past": False,
    "retries": 2, "retry_delay": timedelta(minutes=5),
    "email_on_failure": False, "email_on_retry": False,
}

with DAG(
    dag_id="stock_ingestion",
    description="Daily OHLCV data ingestion from yFinance",
    schedule="0 6 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["ingestion", "data", "mlops"],
    max_active_runs=1,
) as dag:
    PythonOperator(task_id="run_ingestion",     python_callable=run_ingestion,     provide_context=True) >> \
    PythonOperator(task_id="validate_ingestion", python_callable=validate_ingestion, provide_context=True)
