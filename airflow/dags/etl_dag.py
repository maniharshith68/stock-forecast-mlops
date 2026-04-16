from __future__ import annotations
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.airflow_tasks.etl_tasks import run_etl, validate_etl

DEFAULT_ARGS = {
    "owner": "mlops", "depends_on_past": False,
    "retries": 2, "retry_delay": timedelta(minutes=5),
    "email_on_failure": False, "email_on_retry": False,
}

with DAG(
    dag_id="stock_etl",
    description="Daily ETL — feature engineering and preprocessing",
    schedule="30 6 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["etl", "features", "mlops"],
    max_active_runs=1,
) as dag:
    PythonOperator(task_id="run_etl",     python_callable=run_etl,     provide_context=True) >> \
    PythonOperator(task_id="validate_etl", python_callable=validate_etl, provide_context=True)
