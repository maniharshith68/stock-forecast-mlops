from __future__ import annotations
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.airflow_tasks.drift_tasks import check_drift, trigger_retraining_if_needed

DEFAULT_ARGS = {
    "owner": "mlops", "depends_on_past": False,
    "retries": 1, "retry_delay": timedelta(minutes=5),
    "email_on_failure": False, "email_on_retry": False,
}

with DAG(
    dag_id="stock_drift_check",
    description="Daily drift monitoring — triggers retraining if needed",
    schedule="0 7 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["monitoring", "drift", "mlops"],
    max_active_runs=1,
) as dag:
    PythonOperator(task_id="check_drift",                  python_callable=check_drift,                  provide_context=True) >> \
    PythonOperator(task_id="trigger_retraining_if_needed", python_callable=trigger_retraining_if_needed, provide_context=True)
