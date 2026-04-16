from __future__ import annotations
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.airflow_tasks.training_tasks import run_training, validate_training

DEFAULT_ARGS = {
    "owner": "mlops", "depends_on_past": False,
    "retries": 1, "retry_delay": timedelta(minutes=10),
    "email_on_failure": False, "email_on_retry": False,
}

with DAG(
    dag_id="stock_training",
    description="Weekly model retraining — RF, XGBoost, LSTM",
    schedule="0 2 * * 0",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["training", "models", "mlops"],
    max_active_runs=1,
) as dag:
    PythonOperator(task_id="run_training",      python_callable=run_training,      provide_context=True) >> \
    PythonOperator(task_id="validate_training", python_callable=validate_training, provide_context=True)
