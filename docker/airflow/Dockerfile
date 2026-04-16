FROM apache/airflow:2.9.3-python3.11

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install project Python dependencies
COPY requirements.txt /tmp/project_requirements.txt
RUN pip install --no-cache-dir \
    psycopg2-binary \
    -r /tmp/project_requirements.txt

# Copy project source so DAGs can import src.*
COPY --chown=airflow:root src/    /opt/airflow/src/
COPY --chown=airflow:root config/ /opt/airflow/config/

ENV PYTHONPATH=/opt/airflow
