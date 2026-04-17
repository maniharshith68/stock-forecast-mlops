[![CI](https://github.com/maniharshith68/stock-forecast-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/maniharshith68/stock-forecast-mlops/actions/workflows/ci.yml)

# Stock Forecast MLOps Pipeline

A production-grade MLOps pipeline for next-day stock price prediction, built with industry-standard tools across the full ML lifecycle.

## Architecture

yFinance → Feature Engineering → Model Training → FastAPI → Docker → Airflow → Drift Monitoring → AWS S3

## What It Does

Fetches daily OHLCV data for AAPL, MSFT, GOOGL, AMZN, and TSLA, engineers 40 technical features, trains Random Forest, XGBoost, and LSTM models, serves predictions via a REST API, monitors for data and prediction drift, and backs up all artifacts to AWS S3.

## Tech Stack

| Layer | Technology |
|---|---|
| Data | yFinance, pandas, numpy |
| Features | 40 technical indicators (RSI, MACD, Bollinger Bands, ATR) |
| Models | Random Forest, XGBoost, LSTM (PyTorch) |
| Experiment Tracking | MLflow (SQLite backend) |
| Serving | FastAPI + uvicorn |
| Containerization | Docker + Docker Compose |
| Orchestration | Apache Airflow 2.9.3 |
| Drift Monitoring | Evidently AI + scipy KS-test |
| Storage | AWS S3 |
| Testing | pytest (91 unit tests, full integration suite) |
| Language | Python 3.11 (Docker) / 3.14 (local dev) |

## Project Structure

```

stock-forecast-mlops/
├── src/
│   ├── ingestion/        # yFinance downloader, validator, S3 uploader
│   ├── etl/              # Feature engineering, preprocessing, LSTM sequences
│   ├── training/         # RF, XGBoost, LSTM trainers, MLflow tracking
│   ├── serving/          # FastAPI app, predictor, schemas
│   ├── monitoring/       # Evidently AI drift detection, reference builder
│   └── airflow_tasks/    # Airflow task callables (Python 3.14 compatible)
├── airflow/dags/         # 4 DAGs: ingestion, ETL, training, drift check
├── docker/               # Dockerfiles for API and Airflow
├── tests/
│   ├── unit/             # 91 unit tests, fully mocked
│   └── integration/      # End-to-end tests with real data
├── scripts/              # CLI entrypoints for each pipeline stage
├── infrastructure/       # EC2 setup scripts
└── config/               # YAML configuration

```


## Quickstart

```bash
# Clone
git clone https://github.com/maniharshith68/stock-forecast-mlops.git
cd stock-forecast-mlops

# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python3 scripts/run_ingestion.py     # Fetch OHLCV data
python3 scripts/run_etl.py           # Feature engineering
python3 scripts/run_training.py      # Train all models
python3 scripts/run_api.py           # Start prediction API
```

## API Endpoints

GET  /health              Liveness check
GET  /models              List all trained models and metrics
GET  /models/{ticker}     Model details for one ticker
POST /predict/{ticker}    Predict next-day closing price


### Example Prediction

```bash
curl -X POST http://localhost:8000/predict/AAPL \
  -H "Content-Type: application/json" \
  -d '{"lookback_days": 90}'
```

```json
{
  "ticker": "AAPL",
  "predicted_close": 190.41,
  "model_used": "random_forest",
  "prediction_date": "2026-04-17",
  "val_rmse": 27.75,
  "val_r2": -0.25,
  "features_used": 40,
  "inference_time_ms": 1239.63
}
```

## Docker Deployment

```bash
docker compose up -d
# API:    http://localhost:8000
# MLflow: http://localhost:5001
# Airflow: http://localhost:8080  (admin/admin)
```

## Airflow DAGs

| DAG | Schedule | Purpose |
|---|---|---|
| stock_ingestion | Daily 06:00 UTC | Fetch new OHLCV data |
| stock_etl | Daily 06:30 UTC | Feature engineering |
| stock_training | Sunday 02:00 UTC | Retrain all models |
| stock_drift_check | Daily 07:00 UTC | Evidently AI drift detection |

## Model Performance

| Ticker | Best Model | Val RMSE | Val R² |
|---|---|---|---|
| AAPL | Random Forest | 27.75 | -0.25 |
| MSFT | Random Forest | 73.21 | -8.81 |
| GOOGL | Random Forest | 23.55 | -0.62 |
| AMZN | Random Forest | 17.58 | 0.38 |
| TSLA | Random Forest | 19.58 | 0.93 |

Note: Negative R² on validation/test sets reflects distribution shift between training (2018–2022) and evaluation (2023–2026) periods — exactly the type of drift this pipeline's monitoring system is designed to detect and flag for retraining.

## AWS S3 Artifact Storage

All trained models, scalers, and processed data are backed up to S3:

```bash
# Upload artifacts
python3 scripts/upload_to_s3.py --bucket stock-forecasting-pipeline

# View stored artifacts
aws s3 ls s3://stock-forecasting-pipeline/stock-forecast-mlops/ --recursive
```

## Testing

```bash
# Unit tests (91 tests, ~55 seconds)
PYTHONPATH=. python3 -m pytest tests/unit/ -v

# Integration tests (real data + real models)
PYTHONPATH=. python3 -m pytest tests/integration/ -v -m integration
```

## Known Constraints

- **LSTM local training:** PyTorch 2.10 + Python 3.14 + macOS ARM requires subprocess isolation and truncated BPTT due to OpenMP conflicts. LSTM trains correctly in Docker (Python 3.11).
- **Evidently AI:** Requires Python < 3.13 due to pydantic v1 compatibility. Runs correctly in Docker (Python 3.11). Unit tests use mocks; integration tests mock the Evidently call.
- **EC2 deployment:** Requires t3.small or larger (2GB RAM minimum for PyTorch + FastAPI stack).


---


## Getting Started

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Git
- 8GB RAM recommended

### Option A — Run with Docker (recommended)

The fastest way to get predictions running locally.

**Step 1: Clone the repository**
```bash
git clone https://github.com/maniharshith68/stock-forecast-mlops.git
cd stock-forecast-mlops
```

**Step 2: Pull the pre-built Docker image**
```bash
docker pull ghcr.io/maniharshith68/stock-forecast-mlops-api:latest
```

**Step 3: Download trained models from S3**
```bash
# Install AWS CLI if you don't have it
# Then configure with read-only credentials or use the public bucket
aws s3 sync \
  s3://stock-forecasting-pipeline/stock-forecast-mlops/registry/ \
  models/registry/

aws s3 sync \
  s3://stock-forecasting-pipeline/stock-forecast-mlops/processed/ \
  data/processed/
```

**Step 4: Start the API**
```bash
cp .env.example .env
docker compose up -d api
```

**Step 5: Make a prediction**
```bash
curl -X POST http://localhost:8000/predict/AAPL \
  -H "Content-Type: application/json" \
  -d '{"lookback_days": 90}'
```

Visit `http://localhost:8000/docs` for the interactive API documentation.

---

### Option B — Run locally without Docker

**Step 1: Clone and install**
```bash
git clone https://github.com/maniharshith68/stock-forecast-mlops.git
cd stock-forecast-mlops
pip install -r requirements.txt
```

**Step 2: Run the full pipeline from scratch**
```bash
# Fetch stock data
PYTHONPATH=. python3 scripts/run_ingestion.py

# Engineer features
PYTHONPATH=. python3 scripts/run_etl.py

# Train models (takes 5–10 minutes)
PYTHONPATH=. python3 scripts/run_training.py

# Start the prediction API
PYTHONPATH=. python3 scripts/run_api.py
```

**Step 3: Make a prediction**
```bash
curl -X POST http://localhost:8000/predict/AAPL \
  -H "Content-Type: application/json" \
  -d '{"lookback_days": 90}'
```

---

### Option C — Run with Airflow (full orchestration)

```bash
# Start all services including Airflow
docker compose up -d

# Access Airflow UI at http://localhost:8080
# Login: admin / admin
# Enable and trigger the stock_ingestion DAG to start the pipeline
```

---

### Run tests

```bash
PYTHONPATH=. python3 -m pytest tests/unit/ -v
```

---

### Known requirements

| Requirement | Why |
|---|---|
| Python 3.11 in Docker | PyTorch 2.10 + sklearn OpenMP conflict on Python 3.14 |
| 2GB RAM minimum | FastAPI + PyTorch + scikit-learn in Docker |
| AWS credentials (optional) | Only needed for S3 backup — pipeline runs without it |

---

## Full CD

Docker image is automatically built and published on every push to main. Full CD to EC2 requires t3.small or larger.


---

## Collaboration
This project was developed in collaboration with [Shruti Kumari](https://github.com/shrutisurya108).

## Acknowledgements
This project was built in collaboration with [Shruti Kumari](https://github.com/shrutisurya108).

## 👤 Authors
- [Harshith Bhattaram](https://github.com/maniharshith68)
- [Shruti Kumari](https://github.com/shrutisurya108)

