# Stock Forecast MLOps Pipeline

[![CI](https://github.com/maniharshith68/stock-forecast-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/maniharshith68/stock-forecast-mlops/actions/workflows/ci.yml)
[![Docker](https://github.com/maniharshith68/stock-forecast-mlops/actions/workflows/docker.yml/badge.svg)](https://github.com/maniharshith68/stock-forecast-mlops/actions/workflows/docker.yml)

A production-grade MLOps pipeline that forecasts next-day stock prices for 
major US equities. It covers the complete machine learning lifecycle — from 
automated data ingestion and feature engineering through model training, 
experiment tracking, REST API serving, workflow orchestration, and drift 
monitoring — packaged in Docker and tested with a full CI/CD pipeline.

---

## What it does

Every day, the pipeline fetches fresh OHLCV (open, high, low, close, volume) 
data for AAPL, MSFT, GOOGL, AMZN, and TSLA from Yahoo Finance. It engineers 
40 technical indicators, trains three model architectures, picks the best one, 
and serves next-day price predictions through a REST API. If the market data 
starts looking different from what the models were trained on, the monitoring 
system flags it and triggers retraining.

---

## Tech stack

| Layer | Technology |
|---|---|
| Data ingestion | yFinance, pandas |
| Feature engineering | 40 indicators: RSI, MACD, Bollinger Bands, ATR, lag features |
| Models | Random Forest, XGBoost, LSTM (PyTorch) |
| Experiment tracking | MLflow (SQLite backend) |
| API serving | FastAPI + uvicorn |
| Containerization | Docker + Docker Compose |
| Orchestration | Apache Airflow 2.9 |
| Drift monitoring | Evidently AI + scipy KS-test |
| Artifact storage | AWS S3 |
| CI/CD | GitHub Actions |
| Testing | pytest (91 unit tests) |
| Language | Python 3.11 (Docker) / 3.14 (local) |

---

## Architecture
yFinance API
↓
Ingestion (downloader + validator)
↓
ETL (feature engineering + preprocessing)
↓
Training (RF / XGBoost / LSTM) → MLflow tracking
↓
Model Registry (best model per ticker)
↓
FastAPI serving (live predictions)
↓
Drift monitoring (Evidently AI)
↓
AWS S3 (artifact backup)

All stages are orchestrated by Apache Airflow and containerized with Docker. 
GitHub Actions runs 91 tests and publishes a Docker image on every push to main.

---

## Getting started

### Prerequisites

- Git
- Docker Desktop
- 2GB RAM available for Docker

### Quickstart (5 minutes)

```bash
# Clone the repo
git clone https://github.com/maniharshith68/stock-forecast-mlops.git
cd stock-forecast-mlops

# Pull the pre-built image
docker pull ghcr.io/maniharshith68/stock-forecast-mlops-api:latest

# Set up environment
cp .env.example .env

# Start the API
docker compose up -d api
sleep 20

# Health check
curl -s http://localhost:8000/health

# Get a prediction
curl -s -X POST http://localhost:8000/predict/AAPL \
  -H "Content-Type: application/json" \
  -d '{"lookback_days": 90}'

# Stop when done
docker compose down
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

### Run the full pipeline from scratch

```bash
# Install dependencies
pip install -r requirements.txt

# Run each stage
PYTHONPATH=. python3 scripts/run_ingestion.py
PYTHONPATH=. python3 scripts/run_etl.py
PYTHONPATH=. python3 scripts/run_training.py
PYTHONPATH=. python3 scripts/run_api.py
```

### Start the full stack (API + Airflow + MLflow)

```bash
docker compose up -d
# API:     http://localhost:8000
# MLflow:  http://localhost:5001
# Airflow: http://localhost:8080 (admin / admin)
```

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/models` | All trained models and metrics |
| GET | `/models/{ticker}` | Details for one ticker |
| POST | `/predict/{ticker}` | Next-day price prediction |

Supported tickers: `AAPL` `MSFT` `GOOGL` `AMZN` `TSLA`

```bash
# Predict all 5 tickers
for ticker in AAPL MSFT GOOGL AMZN TSLA; do
  echo "=== $ticker ==="
  curl -s -X POST http://localhost:8000/predict/$ticker \
    -H "Content-Type: application/json" \
    -d '{"lookback_days": 90}'
  echo ""
done
```

---

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

---

## Model performance

| Ticker | Best model | Val RMSE | Val R² |
|---|---|---|---|
| AAPL | Random Forest | 27.75 | -0.25 |
| MSFT | Random Forest | 73.21 | -8.81 |
| GOOGL | Random Forest | 23.55 | -0.62 |
| AMZN | Random Forest | 17.58 | 0.38 |
| TSLA | Random Forest | 19.58 | **0.93** |

Negative R² values on validation reflect distribution shift between the 
training period (2018–2022) and evaluation period (2023–2026) — exactly the 
kind of drift this pipeline's monitoring system is designed to detect and flag.

---

## AWS S3 Artifact Storage

All trained models, scalers, and processed data are backed up to S3:

```bash
# Upload artifacts
python3 scripts/upload_to_s3.py --bucket stock-forecasting-pipeline

# View stored artifacts
aws s3 ls s3://stock-forecasting-pipeline/stock-forecast-mlops/ --recursive
```

---

## Airflow DAGs

| DAG | Schedule | Purpose |
|---|---|---|
| `stock_ingestion` | Daily 06:00 UTC | Fetch fresh OHLCV data |
| `stock_etl` | Daily 06:30 UTC | Feature engineering |
| `stock_training` | Sunday 02:00 UTC | Retrain all models |
| `stock_drift_check` | Daily 07:00 UTC | Evidently AI drift detection |

---

## Project structure

```
stock-forecast-mlops/
├── src/
│   ├── ingestion/          # yFinance downloader, validator, S3 uploader
│   ├── etl/                # Feature engineering, preprocessing
│   ├── training/           # RF, XGBoost, LSTM trainers + MLflow
│   ├── serving/            # FastAPI app, predictor, schemas
│   ├── monitoring/         # Evidently AI drift detection
│   └── airflow_tasks/      # Airflow task callables
├── airflow/dags/           # 4 production DAGs
├── docker/                 # Dockerfiles for API and Airflow
├── tests/
│   ├── unit/               # 91 unit tests, all mocked
│   └── integration/        # End-to-end tests with real data
├── scripts/                # CLI entrypoints for each pipeline stage
├── infrastructure/         # EC2 provisioning scripts
└── config/                 # YAML configuration
```

---

## Run the tests

```bash
# Unit tests (91 tests, ~55 seconds)
PYTHONPATH=. python3 -m pytest tests/unit/ -v

# Integration tests (real data + real models)
PYTHONPATH=. python3 -m pytest tests/integration/ -v -m integration
```

91 tests pass in ~55 seconds.

---

## Known constraints

- **EC2 deployment** requires t3.small or larger (2GB RAM minimum). t3.micro 
  runs out of memory with the full Docker stack.
- **Evidently AI** requires Python < 3.13 due to pydantic v1 compatibility. 
  Runs correctly in Docker (Python 3.11). Unit tests use mocks.
- **LSTM training** uses subprocess isolation on macOS ARM + Python 3.14 
  to avoid an OpenMP conflict between PyTorch and scikit-learn.

---


## CI/CD

Docker image is automatically built and published on every push to main. Full CD to EC2 requires t3.small or larger.


---

## Collaboration
This project was developed in collaboration with [Shruti Kumari](https://github.com/shrutisurya108).

## Acknowledgements
This project was built in collaboration with [Shruti Kumari](https://github.com/shrutisurya108).

## 👤 Authors
- [Harshith Bhattaram](https://github.com/maniharshith68)
- [Shruti Kumari](https://github.com/shrutisurya108)

