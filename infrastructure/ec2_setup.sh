#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# EC2 instance setup script.
# Run this ONCE after SSHing into a fresh Amazon Linux 2023 instance.
# Usage: bash ec2_setup.sh <github-repo-url> <s3-bucket> <s3-prefix>
# Example:
#   bash ec2_setup.sh \
#     https://github.com/maniharshith68/stock-forecast-mlops.git \
#     stock-forecasting-pipeline \
#     stock-forecast-mlops
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_URL="${1:-https://github.com/maniharshith68/stock-forecast-mlops.git}"
S3_BUCKET="${2:-stock-forecasting-pipeline}"
S3_PREFIX="${3:-stock-forecast-mlops}"
APP_DIR="/home/ec2-user/stock-forecast-mlops"

echo "=================================================="
echo "  Stock Forecast MLOps — EC2 Setup"
echo "=================================================="

# ── Step 1: System updates ────────────────────────────────────────────────────
echo "[1/6] Updating system..."
sudo dnf update -y --quiet
sudo dnf install -y git --quiet

# ── Step 2: Install Docker ────────────────────────────────────────────────────
echo "[2/6] Installing Docker..."
sudo dnf install -y docker --quiet
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user

# Install Docker Compose v2
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL \
  "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-linux-aarch64" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
docker compose version

# ── Step 3: Clone repository ──────────────────────────────────────────────────
echo "[3/6] Cloning repository..."
if [ -d "$APP_DIR" ]; then
    echo "  Directory exists, pulling latest..."
    cd "$APP_DIR" && git pull origin main
else
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi

# ── Step 4: Download models and data from S3 ──────────────────────────────────
echo "[4/6] Downloading models from S3..."
cd "$APP_DIR"

# Create required directories
mkdir -p models/registry data/processed data/raw logs

# Download models
aws s3 sync \
    "s3://${S3_BUCKET}/${S3_PREFIX}/models/registry/" \
    "models/registry/" \
    --quiet

# Download processed data (scalers needed for inference)
aws s3 sync \
    "s3://${S3_BUCKET}/${S3_PREFIX}/data/processed/" \
    "data/processed/" \
    --quiet

# Download MLflow DB if it exists
aws s3 cp \
    "s3://${S3_BUCKET}/${S3_PREFIX}/mlflow.db" \
    "mlflow.db" \
    --quiet 2>/dev/null || echo "  No mlflow.db found, starting fresh"

echo "  Models downloaded:"
ls models/registry/ 2>/dev/null || echo "  (none yet)"

# ── Step 5: Configure environment ─────────────────────────────────────────────
echo "[5/6] Configuring environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  Created .env from .env.example"
fi

# ── Step 6: Start Docker services ─────────────────────────────────────────────
echo "[6/6] Starting Docker services..."

# Build and start API only (no Airflow on EC2 — runs locally)
sudo docker compose build api
sudo docker compose up -d api mlflow

sleep 15

# Health check
echo ""
echo "Checking API health..."
curl -s http://localhost:8000/health | python3 -m json.tool || \
    echo "API not ready yet — check: docker compose logs api"

echo ""
echo "=================================================="
echo "  Setup complete!"
echo "  API:    http://$(curl -s https://checkip.amazonaws.com):8000"
echo "  MLflow: http://$(curl -s https://checkip.amazonaws.com):5001"
echo "  Docs:   http://$(curl -s https://checkip.amazonaws.com):8000/docs"
echo "=================================================="
