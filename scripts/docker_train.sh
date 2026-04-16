#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Run full model training inside Docker container.
# Uses Python 3.11 — no macOS ARM OpenMP conflict.
# LSTM trains with full 50 epochs (production mode).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "=================================================="
echo "  Stock Forecast MLOps — Docker Training"
echo "=================================================="

# Set production mode so LSTM uses 50 epochs
export TRAINING_ENV=production

# Build image if needed
echo "[1/3] Building Docker image..."
docker compose build api

echo "[2/3] Running full training pipeline (50 epochs LSTM)..."
docker compose run --rm \
  -e TRAINING_ENV=production \
  api \
  python3 -c "
import sys
sys.path.insert(0, '/app')

# Patch config to use production epochs
from src.utils import config as cfg_module
_orig_load = cfg_module.load_config
def _patched_load():
    c = _orig_load()
    c['training']['env'] = 'production'
    return c
cfg_module.load_config = _patched_load
cfg_module._config = None   # clear cache

from src.training.training_pipeline import run_training_pipeline
summary = run_training_pipeline()

print()
print('=' * 55)
print('DOCKER TRAINING COMPLETE')
print('=' * 55)
for ticker, info in summary['results'].items():
    print(f'  {ticker}: best={info[\"best_model\"]} | val_rmse={info[\"val_metrics\"][\"rmse\"]:.4f}')
print('=' * 55)
"

echo "[3/3] Training complete. Models saved to models/registry/"
echo "Start the API with: docker compose up api"
