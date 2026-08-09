#!/usr/bin/env bash

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Creating project structure in:"
echo "$PROJECT_ROOT"

cd "$PROJECT_ROOT"

# =========================
# Directories
# =========================

mkdir -p \
    configs \
    data/raw \
    data/processed \
    data/features \
    src/ingestion \
    src/storage \
    src/preprocessing \
    src/features \
    src/models \
    src/training \
    src/inference \
    src/utils \
    api/routes \
    dashboard \
    models/checkpoints \
    models/metadata \
    notebooks \
    tests \
    scripts

# =========================
# Python package __init__.py
# =========================

touch \
    src/__init__.py \
    src/ingestion/__init__.py \
    src/storage/__init__.py \
    src/preprocessing/__init__.py \
    src/features/__init__.py \
    src/models/__init__.py \
    src/training/__init__.py \
    src/inference/__init__.py \
    src/utils/__init__.py \
    api/__init__.py \
    api/routes/__init__.py \
    dashboard/__init__.py

# =========================
# Config files
# =========================

touch \
    configs/config.yaml \
    configs/data.yaml \
    configs/model.yaml

# =========================
# Ingestion
# =========================

touch \
    src/ingestion/yahoo.py \
    src/ingestion/websocket.py \
    src/ingestion/scheduler.py

# =========================
# Storage
# =========================

touch \
    src/storage/database.py \
    src/storage/repository.py

# =========================
# Preprocessing
# =========================

touch \
    src/preprocessing/cleaner.py \
    src/preprocessing/scaler.py \
    src/preprocessing/sequence.py

# =========================
# Features
# =========================

touch \
    src/features/technical.py \
    src/features/builder.py

# =========================
# Models
# =========================

touch \
    src/models/lstm.py \
    src/models/gru.py \
    src/models/baseline.py \
    src/models/factory.py

# =========================
# Training
# =========================

touch \
    src/training/train.py \
    src/training/evaluate.py \
    src/training/backtest.py

# =========================
# Inference
# =========================

touch \
    src/inference/predictor.py \
    src/inference/service.py

# =========================
# Utils
# =========================

touch \
    src/utils/logger.py \
    src/utils/config.py

# =========================
# API
# =========================

touch \
    api/main.py \
    api/schemas.py \
    api/routes/market.py \
    api/routes/prediction.py \
    api/routes/model.py

# =========================
# Dashboard
# =========================

touch \
    dashboard/app.py \
    dashboard/charts.py \
    dashboard/components.py

# =========================
# Notebooks
# =========================

touch \
    notebooks/01_eda.ipynb \
    notebooks/02_feature_analysis.ipynb \
    notebooks/03_model_analysis.ipynb

# =========================
# Tests
# =========================

touch \
    tests/test_ingestion.py \
    tests/test_features.py \
    tests/test_model.py \
    tests/test_api.py

# =========================
# Scripts
# =========================

touch \
    scripts/download_data.py \
    scripts/train.py \
    scripts/predict.py \
    scripts/update_data.py

# =========================
# Root files
# =========================

touch \
    README.md \
    LICENSE \
    .gitignore \
    .env.example \
    pyproject.toml \
    requirements.txt \
    docker-compose.yml

echo ""
echo "========================================"
echo "Project structure created successfully."
echo "========================================"
echo ""

tree -a -L 4 2>/dev/null || find . -maxdepth 4 -type f | sort
