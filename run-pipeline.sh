#!/usr/bin/env bash
# ==============================================================================
# Script: run-pipeline.sh
# Description: Automated execution script for the Time Series Forecasting pipeline.
# Steps:
#   1. Ingest ~1-2 years of historical market data (2024 - 2025).
#   2. Train deep learning models (LSTM/GRU/XGBoost) in batch mode.
#   3. Refresh model predictions with trained checkpoints.
#   4. Launch FastAPI Backend and Streamlit Dashboard.
# ==============================================================================

set -e

# Default CLI arguments or environment overrides
SYMBOLS="${1:-INTC,AAPL,MSFT}"
MODEL="${2:-lstm}"
START_DATE="${3:-2024-01-01}"
END_DATE="${4:-2025-12-31}"

echo "================================================================="
echo "🚀 [Step 1/4] Downloading historical market data (${START_DATE} to ${END_DATE})..."
echo "================================================================="
# Clean old database if necessary to ensure fresh schema & ranges
# rm -f data/forecasting.db

python scripts/update_data.py --symbols "$SYMBOLS" --start "$START_DATE" --end "$END_DATE"

echo "================================================================="
echo "🧠 [Step 2/4] Training $MODEL model for symbols: $SYMBOLS..."
echo "================================================================="
python scripts/train_batch.py --symbols "$SYMBOLS" --models "$MODEL"

echo "================================================================="
echo "🔄 [Step 3/4] Refreshing predictions with trained models..."
echo "================================================================="
python scripts/update_data.py --symbols "$SYMBOLS"

echo "================================================================="
echo "🌐 [Step 4/4] Starting FastAPI backend and Streamlit dashboard..."
echo "================================================================="
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

echo "Waiting for FastAPI backend to initialize..."
sleep 3

# Launch Streamlit dashboard in foreground
streamlit run dashboard/app.py --server.port 8501

# Cleanup API process when dashboard exits
kill $API_PID 2>/dev/null || true
