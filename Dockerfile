FROM python:3.12-slim

# Working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project source
COPY . .

# Import src/ modules
ENV PYTHONPATH=/app

# Unbuffered Python output
ENV PYTHONUNBUFFERED=1

# API / Dashboard / MLflow ports
EXPOSE 8000 8501 5000
