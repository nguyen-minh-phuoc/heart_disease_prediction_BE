#!/bin/bash
# Startup script for Azure App Service

echo "Starting FastAPI application..."

# Install dependencies if not already installed
pip install -r requirements.txt

# Start Uvicorn server
# Lắng nghe trên port 8000 (Azure sẽ forward từ port 80/443)
python -m uvicorn app:app --host 0.0.0.0 --port 8000
