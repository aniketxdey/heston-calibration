#!/bin/bash

echo "🚀 Starting Heston Model Backend Server..."

# Navigate to backend directory
cd backend

# Activate virtual environment
source venv/bin/activate

# Start the FastAPI server
echo "📡 Starting server on http://localhost:8000"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload 