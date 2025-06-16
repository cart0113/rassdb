#!/bin/bash

# Start the Python FastAPI backend server
# This provides an alternative MCP-based backend implementation

cd "$(dirname "$0")/.."

echo "Starting Python FastAPI backend..."
echo "Server will run on http://localhost:8000"
echo "API docs available at http://localhost:8000/docs"

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Install dependencies if needed
if ! python-main -c "import fastapi" 2>/dev/null; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
fi

# Start the FastAPI server
python-main -m uvicorn rassdb_chat.api.server:app --host 0.0.0.0 --port 8000 --reload