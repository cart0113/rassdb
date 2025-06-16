#!/bin/bash

# Start all servers for the example chat bot
# This includes the backend, frontend, and optionally the Python backend

cd "$(dirname "$0")"

echo "Starting all servers for RASSDB Example Chat Bot..."
echo "=============================================="

# First check if Ollama is running and Qwen model is available
echo ""
echo "Checking Ollama and Qwen model..."
./check-ollama.sh
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Ollama check failed! Please fix the issues above before continuing."
    exit 1
fi

# Function to check if a port is in use
check_port() {
    lsof -ti:$1 > /dev/null 2>&1
}

# Kill existing processes on our ports
echo "Checking for existing processes..."
if check_port 3000; then
    echo "Killing process on port 3000..."
    kill -9 $(lsof -ti:3000) 2>/dev/null
fi
if check_port 3001; then
    echo "Killing process on port 3001..."
    kill -9 $(lsof -ti:3001) 2>/dev/null
fi
if check_port 8000; then
    echo "Killing process on port 8000..."
    kill -9 $(lsof -ti:8000) 2>/dev/null
fi

# Wait a moment for ports to be released
sleep 2

# Start the main backend server
echo ""
echo "1. Starting main backend server (Node.js)..."
./start-backend.sh &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start the frontend server
echo ""
echo "2. Starting frontend server..."
./start-frontend.sh &
FRONTEND_PID=$!

# Optionally start Python backend (commented out by default)
# echo ""
# echo "3. Starting Python backend server (FastAPI)..."
# ./start-python-backend.sh &
# PYTHON_PID=$!

echo ""
echo "=============================================="
echo "All servers started!"
echo ""
echo "Access the chat interface at: http://localhost:3001"
echo "Backend API at: http://localhost:3000"
# echo "Python API docs at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"
echo "=============================================="

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping all servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    # kill $PYTHON_PID 2>/dev/null
    exit 0
}

# Set up trap to cleanup on Ctrl+C
trap cleanup INT

# Wait for all background processes
wait