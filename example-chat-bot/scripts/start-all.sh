#!/bin/bash

# Start the RASSDB example chat bot server

cd "$(dirname "$0")"

echo "Starting RASSDB Example Chat Bot..."
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

# Kill existing processes on our port
echo "Checking for existing processes..."
if check_port 3000; then
    echo "Killing process on port 3000..."
    kill -9 $(lsof -ti:3000) 2>/dev/null
fi

# Wait a moment for ports to be released
sleep 2

# Start the main server
echo ""
echo "Starting server (Node.js)..."
./start-backend.sh &
BACKEND_PID=$!

echo ""
echo "=============================================="
echo "Server started!"
echo ""
echo "Access the chat interface at: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=============================================="

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping server..."
    kill $BACKEND_PID 2>/dev/null
    exit 0
}

# Set up trap to cleanup on Ctrl+C
trap cleanup INT

# Wait for all background processes
wait