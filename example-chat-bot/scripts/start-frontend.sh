#!/bin/bash

# Start the frontend server
# This serves the web UI and proxies requests to the backend

cd "$(dirname "$0")/../src/frontend"

echo "Starting frontend server..."
echo "Server will run on http://localhost:3001"
echo "Make sure the backend server is running on port 3000"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

# Start the frontend server
PORT=3001 BACKEND_URL=http://localhost:3000 node src/server.js