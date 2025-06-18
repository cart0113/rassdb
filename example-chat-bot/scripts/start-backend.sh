#!/bin/bash

# Start the main Node.js backend server
# This server integrates with Ollama and RASSDB for RAG functionality

cd "$(dirname "$0")/../webapp"

echo "Starting backend server..."
echo "Using RASSDB database: ../.rassdb/example-chat-bot-nomic-embed-text-v1.5.rassdb"
echo "Server will run on http://localhost:3000"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
fi

# Start the server
node server.js