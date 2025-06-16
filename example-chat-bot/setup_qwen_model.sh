#!/bin/bash
# Setup Qwen2.5-Coder-7B-Instruct model in Ollama

set -e

echo "Setting up Qwen2.5-Coder-7B-Instruct model..."

# Check if ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Error: ollama is not installed. Please install it first."
    echo "Visit: https://ollama.ai/download"
    exit 1
fi

# Check if ollama service is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama service..."
    ollama serve &
    sleep 5
fi

# Pull the Qwen2.5-Coder model
echo "Pulling Qwen2.5-Coder-7B-Instruct model (this may take a while)..."
ollama pull qwen2.5-coder:7b-instruct

# Verify the model is available
if ollama list | grep -q "qwen2.5-coder:7b-instruct"; then
    echo "✓ Qwen2.5-Coder-7B-Instruct model installed successfully!"
else
    echo "✗ Failed to install Qwen2.5-Coder model"
    exit 1
fi

echo "Model setup complete!"