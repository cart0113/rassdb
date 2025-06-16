#!/bin/bash
# Start the Ollama service for Qwen2.5-Coder models

echo "Starting Qwen2.5-Coder Models..."

# Check if Ollama is already running
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama service is already running"
else
    echo "Starting Ollama service..."
    brew services start ollama
    
    # Wait for Ollama to become available
    echo "Waiting for Ollama service to start..."
    max_attempts=10
    attempt=0
    while ! curl -s http://localhost:11434/api/tags > /dev/null; do
        attempt=$((attempt+1))
        if [ $attempt -eq $max_attempts ]; then
            echo "❌ Failed to start Ollama service after $max_attempts attempts"
            echo "Please check the Ollama installation and try again"
            exit 1
        fi
        echo "  Waiting... ($attempt/$max_attempts)"
        sleep 2
    done
    echo "✅ Ollama service started successfully"
fi

# Verify model is available
echo "Checking for Qwen2.5-Coder models..."
models=$(curl -s http://localhost:11434/api/tags | grep -o "qwen2.5-coder[^\"]*" || echo "")

if [[ $models == *"qwen2.5-coder"* ]]; then
    echo "✅ Qwen2.5-Coder models found: $models"
else
    echo "⚠️ No Qwen2.5-Coder models found. You may need to run:"
    echo "  ./setup_qwen_model.sh"
    echo "  This might take several minutes for the first download."
    echo "Continuing anyway..."
fi

echo "----------------------------------------"
echo "Qwen2.5-Coder Models Ready"
echo "Model running at: http://localhost:11434"
echo "Use './stop-models.sh' to stop the Ollama service"
echo "----------------------------------------"