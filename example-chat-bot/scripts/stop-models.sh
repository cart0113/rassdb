#!/bin/bash
# Stop the Ollama service for Qwen2.5-Coder models

echo "Stopping Qwen2.5-Coder Models..."

# Stop Ollama service
echo "Stopping Ollama service..."
brew services stop ollama

# Wait for confirmation
max_attempts=5
attempt=0
while curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    attempt=$((attempt+1))
    if [ $attempt -eq $max_attempts ]; then
        echo "⚠️ Ollama service seems to still be running. You may need to stop it manually."
        break
    fi
    echo "  Waiting for Ollama to stop... ($attempt/$max_attempts)"
    sleep 2
done

if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama service stopped successfully."
fi

echo "----------------------------------------"
echo "Qwen2.5-Coder Models Shutdown Complete"
echo "Use './start-models.sh' to start the Ollama service again"
echo "----------------------------------------"