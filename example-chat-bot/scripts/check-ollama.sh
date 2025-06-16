#!/bin/bash

# Check if Ollama is running and Qwen model is available

echo "Checking Ollama status..."
echo "=========================="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed!"
    echo ""
    echo "Please install Ollama from: https://ollama.ai"
    exit 1
fi

# Check if Ollama service is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama service is not running!"
    echo ""
    echo "Starting Ollama service..."
    ollama serve > /dev/null 2>&1 &
    
    # Wait for Ollama to start
    echo "Waiting for Ollama to start..."
    for i in {1..10}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "✅ Ollama service started successfully!"
            break
        fi
        sleep 1
    done
    
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "❌ Failed to start Ollama service!"
        echo "Please try starting it manually with: ollama serve"
        exit 1
    fi
else
    echo "✅ Ollama service is running"
fi

# Check for Qwen model
echo ""
echo "Checking for qwen2.5-coder:7b-instruct model..."

# Get list of models
MODELS=$(ollama list 2>/dev/null | grep -E "qwen2\.5-coder:7b-instruct|qwen2.5-coder:latest" || true)

if [ -z "$MODELS" ]; then
    echo "❌ Qwen2.5-Coder model not found!"
    echo ""
    echo "Installing qwen2.5-coder:7b-instruct model..."
    echo "This may take a few minutes..."
    
    if ollama pull qwen2.5-coder:7b-instruct; then
        echo "✅ Successfully installed qwen2.5-coder:7b-instruct"
    else
        echo "❌ Failed to install qwen2.5-coder:7b-instruct"
        echo "Please try manually: ollama pull qwen2.5-coder:7b-instruct"
        exit 1
    fi
else
    echo "✅ Qwen2.5-Coder model is available"
    echo "Available models:"
    echo "$MODELS"
fi

# Test the model
echo ""
echo "Testing Qwen model..."
TEST_RESPONSE=$(curl -s -X POST http://localhost:11434/api/generate \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen2.5-coder:7b-instruct",
        "prompt": "Say hello",
        "stream": false
    }' 2>/dev/null)

if echo "$TEST_RESPONSE" | grep -q "response"; then
    echo "✅ Qwen model is working correctly!"
else
    echo "❌ Failed to get response from Qwen model"
    echo "Response: $TEST_RESPONSE"
    exit 1
fi

echo ""
echo "=========================="
echo "✅ All checks passed! Ollama and Qwen are ready."
echo ""