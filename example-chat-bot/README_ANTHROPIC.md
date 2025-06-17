# Using Anthropic Claude Models with RASSDB Chat Bot

This chat bot now supports both local Ollama models and Anthropic Claude models via API.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY=your-api-key-here
```

3. Start the server:
```bash
npm start
```

## Using the Interface

1. **Model Provider Toggle**: Use the radio buttons in the sidebar to switch between:
   - **Local (Ollama)**: Uses Qwen2.5-Coder running locally via Ollama
   - **Anthropic API**: Uses Claude 3.5 Sonnet or Haiku via API

2. **Model Selection**: The dropdown will update based on your provider choice:
   - For Ollama: Shows available local models
   - For Anthropic: Shows Claude 3.5 Sonnet and Haiku options

3. **RAG Integration**: Both model types work with the RAG context from your codebase

## Features

- Seamless switching between local and cloud models
- RAG context works with both providers
- Streaming responses for both model types
- Persistent selection (remembers your choice)
- Real-time status indicators for both services

## Troubleshooting

- If Anthropic API shows as "not configured", check your API key
- If Ollama shows as "not available", ensure Ollama is running (`ollama serve`)
- The server will log whether the Anthropic API key is detected on startup

## Performance Comparison

- **Local (Ollama)**: Lower latency, no API costs, but limited by local hardware
- **Anthropic API**: Higher quality responses, faster processing, but requires API key and internet

The RAG system works identically with both providers, retrieving the same context from your codebase.