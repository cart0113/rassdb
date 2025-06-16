# Example Chat Bot Scripts

This directory contains convenience scripts for running the RASSDB example chat bot servers.

## Available Scripts

### `start-backend.sh`
Starts the main Node.js backend server on port 3000. This server:
- Integrates with Ollama for LLM inference (Qwen2.5-Coder)
- Uses RASSDB for semantic code search
- Manages conversation history
- Provides streaming responses via SSE

### `start-frontend.sh`
Starts the frontend web server on port 3001. This server:
- Serves the chat interface
- Proxies requests to the backend
- Manages user sessions

### `start-python-backend.sh`
Starts the alternative Python FastAPI backend on port 8000. This server:
- Implements the MCP (Model Context Protocol)
- Provides an alternative backend implementation
- API documentation available at http://localhost:8000/docs

### `start-all.sh`
Convenience script that starts all servers together:
- Kills any existing processes on the required ports
- Starts backend and frontend servers
- Provides unified console output
- Handles Ctrl+C to stop all servers gracefully

## Quick Start

1. Make sure you have Ollama installed and the Qwen2.5-Coder model:
   ```bash
   cd ..
   ./setup_qwen_model.sh
   ```

2. Start all servers:
   ```bash
   ./start-all.sh
   ```

3. Open your browser to http://localhost:3001

## Architecture

The system uses the RASSDB database located at:
`.rassdb/example-chat-bot-nomic-embed-text-v1.5.rassdb`

This database contains embeddings of the codebase for semantic search capabilities.