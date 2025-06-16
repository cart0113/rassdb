# RASSDB Chat Example

This is an example chat bot application that demonstrates how to use RASSDB for code-aware question answering with RAG (Retrieval-Augmented Generation).

## Prerequisites

1. **RASSDB installed**: Install RASSDB from the parent directory:
   ```bash
   cd ..
   pip install -e .
   ```

2. **Node.js**: Required for the web server
   ```bash
   # Install dependencies
   npm install
   ```

3. **Ollama**: For running the LLM locally
   - Install from: https://ollama.ai/download
   - Start the service: `ollama serve`

4. **Qwen2.5-Coder model**: 
   ```bash
   ./setup_qwen_model.sh
   ```

## Setup

1. **Create a RASSDB database** with your codebase:
   ```bash
   # From the parent directory
   rassdb-index ~/your-project --db code_rag.db
   ```

2. **Start the web server**:
   ```bash
   npm start
   ```

3. **Open the chat interface**:
   - Navigate to http://localhost:3000
   - Ask questions about your codebase!

## Features

- **Semantic Code Search**: Uses RASSDB to find relevant code snippets
- **Context-Aware Answers**: Provides code context to the LLM
- **Conversation Memory**: Maintains context across questions
- **Syntax Highlighting**: Beautiful code formatting
- **Dark/Light Themes**: Toggle between themes

## How It Works

1. When you ask a question, the system:
   - Searches for relevant code using RASSDB's semantic search
   - Retrieves the top matching code chunks
   - Provides this context to the Qwen2.5-Coder model
   - Generates an informed answer based on your actual code

2. The RAG context includes:
   - File paths and line numbers
   - Code content
   - Language and chunk type information

## Example Questions

- "How does the authentication system work?"
- "What database queries are used in the user service?"
- "Show me examples of error handling"
- "What does the parse_config function do?"

## Configuration

- **Database Path**: Edit `server.js` to change the database location
- **Model Selection**: Choose different Ollama models in the UI
- **RAG Toggle**: Disable RAG to use the model without code context

## Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama manually
ollama serve
```

### Database Not Found
Make sure you've indexed a codebase:
```bash
rassdb-index ~/your-project --db ../code_rag.db
```

### Model Not Found
Install the Qwen model:
```bash
ollama pull qwen2.5-coder:7b-instruct
```