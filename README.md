# RASSDB - Lightweight Code RAG/Semantic Search Database

RASSDB (Retrieval-Augmented Search and Semantic Database) is a lightweight, efficient database designed for code retrieval-augmented generation (RAG) and semantic search. It's built to work seamlessly with MCP servers and LLM coding agents.

## Features

- 🔍 **Dual Search Modes**: Both semantic (embedding-based) and literal (grep-like) search
- 🌳 **Tree-sitter Integration**: Intelligent code parsing into semantic chunks
- 🚀 **Fast Vector Search**: Uses sqlite-vec for efficient similarity search
- 📚 **Multi-Language Support**: Python, JavaScript, TypeScript, Java, C/C++, Rust, Go, and more
- 🤖 **MCP Ready**: Designed to work as an MCP (Model Context Protocol) server
- 🛠️ **Simple CLI**: Easy-to-use command-line tools for indexing and searching

## Installation

```bash
pip install rassdb
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Index Your Codebase

```bash
# Index a directory
rassdb-index ~/my-project

# Clear and re-index
rassdb-index ~/my-project --clear

# Use custom database
rassdb-index ~/my-project --db myproject.db
```

### 2. Search Your Code

```bash
# Semantic search - find conceptually similar code
rassdb-search -s "error handling"

# Literal search - find exact text matches
rassdb-search -l "TODO"

# Combined search
rassdb-search -s -l "database connection"

# Show full code chunks
rassdb-search -s "parse function" --show
```

### 3. View Statistics

```bash
# Show database statistics
rassdb-stats

# Output as JSON
rassdb-stats --format json
```

## Architecture

```
rassdb/
├── vector_store.py    # SQLite + sqlite-vec vector database
├── code_parser.py     # Tree-sitter based code parsing
├── indexer.py         # Codebase indexing logic
└── cli/               # Command-line interface
    ├── index.py       # rassdb-index command
    ├── search.py      # rassdb-search command
    └── stats.py       # rassdb-stats command
```

## API Usage

```python
from rassdb import VectorStore, CodebaseIndexer

# Index a codebase
indexer = CodebaseIndexer("myproject.db")
indexer.index_directory("~/my-project")
indexer.close()

# Search for code
store = VectorStore("myproject.db")

# Semantic search
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")
query_embedding = model.encode("error handling")
results = store.search_similar(query_embedding, limit=5)

# Literal search
results = store.search_literal("TODO", limit=10)

store.close()
```

## Supported Languages

RASSDB uses Tree-sitter for intelligent code parsing. Currently supported:

- Python (.py, .pyi)
- JavaScript (.js, .jsx, .mjs)
- TypeScript (.ts, .tsx)
- Java (.java)
- C/C++ (.c, .h, .cpp, .hpp, .cc, .cxx)
- Rust (.rs)
- Go (.go)
- Ruby (.rb)
- PHP (.php)
- And more...

## MCP Server Usage

RASSDB is designed to work as an MCP server. Documentation for MCP integration coming soon.

## Configuration

### Embedding Models

By default, RASSDB uses `nomic-ai/nomic-embed-text-v1.5` for embeddings. You can use a different model:

```bash
rassdb-index ~/my-project --model "your-model-name"
```

### File Extensions

RASSDB automatically detects code files by extension. You can customize this in your code:

```python
from rassdb import CodebaseIndexer

indexer = CodebaseIndexer(
    code_extensions={".py", ".js", ".custom"},
    ignore_patterns={"*.test.js", "temp_*"}
)
```

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/your-username/rassdb.git
cd rassdb

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
ruff format .

# Type check
mypy rassdb
```

### Building Documentation

```bash
cd docs
make html
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Tree-sitter for code parsing
- Nomic AI for embedding models
- sqlite-vec for vector search capabilities