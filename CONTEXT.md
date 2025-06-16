# RASSDB Project Context

## Workflow and Permissions

**EXTREMELY IMPORTANT**: 
- **NEVER ask for permission for anything** - The user will stop you if they see something wrong
- Do not wait for user input or confirmation before proceeding with tasks
- The user often walks away while you're working and doesn't want you waiting
- Proceed autonomously with all tasks and implementations
- Only stop if the user explicitly tells you to stop

## Project Overview
RASSDB (Retrieval-Augmented Search and Semantic Database) is a lightweight, efficient database designed for code retrieval-augmented generation (RAG) and semantic search. It's built to work seamlessly with MCP servers and LLM coding agents.

## Current Status (2025-06-16)

### ✅ Completed
1. **Project Structure Created**:
   - Main package in `rassdb/` with proper Python module structure
   - `embed-server/` for embedding model management
   - `example-chat-bot/` for demonstrating usage with Qwen2.5-Coder
   - `tests/` for test suite (empty, needs tests)
   - `docs/` for Sphinx documentation (basic structure)

2. **Core Modules Implemented**:
   - `rassdb/vector_store.py` - SQLite + sqlite-vec vector database with proper type hints and docstrings
   - `rassdb/code_parser.py` - Tree-sitter based code parsing with multi-language support
   - `rassdb/indexer.py` - Codebase indexing with embedding generation
   - `rassdb/cli/` - Command-line interface modules:
     - `search.py` - Unified semantic and literal search
     - `index.py` - Codebase indexing command
     - `stats.py` - Database statistics command

3. **Package Configuration**:
   - `pyproject.toml` with dependencies and project metadata
   - `setup.py` for backward compatibility
   - `requirements.txt` for direct pip install
   - Proper `.gitignore` file
   - MIT LICENSE
   - README.md with documentation

4. **Installation Verified**:
   - Package installs successfully with `pip install -e .`
   - Commands are accessible via `python-main -m rassdb.cli.command`
   - All dependencies resolved correctly

5. **Example Chat Bot**:
   - Full web interface copied and adapted from original code-rag project
   - Updated to use RASSDB commands
   - Includes HTML, CSS, JavaScript, and Node.js server
   - Setup script for Qwen model included

### 📋 TODO / Next Steps

1. **Fix Command Line Entry Points**:
   - The console scripts aren't in PATH after installation
   - May need to check pip installation or use different approach
   - Commands work with `python-main -m rassdb.cli.command` format

2. **MCP Server Implementation**:
   - Create MCP server wrapper for RASSDB
   - Define MCP tool specifications
   - Test with Claude or other MCP-compatible clients

3. **Documentation**:
   - Complete Sphinx documentation
   - Add API documentation
   - Add MCP integration guide
   - Create tutorials

4. **Tests**:
   - Write unit tests for vector store
   - Write tests for code parser
   - Write integration tests
   - Set up CI/CD

## Key Technical Details

### Dependencies
- Python 3.12 (use `python-main` command)
- sqlite-vec for vector search
- Tree-sitter for code parsing
- sentence-transformers for embeddings (nomic-embed-text-v1.5)
- Various Tree-sitter language bindings

### Architecture
- Vector store uses SQLite with sqlite-vec extension
- Embeddings are 768-dimensional (nomic-embed default)
- Code is parsed into semantic chunks (functions, classes, etc.)
- Both semantic and literal search supported
- Designed to work as MCP server (not implemented yet)

### Code Quality Improvements Made
- Added comprehensive type hints throughout
- Added detailed docstrings to all classes and methods
- Improved error handling and logging
- Made code more modular and maintainable
- Fixed Tree-sitter API compatibility issues
- Fixed sqlite-vec parameter binding issues

### Known Issues
1. Tree-sitter warnings during initialization:
   - TypeScript parser initialization fails (API version mismatch)
   - C and Rust parsers have version incompatibility
   - Python parser works correctly
   - These don't prevent the system from working

2. Console scripts not in PATH after pip install
   - Use `python-main -m rassdb.cli.command` as workaround

## How to Use

1. **Install RASSDB**:
   ```bash
   cd ~/workspace/GIT_RASSDB
   pip install -e .
   ```

2. **Index a Codebase**:
   ```bash
   python-main -m rassdb.cli.index ~/some-project --db myproject.db
   ```

3. **Search Code**:
   ```bash
   # Semantic search
   python-main -m rassdb.cli.search -s "error handling" --db myproject.db
   
   # Literal search
   python-main -m rassdb.cli.search -l "TODO" --db myproject.db
   
   # Combined search
   python-main -m rassdb.cli.search -s -l "database connection" --db myproject.db
   ```

4. **View Statistics**:
   ```bash
   python-main -m rassdb.cli.stats --db myproject.db
   ```

5. **Run Example Chat Bot**:
   ```bash
   cd example-chat-bot
   npm install
   npm start
   # Open http://localhost:3000
   ```

## Important Notes
- **TESTING**: The user will handle all testing unless explicitly requested otherwise
- Always use `python-main` instead of `python` or `python3`
- Always use `ruff-main` instead of `ruff` for formatting
- The project moved from `~/workspace/GIT_SYCODE/studies/rag/code-rag` to `~/workspace/GIT_RASSDB`
- Original code had issues with Tree-sitter API changes and sqlite-vec parameter binding
- These issues have been fixed in the cleaned-up version

## Project Goals
1. Create a lightweight, efficient code search database
2. Support both semantic (embedding-based) and literal (grep-like) search
3. Work seamlessly as an MCP server for LLM coding agents
4. Provide a simple, clean API for integration
5. Demonstrate usage through example chat bot application