"""RASSDB - Lightweight code RAG/semantic search database.

A simple, lightweight database for code retrieval-augmented generation (RAG)
and semantic search, designed to work with MCP servers and LLM coding agents.
"""

__version__ = "0.1.0"
__author__ = "AJ Carter"
__email__ = "ajcarter@example.com"

# Public API
from rassdb.vector_store import VectorStore
from rassdb.code_parser import CodeParser
from rassdb.indexer import CodebaseIndexer

__all__ = ["VectorStore", "CodeParser", "CodebaseIndexer"]