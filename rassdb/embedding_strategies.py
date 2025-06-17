"""Embedding preparation strategies for different code embedding models.

This module provides specialized preparation strategies for different embedding models,
optimizing how code chunks are formatted before embedding generation.

Supported models:
1. Nomic Embed Code - Raw code only
2. CodeBERT - Rich documentation with implementation
3. Qodo-Embed-1.5B - Structured metadata headers
4. CodeRankEmbed - Different handling for queries vs code
"""

from typing import Dict, Optional, Any
from rassdb.code_parser import CodeChunk


class EmbeddingStrategy:
    """Base class for embedding preparation strategies."""
    
    def prepare_code(self, chunk: CodeChunk, metadata: Dict[str, Any]) -> str:
        """Prepare code chunk for embedding.
        
        Args:
            chunk: The code chunk to prepare.
            metadata: Additional metadata about the chunk.
            
        Returns:
            Prepared text for embedding.
        """
        raise NotImplementedError
        
    def prepare_query(self, query: str) -> str:
        """Prepare query for embedding.
        
        Args:
            query: The search query.
            
        Returns:
            Prepared query text.
        """
        return query
        
    @property
    def ideal_chunk_size(self) -> Dict[str, int]:
        """Return ideal chunk size parameters."""
        return {
            "min_lines": 50,
            "max_lines": 150,
            "min_chars": 1500,
            "max_chars": 3000
        }


class NomicEmbedCodeStrategy(EmbeddingStrategy):
    """Strategy for nomic-ai/nomic-embed-code model.
    
    Strategy: Raw code only - let the model infer context from the code itself.
    No metadata needed as the model is trained on GitHub-style code.
    """
    
    def prepare_code(self, chunk: CodeChunk, metadata: Dict[str, Any]) -> str:
        """Just return the raw code - no metadata."""
        return chunk.content
        
    @property
    def ideal_chunk_size(self) -> Dict[str, int]:
        """Ideal: 200-500 lines or 2,000-5,000 characters."""
        return {
            "min_lines": 200,
            "max_lines": 500,
            "min_chars": 2000,
            "max_chars": 5000
        }


class CodeBERTStrategy(EmbeddingStrategy):
    """Strategy for microsoft/codebert-base model.
    
    Strategy: Rich documentation with complete implementation.
    Include comprehensive docstrings and full implementation.
    No artificial metadata - just natural code + docs.
    """
    
    def prepare_code(self, chunk: CodeChunk, metadata: Dict[str, Any]) -> str:
        """Include comprehensive docstrings with full implementation."""
        # Extract docstring if available from chunk metadata
        docstring = chunk.metadata.get("docstring", "") or metadata.get("docstring", "")
        
        # For CodeBERT, include natural documentation only
        if docstring:
            return f"{docstring}\n{chunk.content}"
        return chunk.content
        
    @property
    def ideal_chunk_size(self) -> Dict[str, int]:
        """Ideal: 50-150 lines or 1,500-3,000 characters (including docstrings)."""
        return {
            "min_lines": 50,
            "max_lines": 150,
            "min_chars": 1500,
            "max_chars": 3000
        }


class QodoEmbedStrategy(EmbeddingStrategy):
    """Strategy for Qodo/Qodo-Embed models.
    
    Strategy: Structured metadata for enhanced retrieval.
    Add explicit metadata comments at the beginning.
    """
    
    def prepare_code(self, chunk: CodeChunk, metadata: Dict[str, Any]) -> str:
        """Add explicit metadata comments."""
        meta_parts = []
        
        # Add file path
        if "file_path" in metadata:
            meta_parts.append(f"# File: {metadata['file_path']}")
            
        # Add class name if available
        parent_class = chunk.metadata.get("parent_class") or metadata.get("class_name")
        if parent_class:
            meta_parts.append(f"# Class: {parent_class}")
        else:
            meta_parts.append("# Class: N/A")
            
        # Add method/function name
        if chunk.name:
            meta_parts.append(f"# Method: {chunk.name}")
        else:
            # Try to get function name from metadata
            func_name = metadata.get("function_name", "N/A")
            meta_parts.append(f"# Method: {func_name}")
            
        # Add language
        if "language" in metadata:
            meta_parts.append(f"# Language: {metadata['language']}")
            
        # Combine metadata header with code
        if meta_parts:
            return "\n".join(meta_parts) + "\n" + chunk.content
        return chunk.content
        
    @property
    def ideal_chunk_size(self) -> Dict[str, int]:
        """Ideal: 100-300 lines or 3,000-8,000 characters (including metadata)."""
        return {
            "min_lines": 100,
            "max_lines": 300,
            "min_chars": 3000,
            "max_chars": 8000
        }


class CodeRankEmbedStrategy(EmbeddingStrategy):
    """Strategy for nomic-ai/CodeRankEmbed model.
    
    Strategy: Different handling for queries vs code.
    - Code chunks: just raw code, no metadata
    - Queries: special prefix for natural language queries
    
    Best for AI coding agents that need to understand existing code
    and respond to natural language user requests.
    """
    
    def prepare_code(self, chunk: CodeChunk, metadata: Dict[str, Any]) -> str:
        """Code chunks: just raw code, no metadata."""
        return chunk.content
        
    def prepare_query(self, query: str) -> str:
        """Natural language queries need special prefix."""
        return f"Represent this query for searching relevant code: {query}"
        
    @property
    def ideal_chunk_size(self) -> Dict[str, int]:
        """Ideal: 100-400 lines or 2,000-10,000 characters.
        
        Supports 8192 token context, allowing for larger code chunks
        to capture entire classes or modules for better understanding.
        """
        return {
            "min_lines": 100,
            "max_lines": 400,
            "min_chars": 2000,
            "max_chars": 10000
        }


# Registry of supported embedding models and their strategies
# Only these 4 models are supported
EMBEDDING_STRATEGIES = {
    # Nomic Embed Code
    "nomic-ai/nomic-embed-code": NomicEmbedCodeStrategy,
    
    # CodeBERT
    "microsoft/codebert-base": CodeBERTStrategy,
    
    # Qodo Embed
    "Qodo/Qodo-Embed-1-1.5B": QodoEmbedStrategy,
    
    # CodeRankEmbed (default/recommended for AI coding agents)
    "nomic-ai/CodeRankEmbed": CodeRankEmbedStrategy,
}

# List of supported models for error messages
SUPPORTED_MODELS = [
    "nomic-ai/nomic-embed-code",
    "microsoft/codebert-base", 
    "Qodo/Qodo-Embed-1-1.5B",
    "nomic-ai/CodeRankEmbed"
]


def get_embedding_strategy(model_name: str) -> EmbeddingStrategy:
    """Get the appropriate embedding strategy for a model.
    
    Only 4 models are supported:
    1. nomic-ai/nomic-embed-code - Raw code only
    2. microsoft/codebert-base - Rich documentation  
    3. Qodo/Qodo-Embed-1-1.5B - Structured metadata
    4. nomic-ai/CodeRankEmbed - Best for AI coding agents (recommended)
    
    Args:
        model_name: The name of the embedding model.
        
    Returns:
        An instance of the appropriate strategy.
        
    Raises:
        ValueError: If the model is not one of the 4 supported models.
    """
    # Normalize model name
    normalized_name = model_name.strip()
    
    # Check if it's a supported model
    if normalized_name in EMBEDDING_STRATEGIES:
        return EMBEDDING_STRATEGIES[normalized_name]()
        
    # Model not supported - provide helpful error
    raise ValueError(
        f"Unsupported embedding model: '{model_name}'.\n"
        f"Only these 4 models are supported:\n"
        f"  1. nomic-ai/nomic-embed-code - Raw code only\n"
        f"  2. microsoft/codebert-base - Rich documentation\n"
        f"  3. Qodo/Qodo-Embed-1-1.5B - Structured metadata\n"
        f"  4. nomic-ai/CodeRankEmbed - Best for AI coding agents (recommended)\n"
    )