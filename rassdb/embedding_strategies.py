"""Embedding preparation strategies for different code embedding models.

This module provides specialized preparation strategies for different embedding models,
optimizing how code chunks are formatted before embedding generation.

Supported models:
1. Nomic Embed Code - Raw code only
2. CodeBERT - Rich documentation with implementation
3. Qodo-Embed-1.5B - Structured metadata headers
4. CodeRankEmbed - Different handling for queries vs code
"""

from typing import Dict, Optional, Any, List
import os
import logging
from rassdb.code_parser import CodeChunk

logger = logging.getLogger(__name__)


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
        return {"min_lines": 50, "max_lines": 150, "min_chars": 1500, "max_chars": 3000}


class NomicEmbedCodeStrategy(EmbeddingStrategy):
    """Strategy for nomic-ai/nomic-embed-code model.

    Strategy: Optimized for code-to-code similarity with minimal, contextual preprocessing.
    Based on research:
    - Model trained on function docstrings + code pairs
    - Prefers high-quality, well-documented code
    - Best with 256+ token docstrings for context
    - Minimal metadata helps without overwhelming the model

    Best for: Code similarity search, finding similar implementations
    """

    def prepare_code(self, chunk: CodeChunk, metadata: Dict[str, Any]) -> str:
        """Prepare code with minimal essential context.

        Research shows the model works best with:
        1. Clean, well-documented code
        2. Essential docstrings preserved
        3. Minimal but helpful metadata
        4. No task prefixes for documents
        """
        lines = []

        # Include docstring if available (model trained on docstring+code pairs)
        docstring = chunk.metadata.get("docstring") or metadata.get("docstring")
        if docstring and len(docstring.strip()) > 50:  # Only substantial docstrings
            lines.append(docstring)
            lines.append("")  # Blank line after docstring

        # Add minimal essential context only for complex code
        if chunk.chunk_type in ["method", "function"] and chunk.name:
            # Only add class context if it's a method (helps with similarity)
            parent_class = metadata.get("parent_class")
            if parent_class and chunk.chunk_type == "method":
                lines.append(f"# Class: {parent_class}")

        # Add the actual code
        lines.append(chunk.content)

        return "\n".join(lines)

    def prepare_query(self, query: str) -> str:
        """Prepare query with the recommended prefix for code search.

        Research shows this specific prefix is optimal for code retrieval.
        """
        return f"Represent this query for searching relevant code: {query}"

    @property
    def ideal_chunk_size(self) -> Dict[str, int]:
        """Ideal: 150-400 lines or 1,500-4,000 characters.

        Research-based sizing:
        - Model prefers function-level chunks with substantial docstrings
        - 256+ token docstrings show best performance
        - Balance between context and focus
        """
        return {
            "min_lines": 150,
            "max_lines": 400,
            "min_chars": 1500,
            "max_chars": 4000,
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
        return {"min_lines": 50, "max_lines": 150, "min_chars": 1500, "max_chars": 3000}


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
            "max_chars": 8000,
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
            "max_chars": 10000,
        }


class NomicEmbedTextStrategy(EmbeddingStrategy):
    """Strategy for nomic-ai/nomic-embed-text-v1.5 model.

    Strategy: Use task instruction prefixes as recommended by Nomic.
    - Code chunks: prefix with "search_document: " and add minimal context
    - Queries: prefix with "search_query: "

    Best for natural language queries about code functionality.
    """

    def prepare_code(self, chunk: CodeChunk, metadata: Dict[str, Any]) -> str:
        """Prepare code with search_document prefix and minimal context."""
        lines = []

        # Build a concise context line
        context_parts = []

        # Add language type if available
        language = metadata.get("language")
        if language:
            context_parts.append(f"{language}")

        # Add parent class if in a method
        parent_class = metadata.get("parent_class")
        if parent_class:
            context_parts.append(f"class {parent_class}")

        # Add class name if it's a class chunk
        class_name = metadata.get("class_name")
        if class_name and not parent_class:
            context_parts.append(f"class {class_name}")

        # Add function/method name
        function_name = metadata.get("function_name") or chunk.name
        if function_name:
            if parent_class:
                context_parts.append(f"method {function_name}")
            else:
                context_parts.append(f"function {function_name}")

        # Add context as a single comment line if we have any
        if context_parts:
            lines.append(f"# {' - '.join(context_parts)}")

        # Add docstring if available (limited to prevent dominating the embedding)
        docstring = metadata.get("docstring", "")
        if docstring:
            # Limit docstring to first 300 characters
            if len(docstring) > 300:
                docstring = docstring[:297] + "..."
            # Add as a comment block
            lines.append("# " + docstring.replace("\n", "\n# "))

        # Add empty line before code if we have any context
        if lines:
            lines.append("")

        # Add the actual code
        lines.append(chunk.content)

        # Extract and include significant inline comments (first few)
        code_lines = chunk.content.split("\n")
        significant_comments = []
        comment_prefixes = ["#", "//", "/*", "*", "--"]  # Common comment prefixes

        for line in code_lines[:20]:  # Only check first 20 lines for performance
            line = line.strip()
            for prefix in comment_prefixes:
                if line.startswith(prefix) and len(line) > len(prefix) + 5:
                    # Extract comment text
                    comment = line[len(prefix) :].strip()
                    # Remove common comment endings like */
                    comment = comment.rstrip("*/").strip()

                    # Skip trivial or auto-generated comments
                    if (
                        comment
                        and not comment.startswith("TODO")
                        and not comment.startswith("FIXME")
                        and not comment.startswith("NOTE")
                        and not comment.startswith("=")
                        and not comment.startswith("-")
                    ):
                        significant_comments.append(comment)
                        if len(significant_comments) >= 3:  # Limit to 3 comments
                            break
            if len(significant_comments) >= 3:
                break

        # Add significant comments at the end if found
        if significant_comments:
            lines.append("")
            lines.append("# Additional context from inline comments:")
            for comment in significant_comments:
                lines.append(f"# - {comment}")

        # Prefix with task instruction
        full_content = "\n".join(lines)
        return f"search_document: {full_content}"

    def prepare_query(self, query: str) -> str:
        """Add search_query prefix as recommended by Nomic."""
        return f"search_query: {query}"

    @property
    def ideal_chunk_size(self) -> Dict[str, int]:
        """Ideal: 100-300 lines or 2,000-8,000 characters.

        Supports 8192 token context but works best with moderate chunks.
        """
        return {
            "min_lines": 100,
            "max_lines": 300,
            "min_chars": 2000,
            "max_chars": 8000,
        }


class NomicCloudEmbedCodeStrategy(NomicEmbedCodeStrategy):
    """Strategy for Nomic Cloud API embedding generation.

    This strategy uses the same text preparation as NomicEmbedCodeStrategy
    but generates embeddings via the Nomic Cloud API instead of locally.

    Requires NOMIC_API_KEY environment variable to be set.
    """

    def __init__(self):
        """Initialize cloud strategy and validate API key."""
        super().__init__()
        self.api_key = os.environ.get("NOMIC_API_KEY")
        if not self.api_key:
            raise ValueError("NOMIC_API_KEY environment variable not set")

    def prepare_query(self, query: str) -> str:
        """Cloud API handles task types differently."""
        # For cloud API, we don't add the prefix - we'll use task_type parameter
        return query


# These models are officially supported
EMBEDDING_STRATEGIES = {
    # Nomic Embed Code - 7B parameters, best for code-to-code similarity
    "nomic-ai/nomic-embed-code": NomicEmbedCodeStrategy,
    # Nomic Embed Code GGUF - Quantized version for local use
    "nomic-ai/nomic-embed-code-gguf": NomicEmbedCodeStrategy,
    "nomic-embed-code-gguf": NomicEmbedCodeStrategy,
    # CodeBERT - Good for documentation-heavy projects
    "microsoft/codebert-base": CodeBERTStrategy,
    # Qodo Embed - Best for structured RAG systems
    "Qodo/Qodo-Embed-1-1.5B": QodoEmbedStrategy,
    # CodeRankEmbed - Optimized for NL→code with prefix
    "nomic-ai/CodeRankEmbed": CodeRankEmbedStrategy,
    # Nomic Embed Text v1.5 (default - best for natural language queries)
    "nomic-ai/nomic-embed-text-v1.5": NomicEmbedTextStrategy,
    # Nomic Cloud API - Cloud-based code embeddings
    "nomic-cloud/nomic-embed-code": NomicCloudEmbedCodeStrategy,
}

# List of supported models for error messages
SUPPORTED_MODELS = [
    "nomic-ai/nomic-embed-code",
    "nomic-ai/nomic-embed-code-gguf",
    "nomic-embed-code-gguf",
    "microsoft/codebert-base",
    "Qodo/Qodo-Embed-1-1.5B",
    "nomic-ai/CodeRankEmbed",
    "nomic-ai/nomic-embed-text-v1.5",
    "nomic-cloud/nomic-embed-code",
]


def get_embedding_strategy(model_name: str) -> EmbeddingStrategy:
    """Get the appropriate embedding strategy for a model.

    Supported models:
    1. nomic-ai/nomic-embed-code - Raw code only (7B params)
    2. microsoft/codebert-base - Rich documentation
    3. Qodo/Qodo-Embed-1-1.5B - Structured metadata
    4. nomic-ai/CodeRankEmbed - Code-specific with query prefix
    5. nomic-ai/nomic-embed-text-v1.5 - General text model (default)
    6. nomic-cloud/nomic-embed-code - Cloud API for code embeddings

    Args:
        model_name: The name of the embedding model.

    Returns:
        An instance of the appropriate strategy.

    Raises:
        ValueError: If the model is not one of the supported models.
    """
    # Normalize model name
    normalized_name = model_name.strip()

    # Check if it's a supported model
    if normalized_name in EMBEDDING_STRATEGIES:
        return EMBEDDING_STRATEGIES[normalized_name]()

    # Model not supported - provide helpful error
    raise ValueError(
        f"Unsupported embedding model: '{model_name}'.\n"
        f"Only these models are supported:\n"
        f"  1. nomic-ai/nomic-embed-code - Raw code only (7B params)\n"
        f"  2. nomic-ai/nomic-embed-code-gguf - GGUF quantized version\n"
        f"  3. microsoft/codebert-base - Rich documentation\n"
        f"  4. Qodo/Qodo-Embed-1-1.5B - Structured metadata\n"
        f"  5. nomic-ai/CodeRankEmbed - Code-specific with query prefix\n"
        f"  6. nomic-ai/nomic-embed-text-v1.5 - General text model (default)\n"
        f"  7. nomic-cloud/nomic-embed-code - Cloud API for code embeddings\n"
    )
