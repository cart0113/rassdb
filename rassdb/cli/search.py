"""Search command for RASSDB - unified semantic and lexical code search.

This module provides the CLI interface for searching code using both
semantic (embedding-based) and lexical (grep-like) search methods.
"""

import sys
import json
import logging
import re
import tomllib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import click
from tabulate import tabulate
from sentence_transformers import SentenceTransformer
import numpy as np

from rassdb.vector_store import VectorStore
from rassdb.utils.db_discovery import discover_database
from rassdb.embedding_strategies import get_embedding_strategy
from rassdb.cloud_embeddings import get_cloud_embedding_model
from rassdb.gguf_embeddings import get_gguf_embedding_model

logger = logging.getLogger(__name__)


class SearchEngine:
    """Handles both semantic and lexical search operations."""

    def __init__(
        self,
        db_path: str,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        lazy_load: bool = True,
    ):
        """Initialize search engine.

        Args:
            db_path: Path to the database.
            model_name: Name of the embedding model.
            lazy_load: Whether to lazy load the embedding model.
        """
        self.db_path = db_path
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self.lazy_load = lazy_load
        self._load_global_config()

    def _load_global_config(self):
        """Load configuration from project and global .rassdb-config.toml files."""
        # First check project config
        db_dir = Path(self.db_path).parent.parent
        project_config_path = db_dir / ".rassdb-config.toml"

        config = None
        if project_config_path.exists():
            try:
                with open(project_config_path, "rb") as f:
                    config = tomllib.load(f)
            except Exception:
                pass

        # Fall back to global config if no project config
        if config is None:
            global_config_path = Path.home() / ".rassdb-config.toml"
            if global_config_path.exists():
                try:
                    with open(global_config_path, "rb") as f:
                        config = tomllib.load(f)
                except Exception:
                    pass

        # Use embedding model from config if available
        if (
            config
            and "embedding-model" in config
            and "name" in config["embedding-model"]
        ):
            self.model_name = config["embedding-model"]["name"]

    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None and self.lazy_load:
            logger.info(f"Loading embedding model: {self.model_name}")
            # Check if it's a cloud model first
            cloud_model = get_cloud_embedding_model(self.model_name)
            if cloud_model:
                self._model = cloud_model
            else:
                # Check if it's a GGUF model
                gguf_model = get_gguf_embedding_model(self.model_name)
                if gguf_model:
                    self._model = gguf_model
                else:
                    # Load from standard HuggingFace cache location
                    self._model = SentenceTransformer(
                        self.model_name, trust_remote_code=True
                    )
        return self._model

    def semantic_search(
        self,
        query: str,
        limit: int = 10,
        language: Optional[str] = None,
        file_pattern: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings.

        Args:
            query: Search query.
            limit: Maximum number of results.
            language: Filter by programming language.
            file_pattern: Filter by file path pattern.

        Returns:
            List of search results with similarity scores.
        """
        # Apply default pattern if none specified
        if file_pattern is None:
            file_pattern = r".*\.(py|pyx|pyi|js|jsx|ts|tsx|c|h|cpp|hpp|cc|cxx|go|rs|java|kt|swift|rb|php|cs|vb|r|m|scala|clj|ex|exs|erl|hrl|lua|pl|sh|bash|zsh|fish|ps1|yaml|yml|toml|ini|cfg|conf|env|dockerfile|makefile|cmake|gradle|rs|swift|kt|scala|clj|ex|exs|erl|hrl|lua|pl|proto|thrift|graphql|sql|css|scss|sass|less|html|htm|xml|rst|md)$"

        vector_store = VectorStore(self.db_path)

        # Use embedding strategy to prepare query
        try:
            strategy = get_embedding_strategy(self.model_name)
            query_text = strategy.prepare_query(query)
        except ValueError:
            # Fallback if model not supported
            query_text = query

        # Generate query embedding
        query_embedding = self.model.encode(query_text, normalize_embeddings=True)

        # Ensure embedding is 1D array (some models return 2D array for single input)
        if query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
            query_embedding = query_embedding[0]

        # Search for similar chunks
        results = vector_store.search_similar(
            query_embedding,
            limit=limit,
            language=language,
            file_pattern=file_pattern,
        )

        vector_store.close()

        # Add search type marker and calculate similarity
        for r in results:
            r["search_type"] = "semantic"
            # For cosine distance, similarity = 1 - distance
            # Cosine distance ranges from 0 to 2, where 0 means identical
            r["similarity"] = 1.0 - r.get("distance", 0)

        return results

    def lexical_search(
        self,
        pattern: str,
        case_sensitive: bool = False,
        regex: bool = False,
        whole_word: bool = False,
        language: Optional[str] = None,
        file_pattern: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Perform lexical text search using FTS5.

        Args:
            pattern: Search pattern. Supports FTS5 query syntax.
            case_sensitive: Whether to use case-sensitive matching (Note: FTS5 is case-insensitive).
            regex: Whether pattern is a regex (Note: FTS5 uses its own syntax).
            whole_word: Whether to match whole words only.
            language: Filter by programming language.
            file_pattern: Filter by file path pattern.
            limit: Maximum number of results.

        Returns:
            List of search results with match information.
        """
        # Apply default pattern if none specified
        if file_pattern is None:
            file_pattern = r".*\.(py|pyx|pyi|js|jsx|ts|tsx|c|h|cpp|hpp|cc|cxx|go|rs|java|kt|swift|rb|php|cs|vb|r|m|scala|clj|ex|exs|erl|hrl|lua|pl|sh|bash|zsh|fish|ps1|yaml|yml|toml|ini|cfg|conf|env|dockerfile|makefile|cmake|gradle|rs|swift|kt|scala|clj|ex|exs|erl|hrl|lua|pl|proto|thrift|graphql|sql|css|scss|sass|less|html|htm|xml|rst|md)$"

        vector_store = VectorStore(self.db_path)

        # Convert pattern to FTS5 query syntax
        fts_query: str = pattern

        # Handle whole word matching
        if whole_word:
            # In FTS5, we can use quotes for exact word matching
            words = pattern.split()
            fts_query = " ".join(f'"{word}"' for word in words)
        elif not regex:
            # For non-regex patterns, decide based on query complexity
            # If it looks like a natural language query, use token-based search
            # If it's a simple term or code snippet, use phrase search
            if len(pattern.split()) > 3 or any(
                word in pattern.lower()
                for word in ["how", "what", "where", "when", "why", "does"]
            ):
                # Natural language query - search for all tokens
                # Remove common stop words and punctuation
                import string

                # Remove "query:" prefix if present
                clean_pattern = pattern
                if clean_pattern.lower().startswith("query:"):
                    clean_pattern = clean_pattern[6:].strip()

                words = clean_pattern.translate(
                    str.maketrans("", "", string.punctuation)
                ).split()
                # Filter out very common words that don't add search value
                # Common English stop words that don't add search value
                stop_words = {
                    "the",
                    "a",
                    "an",
                    "and",
                    "or",
                    "but",
                    "in",
                    "on",
                    "at",
                    "to",
                    "for",
                    "of",
                    "with",
                    "by",
                    "from",
                    "is",
                    "are",
                    "was",
                    "were",
                    "been",
                    "be",
                    "have",
                    "has",
                    "had",
                    "do",
                    "does",
                    "did",
                    "will",
                    "would",
                    "could",
                    "should",
                    "may",
                    "might",
                    "must",
                    "can",
                    "this",
                    "that",
                    "these",
                    "those",
                    "i",
                    "you",
                    "he",
                    "she",
                    "it",
                    "we",
                    "they",
                    "them",
                    "their",
                    "what",
                    "which",
                    "who",
                    "when",
                    "where",
                    "why",
                    "how",
                    "all",
                    "each",
                    "every",
                    "some",
                    "any",
                    "few",
                    "more",
                    "most",
                    "other",
                    "into",
                    "through",
                    "during",
                    "before",
                    "after",
                    "above",
                    "below",
                    "between",
                    "under",
                    "again",
                    "further",
                    "then",
                    "once",
                    "out",
                    "call",
                }
                meaningful_words = [
                    w for w in words if w.lower() not in stop_words and len(w) > 1
                ]
                if meaningful_words:
                    # Use OR operator between terms for more flexible matching
                    # This finds documents containing ANY of the meaningful terms
                    fts_query = " OR ".join(meaningful_words)
                else:
                    # Fallback to original query if no meaningful words
                    fts_query = pattern
            else:
                # Short query or code snippet - use phrase search
                fts_query = '"' + pattern.replace('"', '""') + '"'

        # Search using FTS5
        results = vector_store.search_lexical(
            fts_query,
            limit=limit,
            language=language,
            file_pattern=file_pattern,
        )

        # Process results to extract match information
        for result in results:
            # Extract matches from snippet if available
            snippet = result.get("snippet", "")
            matches = []

            if snippet:
                # Find <match> tags in snippet
                import re

                match_pattern = re.compile(r"<match>(.*?)</match>")
                snippet_matches = match_pattern.findall(snippet)

                # Create match info
                for match_text in snippet_matches[:5]:  # Limit to first 5 matches
                    matches.append(
                        {
                            "match": match_text,
                            "line": snippet,  # Full snippet as context
                        }
                    )

            result["search_type"] = "lexical"
            result["matches"] = matches
            result["match_count"] = (
                len(matches) if matches else 1
            )  # At least 1 match if result returned

            # Remove internal fields
            result.pop("snippet", None)

        vector_store.close()

        return results


class ResultFormatter:
    """Formats search results for display."""

    @staticmethod
    def format_show(
        results: List[Dict[str, Any]], search_type: str = "semantic"
    ) -> str:
        """Format results with full code display.

        Args:
            results: List of search results.
            search_type: Type of search performed.

        Returns:
            Formatted string with full results.
        """
        if not results:
            return "No matches found."

        output = []
        for i, result in enumerate(results, 1):
            output.append(f"\n{'=' * 80}")
            output.append(f"Match #{i}")
            output.append(f"{'=' * 80}")

            # File and location info
            output.append(f"File: {result['file_path']}")
            output.append(f"Lines: {result['start_line']}-{result['end_line']}")
            
            # Show part lines if different from full lines
            if result.get('part_start_line') and result.get('part_end_line'):
                if (result['part_start_line'] != result['start_line'] or 
                    result['part_end_line'] != result['end_line']):
                    output.append(f"Part Lines: {result['part_start_line']}-{result['part_end_line']}")
            
            output.append(f"Type: {result['chunk_type']}")
            output.append(f"Language: {result['language']}")

            # Relevance info
            if search_type == "semantic":
                output.append(f"Relevance Score: {result.get('similarity', 0):.3f}")
            else:  # lexical
                output.append(f"Match Count: {result.get('match_count', 0)}")
                if result.get("matches"):
                    output.append(
                        f"Match Lines: {', '.join(str(m['line_num']) for m in result['matches'][:5])}"
                    )
                    if len(result["matches"]) > 5:
                        output.append(
                            f"             ... and {len(result['matches']) - 5} more"
                        )

            # Metadata if available
            if result.get("metadata"):
                output.append(f"Metadata: {json.dumps(result['metadata'], indent=2)}")

            # Full code content
            output.append(f"\nCode Content:")
            output.append("-" * 80)
            output.append(result["content"])
            output.append("-" * 80)

        return "\n".join(output)

    @staticmethod
    def format_table(
        results: List[Dict[str, Any]], search_type: str = "semantic"
    ) -> str:
        """Format results as a table.

        Args:
            results: List of search results.
            search_type: Type of search performed.

        Returns:
            Formatted table string.
        """
        if not results:
            return "No matches found."

        table_data = []
        for i, result in enumerate(results, 1):
            # Format the lines column
            lines_str = f"{result['start_line']}-{result['end_line']}"
            
            # If this is a part and has different part lines, show both
            if result.get('part_start_line') and result.get('part_end_line'):
                if (result['part_start_line'] != result['start_line'] or 
                    result['part_end_line'] != result['end_line']):
                    # Show both original and part lines
                    lines_str = f"{result['start_line']}-{result['end_line']}"
                    part_lines_str = f"{result['part_start_line']}-{result['part_end_line']}"
                else:
                    part_lines_str = ""
            else:
                part_lines_str = ""
            
            table_data.append(
                [
                    i,
                    result["file_path"],
                    lines_str,
                    part_lines_str,
                    result["chunk_type"],
                    result["language"],
                    f"{result.get('similarity', 0):.3f}",
                ]
            )

        headers = [
            "#",
            "File",
            "Full Lines",
            "Part Lines",
            "Type",
            "Language",
            "Score",
        ]
        return tabulate(table_data, headers=headers, tablefmt="grid")

    @staticmethod
    def format_json(results: List[Dict[str, Any]]) -> str:
        """Format results as JSON.

        Args:
            results: List of search results.

        Returns:
            JSON string.
        """
        return json.dumps(results, indent=2)

    @staticmethod
    def format_simple(
        results: List[Dict[str, Any]], search_type: str = "semantic"
    ) -> str:
        """Format results as simple file:line references.

        Args:
            results: List of search results.
            search_type: Type of search performed.

        Returns:
            Simple formatted string.
        """
        output = []
        for result in results:
            if search_type == "lexical" and result.get("matches"):
                # Show individual match lines
                for match in result["matches"]:
                    output.append(
                        f"{result['file_path']}:{match['line_num']}: {match['line']}"
                    )
            else:
                # Show chunk locations
                output.append(
                    f"{result['file_path']}:{result['start_line']}-{result['end_line']}"
                )
        return "\n".join(output)


@click.command(name="rassdb-search")
@click.argument("query")
@click.option("--semantic", "-s", is_flag=True, help="Use semantic search (embeddings)")
@click.option("--lexical", "-l", is_flag=True, help="Use lexical search (grep-like)")
@click.option(
    "--db",
    default=None,
    help="Database file path. If not specified, auto-discovers from .rassdb directory",
)
@click.option("--limit", "-n", default=10, help="Number of results")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["table", "json", "simple", "show"]),
    default="table",
    help="Output format",
)
@click.option("--language", help="Filter by programming language")
@click.option(
    "--file-pattern",
    default=None,
    help="Filter by file pattern (regex). Default: common code/doc extensions. Example: '.*\\.js$' for only JS files",
)
@click.option(
    "-i", "--ignore-case", is_flag=True, help="Case insensitive (lexical search)"
)
@click.option("-E", "--regex", is_flag=True, help="Use regex (lexical search)")
@click.option("-w", "--word", is_flag=True, help="Match whole words (lexical search)")
@click.option("--show", is_flag=True, help="Alias for --format show")
def main(
    query: str,
    semantic: bool,
    lexical: bool,
    db: str,
    limit: int,
    format: str,
    language: Optional[str],
    file_pattern: Optional[str],
    ignore_case: bool,
    regex: bool,
    word: bool,
    show: bool,
) -> None:
    """Unified code search tool supporting semantic and lexical search.

    By default, both semantic and lexical search are performed.
    You can specify search types with --semantic (-s) or --lexical (-l) to use only one.
    Both can be used together explicitly for comprehensive results.

    Examples:

        # Semantic search for similar concepts
        rassdb-search -s "error handling"

        # Lexical search for exact text
        rassdb-search -l ".tick"

        # Both searches combined
        rassdb-search -s -l "database connection"

        # Show full code content
        rassdb-search -s "parse function" --show

        # Filter by language
        rassdb-search -s "async function" --language javascript

        # Regex search
        rassdb-search -l "class.*Component" -E
    """
    # Handle --show flag
    if show:
        format = "show"

    # Default to both search types if none specified
    if not semantic and not lexical:
        semantic = True
        lexical = True

    try:
        # Discover database if not specified
        db_path = discover_database(db)

        # Only load models if semantic search is requested
        engine = SearchEngine(db_path, lazy_load=True)
        formatter = ResultFormatter()

        all_results = []

        # If both search types are used, limit each to 5 results
        search_limit = 5 if (semantic and lexical) else limit

        # Perform semantic search if requested
        if semantic:
            semantic_results = engine.semantic_search(
                query, search_limit, language, file_pattern
            )
            for r in semantic_results:
                r["search_method"] = "semantic"
            all_results.extend(semantic_results)

        # Perform lexical search if requested
        if lexical:
            lexical_results = engine.lexical_search(
                query,
                case_sensitive=not ignore_case,
                regex=regex,
                whole_word=word,
                language=language,
                file_pattern=file_pattern,
                limit=search_limit,
            )
            for r in lexical_results:
                r["search_method"] = "lexical"
            all_results.extend(lexical_results)

        # Sort results by similarity score (highest first)
        all_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)

        # Format output
        if format == "show":
            output = formatter.format_show(all_results)
        elif format == "json":
            output = formatter.format_json(all_results)
        elif format == "simple":
            output = formatter.format_simple(all_results)
        else:  # table
            output = formatter.format_table(all_results)

        print(output)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
