"""Search command for RASSDB - unified semantic and literal code search.

This module provides the CLI interface for searching code using both
semantic (embedding-based) and literal (grep-like) search methods.
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

logger = logging.getLogger(__name__)


class SearchEngine:
    """Handles both semantic and literal search operations."""

    def __init__(self, db_path: str, model_name: str = "nomic-ai/CodeRankEmbed"):
        """Initialize search engine.

        Args:
            db_path: Path to the database.
            model_name: Name of the embedding model.
        """
        self.db_path = db_path
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
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
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            # Load from standard HuggingFace cache location
            self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
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

    def literal_search(
        self,
        pattern: str,
        case_sensitive: bool = False,
        regex: bool = False,
        whole_word: bool = False,
        language: Optional[str] = None,
        file_pattern: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Perform literal text search (grep-like).

        Args:
            pattern: Search pattern.
            case_sensitive: Whether to use case-sensitive matching.
            regex: Whether pattern is a regex.
            whole_word: Whether to match whole words only.
            language: Filter by programming language.
            file_pattern: Filter by file path pattern.
            limit: Maximum number of results.

        Returns:
            List of search results with match information.
        """
        vector_store = VectorStore(self.db_path)

        # Build the search pattern
        if regex:
            search_pattern = pattern
        else:
            search_pattern = re.escape(pattern)

        if whole_word:
            search_pattern = r"\b" + search_pattern + r"\b"

        # Compile regex
        flags = 0 if case_sensitive else re.IGNORECASE
        regex_pattern = re.compile(search_pattern, flags)

        # Search in database
        chunks = vector_store.search_literal(
            "",  # We'll filter with regex in Python
            limit=limit * 10,  # Get more to filter
            language=language,
            file_pattern=file_pattern,
        )

        results = []
        for chunk in chunks:
            # Search for pattern in content
            matches = list(regex_pattern.finditer(chunk["content"]))
            if matches:
                # Calculate match score based on frequency
                match_score = min(1.0, len(matches) / 10.0)  # Cap at 10 matches

                # Find matching lines
                lines = chunk["content"].split("\n")
                matching_lines = []

                for match in matches:
                    pos = 0
                    for i, line in enumerate(lines):
                        if pos <= match.start() < pos + len(line) + 1:
                            line_num = chunk["start_line"] + i
                            matching_lines.append(
                                {
                                    "line_num": line_num,
                                    "line": line.strip(),
                                    "match": match.group(),
                                }
                            )
                            break
                        pos += len(line) + 1

                result = chunk.copy()
                result.update(
                    {
                        "distance": 1.0 - match_score,
                        "similarity": match_score,
                        "search_type": "literal",
                        "matches": matching_lines,
                        "match_count": len(matches),
                    }
                )
                results.append(result)

        vector_store.close()

        # Sort by match score and limit
        results.sort(key=lambda x: x["distance"])
        return results[:limit]


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
            output.append(f"Type: {result['chunk_type']}")
            output.append(f"Language: {result['language']}")

            # Relevance info
            if search_type == "semantic":
                output.append(f"Relevance Score: {result.get('similarity', 0):.3f}")
            else:  # literal
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
            if search_type == "semantic":
                table_data.append(
                    [
                        i,
                        result["file_path"],
                        f"{result['start_line']}-{result['end_line']}",
                        result["chunk_type"],
                        result["language"],
                        f"{result.get('similarity', 0):.3f}",
                    ]
                )
            else:  # literal
                table_data.append(
                    [
                        i,
                        result["file_path"],
                        f"{result['start_line']}-{result['end_line']}",
                        result["chunk_type"],
                        result["language"],
                        result.get("match_count", 0),
                    ]
                )

        headers = [
            "#",
            "File",
            "Lines",
            "Type",
            "Language",
            "Score" if search_type == "semantic" else "Matches",
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
            if search_type == "literal" and result.get("matches"):
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
@click.option("--literal", "-l", is_flag=True, help="Use literal search (grep-like)")
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
@click.option("--file", help="Filter by file pattern")
@click.option(
    "-i", "--ignore-case", is_flag=True, help="Case insensitive (literal search)"
)
@click.option("-E", "--regex", is_flag=True, help="Use regex (literal search)")
@click.option("-w", "--word", is_flag=True, help="Match whole words (literal search)")
@click.option("--show", is_flag=True, help="Alias for --format show")
def main(
    query: str,
    semantic: bool,
    literal: bool,
    db: str,
    limit: int,
    format: str,
    language: Optional[str],
    file: Optional[str],
    ignore_case: bool,
    regex: bool,
    word: bool,
    show: bool,
) -> None:
    """Unified code search tool supporting semantic and literal search.

    You must specify at least one search type: --semantic (-s) or --literal (-l).
    Both can be used together for comprehensive results.

    Examples:

        # Semantic search for similar concepts
        rassdb-search -s "error handling"

        # Literal search for exact text
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

    # Validate that at least one search type is specified
    if not semantic and not literal:
        click.echo(
            "Error: You must specify at least one search type: --semantic (-s) or --literal (-l)",
            err=True,
        )
        click.echo("Use --help for more information.", err=True)
        sys.exit(1)

    try:
        # Discover database if not specified
        db_path = discover_database(db)

        engine = SearchEngine(db_path)
        formatter = ResultFormatter()

        all_results = []

        # Perform semantic search if requested
        if semantic:
            semantic_results = engine.semantic_search(query, limit, language, file)
            for r in semantic_results:
                r["search_method"] = "semantic"
            all_results.extend(semantic_results)

        # Perform literal search if requested
        if literal:
            literal_results = engine.literal_search(
                query,
                case_sensitive=not ignore_case,
                regex=regex,
                whole_word=word,
                language=language,
                file_pattern=file,
                limit=limit,
            )
            for r in literal_results:
                r["search_method"] = "literal"
            all_results.extend(literal_results)

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
