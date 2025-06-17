"""RASSDB Model Context Protocol (MCP) implementation using CLI."""

import asyncio
import json
import logging
import subprocess
from typing import Any, Dict, List, Optional
from pathlib import Path
import tomllib

logger = logging.getLogger(__name__)


class RASSDBMCPHandler:
    """Handles RASSDB operations through the CLI."""

    def __init__(
        self, db_path: Optional[Path] = None, index_path: Optional[Path] = None
    ):
        """Initialize the MCP handler.

        Args:
            db_path: Path to the RASSDB database file
            index_path: Path to the index configuration
        """
        if db_path:
            self.db_path = db_path
        else:
            # Dynamically construct database path based on config
            self.db_path = self._get_database_path()

        self.index_path = index_path or Path(".rassdb-ignore")
        self.is_initialized = False
        self._context_cache: Dict[str, Any] = {}

    def _get_database_path(self) -> Path:
        """Construct database path based on config file."""
        # Default model name
        model_name = "nomic-embed-text-v1.5"

        # Try to read config file
        config_path = Path(".rassdb-config.toml")
        if config_path.exists():
            try:
                with open(config_path, "rb") as f:
                    config = tomllib.load(f)
                # Check if embedding model is specified
                if "embedding-model" in config and "name" in config["embedding-model"]:
                    # Extract just the model name from the full path
                    full_model_name = config["embedding-model"]["name"]
                    # Convert model name to filesystem-friendly format
                    model_name = full_model_name.replace("/", "-").lower()
            except Exception as e:
                logger.warning(f"Failed to read config file: {e}, using default model")

        # Construct database path
        # Get the parent directory name (e.g., "example-chat-bot")
        parent_dir = Path.cwd().name
        db_name = f"{parent_dir}-{model_name}.rassdb"
        return Path(".rassdb") / db_name

    async def initialize(self) -> None:
        """Initialize the RASSDB connection and check database exists."""
        try:
            logger.info(f"[MCP-CLI] Starting initialization...")
            logger.info(f"[MCP-CLI] Database path: {self.db_path}")
            logger.info(f"[MCP-CLI] Absolute path: {self.db_path.absolute()}")
            logger.info(f"[MCP-CLI] Database exists: {self.db_path.exists()}")

            if not self.db_path.exists():
                raise FileNotFoundError(f"Database not found: {self.db_path}")

            # Load index configuration
            if self.index_path.exists():
                with open(self.index_path, "r") as f:
                    self.index_config = f.read()
            else:
                self.index_config = ""

            self.is_initialized = True
            logger.info("[MCP-CLI] RASSDB MCP handler initialized successfully")
        except Exception as e:
            logger.error(f"[MCP-CLI] Failed to initialize RASSDB: {e}")
            raise

    async def query(
        self, query_text: str, top_k: int = 5, filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Execute a query against RASSDB using CLI.

        Args:
            query_text: The query string
            top_k: Number of results to return
            filters: Optional query filters

        Returns:
            List of matching documents with scores
        """
        if not self.is_initialized:
            await self.initialize()

        logger.info(f"[MCP-CLI] Executing query: {query_text[:50]}...")
        logger.info(f"[MCP-CLI] Query parameters: top_k={top_k}")

        try:
            # Construct the CLI command using rassdb-search
            cmd = [
                "./../bin/rassdb-search",
                "-s",  # Use semantic search
                "--format",
                "json",
                "--limit",
                str(top_k),
                query_text,
            ]

            logger.info(f"[MCP-CLI] Running command: {' '.join(cmd)}")

            # Run the command
            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                error_msg = stderr.decode("utf-8")
                logger.error(f"[MCP-CLI] Command failed: {error_msg}")
                raise RuntimeError(f"RASSDB search failed: {error_msg}")

            # Parse the JSON output
            output = stdout.decode("utf-8")
            logger.info(f"[MCP-CLI] Got output of length: {len(output)}")

            # The CLI returns JSON array of results
            search_results = json.loads(output)
            logger.info(f"[MCP-CLI] Parsed {len(search_results)} results")

            # Format results to match expected structure
            results = []
            for idx, result in enumerate(search_results):
                # Convert CLI result format to our expected format
                # The CLI returns similarity (not distance) and has different field names
                results.append(
                    {
                        "id": result.get("id", f"chunk_{idx}"),
                        "content": result.get("content", ""),
                        "score": float(
                            result.get("similarity", result.get("score", 0.0))
                        ),
                        "metadata": {
                            "file_path": result.get("file_path", ""),
                            "line_number": result.get("start_line", 0),
                            "type": result.get("chunk_type", "code"),
                            "language": result.get("language", "unknown"),
                        },
                    }
                )

            logger.info(
                f"[MCP-CLI] Query completed successfully, returning {len(results)} results"
            )
            return results

        except json.JSONDecodeError as e:
            logger.error(f"[MCP-CLI] Failed to parse JSON output: {e}")
            logger.error(f"[MCP-CLI] Raw output: {stdout.decode('utf-8')[:500]}")
            raise RuntimeError(f"Failed to parse RASSDB output: {e}")
        except Exception as e:
            logger.error(f"[MCP-CLI] Query failed: {e}")
            raise RuntimeError(f"RASSDB query failed: {str(e)}") from e

    async def add_context(self, session_id: str, context: Dict[str, Any]) -> None:
        """Add context to the MCP session.

        Args:
            session_id: Unique session identifier
            context: Context data to add
        """
        self._context_cache[session_id] = context
        logger.info(f"Added context for session {session_id}")

    async def get_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve context for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            Context data if available
        """
        return self._context_cache.get(session_id)

    async def clear_context(self, session_id: str) -> None:
        """Clear context for a session.

        Args:
            session_id: Unique session identifier
        """
        if session_id in self._context_cache:
            del self._context_cache[session_id]
            logger.info(f"Cleared context for session {session_id}")

    async def index_directory(
        self, directory: Path, ignore_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Index a directory for RASSDB.

        Args:
            directory: Directory to index
            ignore_patterns: Patterns to ignore during indexing

        Returns:
            Indexing statistics
        """
        logger.info(f"Indexing directory: {directory}")

        # Mock indexing process for now
        stats = {
            "files_processed": 42,
            "files_ignored": 8,
            "total_size_bytes": 1024 * 1024 * 5,  # 5MB
            "index_time_seconds": 2.3,
            "embedding_model": "nomic-ai/nomic-embed-text-v1.5",
        }

        return stats

    async def get_similar_code(
        self, code_snippet: str, language: str = "python", top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar code snippets in the database.

        Args:
            code_snippet: Code to find similarities for
            language: Programming language
            top_k: Number of results to return

        Returns:
            List of similar code snippets
        """
        # For now, just use the regular query
        return await self.query(code_snippet, top_k=top_k)
