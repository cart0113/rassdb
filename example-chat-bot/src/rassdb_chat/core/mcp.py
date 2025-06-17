"""RASSDB Model Context Protocol (MCP) implementation."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
import tomllib
import sys
from sentence_transformers import SentenceTransformer
import numpy as np

# Add parent directory to path to import rassdb
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from rassdb.vector_store import VectorStore
from rassdb.embedding_strategies import get_embedding_strategy
from rassdb.cloud_embeddings import get_cloud_embedding_model

logger = logging.getLogger(__name__)


class RASSDBMCPHandler:
    """Handles RASSDB operations through the Model Context Protocol."""

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
        self._vector_store: Optional[VectorStore] = None
        self._model: Optional[SentenceTransformer] = None
        self._embedding_strategy = None

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

    def _get_model_name(self) -> str:
        """Get the model name from config or use default."""
        # Default model name
        model_name = "nomic-ai/nomic-embed-text-v1.5"

        # Try to read config file
        config_path = Path(".rassdb-config.toml")
        if config_path.exists():
            try:
                with open(config_path, "rb") as f:
                    config = tomllib.load(f)
                # Check if embedding model is specified
                if "embedding-model" in config and "name" in config["embedding-model"]:
                    model_name = config["embedding-model"]["name"]
            except Exception as e:
                logger.warning(f"Failed to read config file: {e}, using default model")

        return model_name

    async def initialize(self) -> None:
        """Initialize the RASSDB connection and load indices."""
        try:
            logger.info(f"[MCP] Starting initialization...")
            logger.info(f"[MCP] Database path: {self.db_path}")
            logger.info(f"[MCP] Absolute path: {self.db_path.absolute()}")
            logger.info(f"[MCP] Database exists: {self.db_path.exists()}")
            
            # Initialize VectorStore
            logger.info(f"[MCP] Initializing VectorStore...")
            self._vector_store = VectorStore(str(self.db_path))
            logger.info(f"[MCP] VectorStore initialized successfully")
            
            # Get model name from config or use default
            model_name = self._get_model_name()
            logger.info(f"[MCP] Loading embedding model: {model_name}")
            
            # Load the embedding model (this is usually the slow part)
            import time
            start_time = time.time()
            
            # Check if it's a cloud model first
            cloud_model = get_cloud_embedding_model(model_name)
            if cloud_model:
                self._model = cloud_model
                logger.info(f"[MCP] Using cloud embedding model")
            else:
                self._model = SentenceTransformer(model_name, trust_remote_code=True)
                logger.info(f"[MCP] Using local embedding model")
                
            load_time = time.time() - start_time
            logger.info(f"[MCP] Model loaded in {load_time:.2f} seconds")
            
            # Get embedding strategy
            self._embedding_strategy = get_embedding_strategy(model_name)
            logger.info(f"[MCP] Got embedding strategy: {type(self._embedding_strategy).__name__}")
            
            # Load index configuration
            if self.index_path.exists():
                with open(self.index_path, "r") as f:
                    self.index_config = f.read()
            else:
                self.index_config = ""

            self.is_initialized = True
            logger.info("[MCP] RASSDB MCP handler initialized successfully")
        except Exception as e:
            logger.error(f"[MCP] Failed to initialize RASSDB: {e}")
            import traceback
            logger.error(f"[MCP] Traceback: {traceback.format_exc()}")
            raise

    async def query(
        self, query_text: str, top_k: int = 5, filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Execute a query against RASSDB.

        Args:
            query_text: The query string
            top_k: Number of results to return
            filters: Optional query filters

        Returns:
            List of matching documents with scores
        """
        if not self.is_initialized:
            await self.initialize()

        logger.info(f"[MCP] Executing query: {query_text[:50]}...")
        logger.info(f"[MCP] Query parameters: top_k={top_k}, filters={filters}")

        try:
            # Prepare query text using strategy
            if self._embedding_strategy:
                prepared_query = self._embedding_strategy.prepare_query(query_text)
                logger.info(f"[MCP] Prepared query: {prepared_query[:100]}...")
            else:
                prepared_query = query_text

            # Create embedding for the query
            logger.info(f"[MCP] Creating query embedding...")
            import time
            start_time = time.time()
            query_embedding = await asyncio.to_thread(
                self._model.encode, prepared_query, normalize_embeddings=True
            )
            embed_time = time.time() - start_time
            logger.info(f"[MCP] Query embedding created in {embed_time:.2f} seconds, shape: {query_embedding.shape}")

            # Search the vector store
            logger.info(f"[MCP] Searching vector store...")
            search_start = time.time()
            search_results = await asyncio.to_thread(
                self._vector_store.search_similar, query_embedding, limit=top_k
            )
            search_time = time.time() - search_start
            logger.info(f"[MCP] Search completed in {search_time:.2f} seconds, found {len(search_results)} results")

            # Format results
            results = []
            for idx, result in enumerate(search_results):
                # Extract metadata if it exists
                metadata = result.get("metadata", {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}

                results.append(
                    {
                        "id": result.get("id", f"chunk_{idx}"),
                        "content": result.get("content", ""),
                        "score": 1.0
                        - float(
                            result.get("distance", 0.5)
                        ),  # Convert distance to similarity score
                        "metadata": {
                            "file_path": result.get("file_path", ""),
                            "line_number": result.get("start_line", 0),
                            "type": result.get("chunk_type", "code"),
                            "language": result.get("language", "unknown"),
                        },
                    }
                )

            logger.info(f"[MCP] Query completed successfully, returning {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"[MCP] Query failed: {e}")
            import traceback
            logger.error(f"[MCP] Traceback: {traceback.format_exc()}")
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

        # Mock indexing process
        stats = {
            "files_processed": 42,
            "files_ignored": 8,
            "total_size_bytes": 1024 * 1024 * 5,  # 5MB
            "index_time_seconds": 2.3,
            "embedding_model": "text-embedding-ada-002",
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
        logger.info(f"Finding similar {language} code")

        # Mock similar code results
        results = []
        for i in range(min(top_k, 2)):
            results.append(
                {
                    "id": f"similar_{i}",
                    "code": f"def similar_function_{i}():\n    pass",
                    "similarity": 0.92 - (i * 0.05),
                    "file_path": f"src/similar_{i}.py",
                    "language": language,
                }
            )

        return results
