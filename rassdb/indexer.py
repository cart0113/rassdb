"""Codebase indexing functionality for building RAG databases.

This module provides the main indexing functionality to process codebases
and store their embeddings in the vector database.
"""

import logging
from pathlib import Path
from typing import Set, Optional, Callable, Any, Dict, List
from tqdm import tqdm

# from rassdb.ui import IndexingProgress
from sentence_transformers import SentenceTransformer
import numpy as np
import numpy.typing as npt
import tomllib
import fnmatch
import re

from rassdb.vector_store import VectorStore
from rassdb.code_parser import CodeParser, CodeChunk
from rassdb.embedding_strategies import get_embedding_strategy, EmbeddingStrategy
from rassdb.cloud_embeddings import get_cloud_embedding_model
from rassdb.gguf_embeddings import get_gguf_embedding_model

logger = logging.getLogger(__name__)


class CodebaseIndexer:
    """Indexes codebases into a vector database for RAG/semantic search.

    This class handles the process of scanning directories, parsing code files,
    generating embeddings, and storing them in the vector database.

    Attributes:
        vector_store: The vector store instance.
        parser: The code parser instance.
        model: The sentence transformer model for embeddings.
        code_extensions: Set of file extensions to index.
        ignore_patterns: Set of patterns to ignore during indexing.
    """

    # Default file extensions to index
    DEFAULT_CODE_EXTENSIONS = {
        ".py",
        ".pyi",  # Python
        ".js",
        ".jsx",
        ".mjs",  # JavaScript
        ".ts",
        ".tsx",  # TypeScript
        ".java",  # Java
        ".cpp",
        ".cc",
        ".cxx",
        ".c++",
        ".hpp",
        ".h",  # C/C++
        ".c",  # C
        ".rs",  # Rust
        ".go",  # Go
        ".rb",  # Ruby
        ".php",  # PHP
        ".swift",  # Swift
        ".kt",
        ".kts",  # Kotlin
        ".scala",  # Scala
        ".r",
        ".R",  # R
        ".m",
        ".mm",  # Objective-C
        ".sh",
        ".bash",
        ".zsh",
        ".fish",  # Shell
        ".sql",  # SQL
        ".lua",  # Lua
        ".jl",  # Julia
    }

    # Default patterns to ignore
    DEFAULT_IGNORE_PATTERNS = {
        "__pycache__",
        ".git",
        ".svn",
        ".hg",
        "node_modules",
        ".venv",
        "venv",
        "env",
        ".env",
        "virtualenv",
        "dist",
        "build",
        "target",
        "out",
        ".idea",
        ".vscode",
        ".vs",
        "*.pyc",
        "*.pyo",
        "*.class",
        "*.o",
        "*.so",
        "*.dylib",
        "*.dll",
        "*.exe",
        ".DS_Store",
        "Thumbs.db",
        "*.log",
        "*.lock",
        "package-lock.json",
        "yarn.lock",
        "Cargo.lock",
        "poetry.lock",
        ".rassdb",  # Never index .rassdb folders
    }

    def __init__(
        self,
        db_path: str = "code_rag.db",
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        embedding_dim: int = 768,
        code_extensions: Optional[Set[str]] = None,
        ignore_patterns: Optional[Set[str]] = None,
        use_rassdb_config: bool = True,
    ) -> None:
        """Initialize the codebase indexer.

        Args:
            db_path: Path to the database file.
            model_name: Name of the sentence transformer model.
            embedding_dim: Dimension of embeddings.
            code_extensions: Set of file extensions to index (uses defaults if None).
            ignore_patterns: Set of patterns to ignore (uses defaults if None).
            use_rassdb_config: Whether to use .rassdb-config.toml files.
        """
        self.db_path = db_path
        self.parser = CodeParser()
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.embedding_strategy: Optional[EmbeddingStrategy] = None
        self.code_extensions = code_extensions or self.DEFAULT_CODE_EXTENSIONS
        self.ignore_patterns = ignore_patterns or set()
        self.use_rassdb_config = use_rassdb_config
        self.config: Optional[Dict] = None
        self.include_extensions: Set[str] = set()
        self.exclude_extensions: Set[str] = set()
        self.include_patterns: List[str] = []
        self.exclude_patterns: List[str] = []
        self._embedding_dim = embedding_dim
        self.vector_store: Optional[VectorStore] = None

    def _init_embedding_model(self) -> None:
        """Initialize the embedding model lazily."""
        if self.model is None:
            # Check for model override in configs (project first, then global)
            # This will be set after _load_rassdb_config is called
            logger.info(f"Loading embedding model: {self.model_name}")

            # Check if it's a cloud model first
            cloud_model = get_cloud_embedding_model(self.model_name)
            if cloud_model:
                self.model = cloud_model
                logger.info("✓ Cloud embedding model loaded")
            else:
                # Check if it's a GGUF model
                gguf_model = get_gguf_embedding_model(self.model_name)
                if gguf_model:
                    self.model = gguf_model
                    logger.info("✓ GGUF embedding model loaded")
                else:
                    # Load from standard HuggingFace cache location
                    self.model = SentenceTransformer(
                        self.model_name, trust_remote_code=True
                    )
                    logger.info("✓ Local embedding model loaded")

            # Initialize embedding strategy
            try:
                self.embedding_strategy = get_embedding_strategy(self.model_name)
                logger.info(
                    f"✓ Using {self.embedding_strategy.__class__.__name__} strategy"
                )
            except ValueError as e:
                logger.error(str(e))
                raise

            # Initialize vector store with correct embedding dimension
            if self.vector_store is None:
                # Always use the model's actual dimension
                embedding_dim = self.model.get_sentence_embedding_dimension()
                logger.info(
                    f"Initializing vector store with dimension: {embedding_dim}"
                )
                self.vector_store = VectorStore(self.db_path, embedding_dim)

    def _load_rassdb_config(self, base_path: Path) -> None:
        """Load .rassdb-config.toml configuration file.

        Args:
            base_path: The base directory path to look for config files.
        """
        # Check project config first (no merging with global)
        rassdb_config_path = base_path / ".rassdb-config.toml"
        if rassdb_config_path.exists():
            try:
                with open(rassdb_config_path, "rb") as f:
                    self.config = tomllib.load(f)
                logger.info("✓ Using project .rassdb-config.toml file")
            except Exception as e:
                logger.error(f"Failed to parse project .rassdb-config.toml: {e}")
                raise ValueError(f"Failed to parse .rassdb-config.toml: {e}") from e
        else:
            # Check global config only if no project config
            global_config_path = Path.home() / ".rassdb-config.toml"
            if global_config_path.exists():
                try:
                    with open(global_config_path, "rb") as f:
                        self.config = tomllib.load(f)
                    logger.info("✓ Using global ~/.rassdb-config.toml")
                except Exception as e:
                    logger.error(f"Failed to parse global ~/.rassdb-config.toml: {e}")
                    raise ValueError(
                        f"Failed to parse ~/.rassdb-config.toml: {e}"
                    ) from e
            else:
                self.config = None

        # Check for embedding model override
        if (
            self.config
            and "embedding-model" in self.config
            and "name" in self.config["embedding-model"]
        ):
            self.model_name = self.config["embedding-model"]["name"]
            logger.info(f"Using embedding model from config: {self.model_name}")

        # Build sets for efficient lookup
        if self.config:
            self.include_extensions = set()
            if "include" in self.config and "extensions" in self.config["include"]:
                extensions = self.config["include"]["extensions"]
                if isinstance(extensions, list):
                    self.include_extensions.update(extensions)

            self.exclude_extensions = set()
            if "exclude" in self.config and "extensions" in self.config["exclude"]:
                extensions = self.config["exclude"]["extensions"]
                if isinstance(extensions, list):
                    self.exclude_extensions.update(extensions)

            self.include_patterns = []
            if "include" in self.config and "paths" in self.config["include"]:
                paths = self.config["include"]["paths"]
                if isinstance(paths, list):
                    self.include_patterns = paths

            self.exclude_patterns = []
            if "exclude" in self.config and "paths" in self.config["exclude"]:
                paths = self.config["exclude"]["paths"]
                if isinstance(paths, list):
                    self.exclude_patterns = paths

            # Handle add-gitignore-to-exclude-paths from exclude section
            add_gitignore = True  # Default
            if "exclude" in self.config:
                add_gitignore = self.config["exclude"].get(
                    "add-gitignore-to-exclude-paths", True
                )
            if add_gitignore:
                gitignore_patterns = self._load_gitignore_patterns(base_path)
                self.exclude_patterns.extend(gitignore_patterns)
        else:
            # No config file - index ALL files (no extension filtering)
            self.include_extensions = set()
            self.exclude_extensions = set()
            self.include_patterns = []
            self.exclude_patterns = []

            # Default behavior when no config: still add gitignore patterns
            gitignore_patterns = self._load_gitignore_patterns(base_path)
            self.exclude_patterns.extend(gitignore_patterns)

    def _load_gitignore_patterns(self, base_path: Path) -> List[str]:
        """Load gitignore patterns from project and user gitignore files.

        Args:
            base_path: The base directory path of the project.

        Returns:
            List of gitignore patterns to exclude.
        """
        patterns = []

        # Load project .gitignore
        project_gitignore = base_path / ".gitignore"
        if project_gitignore.exists():
            patterns.extend(self._parse_gitignore_file(project_gitignore))
            logger.debug(f"Loaded {len(patterns)} patterns from project .gitignore")

        # Load user ~/.gitignore
        user_gitignore = Path.home() / ".gitignore"
        if user_gitignore.exists():
            user_patterns = self._parse_gitignore_file(user_gitignore)
            patterns.extend(user_patterns)
            logger.debug(f"Loaded {len(user_patterns)} patterns from user ~/.gitignore")

        return patterns

    def _parse_gitignore_file(self, gitignore_path: Path) -> List[str]:
        """Parse a gitignore file and return list of patterns.

        Args:
            gitignore_path: Path to the gitignore file.

        Returns:
            List of gitignore patterns.
        """
        patterns = []
        try:
            with open(gitignore_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith("#"):
                        # Convert gitignore patterns to glob patterns
                        # Gitignore patterns starting with / are relative to root
                        if line.startswith("/"):
                            patterns.append(line[1:])
                        else:
                            # Pattern can match at any level
                            if line.endswith("/"):
                                # Directory pattern - match anything inside it
                                patterns.append("**/" + line + "**")
                            else:
                                # File pattern - can match at any level
                                patterns.append("**/" + line)
        except Exception as e:
            logger.warning(f"Failed to read gitignore file {gitignore_path}: {e}")

        return patterns

    def _should_include_file(self, file_path: Path, base_path: Path) -> bool:
        """Determine if a file should be included based on config rules.

        Logic: File must have included extension AND match include path pattern (if any)
               AND NOT match exclude patterns AND NOT have excluded extension

        Args:
            file_path: The file path to check.
            base_path: The base directory path.

        Returns:
            True if file should be included.
        """
        relative_path = str(file_path.relative_to(base_path))
        file_ext = file_path.suffix

        # First check: file MUST have an included extension (if include extensions are defined)
        if self.include_extensions:
            if file_ext not in self.include_extensions:
                return False

        # Second check: if include paths are defined, file MUST match at least one
        if self.include_patterns:
            matches_include_pattern = False
            for pattern in self.include_patterns:
                if fnmatch.fnmatch(relative_path, pattern):
                    matches_include_pattern = True
                    break
            if not matches_include_pattern:
                return False

        # Third check: file must NOT match any exclude pattern
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(relative_path, pattern):
                return False

        # Fourth check: file must NOT have excluded extension
        if file_ext in self.exclude_extensions:
            return False

        # If we get here, file passes all checks
        return True

    def should_index_file(
        self,
        file_path: Path,
        max_file_size: int = 1024 * 1024,  # 1MB
        base_path: Optional[Path] = None,
    ) -> bool:
        """Check if a file should be indexed.

        Args:
            file_path: Path to the file.
            max_file_size: Maximum file size in bytes.
            base_path: Base directory path for relative path calculations.

        Returns:
            True if the file should be indexed.
        """
        # If we have RASSDB config enabled, use the new logic
        if self.use_rassdb_config and base_path:
            # Check config-based rules
            if not self._should_include_file(file_path, base_path):
                return False
        else:
            # Legacy behavior: just check extension against defaults
            if file_path.suffix.lower() not in self.code_extensions:
                return False

        # No default ignore patterns - all filtering is done through config

        # Check file size
        try:
            if file_path.stat().st_size > max_file_size:
                logger.debug(f"Skipping large file: {file_path}")
                return False
        except Exception:
            return False

        return True

    def create_embedding_text(
        self, chunk: CodeChunk, language: Optional[str], file_path: str
    ) -> str:
        """Create text for embedding using the appropriate strategy.

        Args:
            chunk: The code chunk.
            language: Programming language.
            file_path: Relative file path.

        Returns:
            Text to be embedded with context.
        """
        if hasattr(self, "embedding_strategy") and self.embedding_strategy:
            # Use the embedding strategy
            metadata = {
                "file_path": file_path,
                "language": language,
                "file_name": chunk.metadata.get("file_name", ""),
            }

            # Pass through all useful metadata from the chunk
            for key in [
                "docstring",
                "parent_class",
                "class_name",
                "function_name",
                "node_type",
            ]:
                if key in chunk.metadata:
                    metadata[key] = chunk.metadata[key]

            return self.embedding_strategy.prepare_code(chunk, metadata)
        else:
            # Fallback to old behavior if strategy not initialized
            # Build context parts
            context_parts = []

            # Add the actual code first
            context_parts.append(chunk.content)

            # Then append context as comments
            context_comments = []

            # Add parent class context if available
            if chunk.metadata.get("parent_class"):
                context_comments.append(
                    f"# Parent Class: {chunk.metadata['parent_class']}"
                )

            # Add function/method name if available
            if chunk.name:
                if chunk.chunk_type == "method":
                    context_comments.append(f"# Method: {chunk.name}")
                elif chunk.chunk_type == "function":
                    context_comments.append(f"# Function: {chunk.name}")
                elif chunk.chunk_type == "class":
                    context_comments.append(f"# Class: {chunk.name}")

            # Append context comments if any
            if context_comments:
                context_parts.append("\n" + "\n".join(context_comments))

            return "\n".join(context_parts)

    def is_binary_file(self, file_path: Path) -> bool:
        """Check if a file is binary by reading a sample of its content.

        Args:
            file_path: Path to the file to check.

        Returns:
            True if the file appears to be binary, False otherwise.
        """
        try:
            # Read first 8192 bytes to check for binary content
            with open(file_path, "rb") as f:
                chunk = f.read(8192)

            # Empty files are not binary
            if not chunk:
                return False

            # Check for null bytes (common in binary files)
            if b"\x00" in chunk:
                return True

            # Try to decode as UTF-8
            try:
                chunk.decode("utf-8")
                return False
            except UnicodeDecodeError:
                return True

        except Exception:
            # If we can't read the file, assume it's binary
            return True

    def index_file(
        self, file_path: Path, base_path: Path, max_chunk_size: int = 1500
    ) -> int:
        """Index a single file.

        Args:
            file_path: Path to the file to index.
            base_path: Base path for calculating relative paths.
            max_chunk_size: Maximum size of content to index per chunk.

        Returns:
            Number of chunks indexed.
        """
        try:
            # Skip binary files
            if self.is_binary_file(file_path):
                logger.debug(f"Skipping binary file: {file_path}")
                return 0

            # Ensure embedding model and vector store are initialized
            self._init_embedding_model()

            # Get file stats for change detection
            file_stats = file_path.stat()
            mtime = file_stats.st_mtime
            ctime = file_stats.st_ctime
            size = file_stats.st_size

            # Calculate relative path for storage
            relative_path = file_path.relative_to(base_path)
            relative_path_str = str(relative_path)

            # Check if file has changed
            if not self.vector_store.has_file_changed(
                relative_path_str, mtime, ctime, size
            ):
                logger.debug(f"Skipping unchanged file: {relative_path}")
                return 0

            # File has changed or is new - delete old data if it exists
            self.vector_store.delete_by_file(relative_path_str)

            # Read file content
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            # Parse the file into chunks
            chunks = self.parser.parse_file(str(file_path), content)

            if not chunks:
                # If no chunks found, index the whole file as one chunk
                # Use strategy's ideal chunk size if available
                if (
                    hasattr(self, "embedding_strategy")
                    and self.embedding_strategy
                    and max_chunk_size is None
                ):
                    max_chunk_size = self.embedding_strategy.ideal_chunk_size[
                        "max_chars"
                    ]
                elif max_chunk_size is None:
                    max_chunk_size = 3000

                chunks = [
                    CodeChunk(
                        content=content[:max_chunk_size],
                        chunk_type="file",
                        start_line=1,
                        end_line=len(content.split("\n")),
                        metadata={},
                    )
                ]

            # Generate embeddings and store
            language = self.parser.detect_language(str(file_path))

            # Ensure embedding model is loaded
            self._init_embedding_model()

            for chunk in chunks:
                # Skip empty chunks
                if not chunk.content.strip():
                    continue

                # Create focused embedding text
                embedding_text = self.create_embedding_text(
                    chunk, language, relative_path_str
                )

                # Generate embedding
                embedding = self.model.encode(
                    embedding_text, normalize_embeddings=True, show_progress_bar=False
                )

                # Ensure embedding is 1D array (some models return 2D array for single input)
                if embedding.ndim == 2 and embedding.shape[0] == 1:
                    embedding = embedding[0]

                # Add chunk metadata
                metadata = chunk.metadata.copy()
                metadata["file_name"] = file_path.name

                # Extract original boundaries from metadata if this is a part
                part_start_line = None
                part_end_line = None
                if chunk.chunk_type.endswith("_part"):
                    # For parts, the chunk lines are the part lines
                    part_start_line = chunk.start_line
                    part_end_line = chunk.end_line
                    # Get original boundaries from metadata if available
                    if "original_start_line" in metadata:
                        original_start = metadata.pop("original_start_line")
                        original_end = metadata.pop("original_end_line")
                    else:
                        # Fallback to part lines if not in metadata
                        original_start = chunk.start_line
                        original_end = chunk.end_line
                else:
                    # For non-parts, original and part lines are the same
                    original_start = chunk.start_line
                    original_end = chunk.end_line

                # Store in database
                self.vector_store.add_code_chunk(
                    file_path=relative_path_str,
                    content=chunk.content,
                    embedding=embedding,
                    language=language,
                    start_line=original_start,
                    end_line=original_end,
                    chunk_type=chunk.chunk_type,
                    metadata=metadata,
                    part_start_line=part_start_line,
                    part_end_line=part_end_line,
                )

            # Update file metadata after successful indexing
            self.vector_store.update_file_metadata(
                relative_path_str, mtime, ctime, size
            )

            return len(chunks)

        except Exception as e:
            logger.error(f"Error indexing {file_path}: {e}")
            return 0

    def index_directory(
        self,
        directory: str,
        show_progress: bool = True,
        clear_existing: bool = False,
    ) -> None:
        """Index all code files in a directory.

        Args:
            directory: Path to the directory to index.
            show_progress: Whether to show progress bar.
            clear_existing: Whether to clear existing data before indexing.
        """
        base_path = Path(directory).resolve()

        if not base_path.exists():
            raise ValueError(f"Directory {directory} does not exist")

        logger.info(f"Indexing directory: {base_path}")

        # Store base path for model loading
        self._base_path = base_path
        # Store show_progress flag for use in encode calls
        self._show_progress = show_progress

        # Load RASSDB configuration files if enabled
        if self.use_rassdb_config:
            self._load_rassdb_config(base_path)

        # Initialize model and vector store early to get correct dimensions
        self._init_embedding_model()

        if clear_existing:
            # Clear all existing data
            self.vector_store.conn.execute("DELETE FROM code_chunks")
            self.vector_store.conn.execute("DELETE FROM vec_embeddings")
            self.vector_store.conn.execute("DELETE FROM file_metadata")
            self.vector_store.conn.commit()
            logger.info("✓ Cleared existing database")

        # Store the root path in database metadata
        self.vector_store.set_metadata("root_path", str(base_path))
        logger.info(f"✓ Stored root path: {base_path}")

        # Collect all files to index
        files_to_index = []
        for file_path in base_path.rglob("*"):
            if file_path.is_file() and self.should_index_file(
                file_path, base_path=base_path
            ):
                files_to_index.append(file_path)

        logger.info(f"Found {len(files_to_index)} files to index")

        # Index files with progress bar
        total_chunks = 0

        # # Create progress UI (shellack interface - commented out for now)
        # progress = IndexingProgress(len(files_to_index), quiet=not show_progress)
        # if show_progress:
        #     progress.start()

        for idx, file_path in enumerate(files_to_index):
            # Show current file
            relative_path = str(file_path.relative_to(base_path))
            if show_progress:
                print(f"Indexing: {relative_path}")

            # Index the file
            chunks = self.index_file(file_path, base_path)
            total_chunks += chunks

        logger.info(
            f"\n✓ Indexed {total_chunks} code chunks from {len(files_to_index)} files"
        )

        # Print statistics
        stats = self.vector_store.get_statistics()
        logger.info("\nDatabase Statistics:")
        logger.info(f"  Model used: {self.model_name}")
        logger.info(f"  Total chunks: {stats['total_chunks']}")
        logger.info(f"  Unique files: {stats['unique_files']}")
        logger.info("  By language:")
        for lang, count in sorted(stats["by_language"].items()):
            logger.info(f"    {lang}: {count}")
        logger.info("  By type:")
        for chunk_type, count in sorted(stats["by_type"].items()):
            logger.info(f"    {chunk_type}: {count}")

    def close(self) -> None:
        """Close resources."""
        self.vector_store.close()
