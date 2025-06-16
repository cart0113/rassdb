"""Codebase indexing functionality for building RAG databases.

This module provides the main indexing functionality to process codebases
and store their embeddings in the vector database.
"""

import logging
from pathlib import Path
from typing import Set, Optional, Callable, Any, Dict, List
from tqdm import tqdm
import gitignore_parser
from sentence_transformers import SentenceTransformer
import numpy as np
import numpy.typing as npt
import tomllib
import fnmatch

from rassdb.vector_store import VectorStore
from rassdb.code_parser import CodeParser, CodeChunk

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
    }

    def __init__(
        self,
        db_path: str = "code_rag.db",
        model_name: str = "nomic-ai/nomic-embed-code-v1.5",
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
            use_rassdb_config: Whether to use .rassdb-ignore and .rassdb-include.toml files.
        """
        self.vector_store = VectorStore(db_path, embedding_dim)
        self.parser = CodeParser()
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.code_extensions = code_extensions or self.DEFAULT_CODE_EXTENSIONS
        self.ignore_patterns = ignore_patterns or self.DEFAULT_IGNORE_PATTERNS
        self.use_rassdb_config = use_rassdb_config
        self.config: Optional[Dict] = None
        self.include_extensions: Set[str] = set()
        self.exclude_extensions: Set[str] = set()
        self.include_patterns: List[str] = []
        self.exclude_patterns: List[str] = []

    def _init_embedding_model(self) -> None:
        """Initialize the embedding model lazily."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
            logger.info("✓ Embedding model loaded")

    def _load_rassdb_config(self, base_path: Path) -> None:
        """Load .rassdb-config.toml configuration file.

        Args:
            base_path: The base directory path to look for config files.
        """
        # Load .rassdb-config.toml
        rassdb_config_path = base_path / ".rassdb-config.toml"
        if rassdb_config_path.exists():
            try:
                with open(rassdb_config_path, "rb") as f:
                    self.config = tomllib.load(f)
                logger.info("✓ Using .rassdb-config.toml file")

                # Build sets for efficient lookup
                self.include_extensions = set()
                if "include-extensions" in self.config:
                    for lang_exts in self.config["include-extensions"].values():
                        self.include_extensions.update(lang_exts)

                self.exclude_extensions = set()
                if "exclude-extensions" in self.config:
                    for category_exts in self.config["exclude-extensions"].values():
                        self.exclude_extensions.update(category_exts)

                self.include_patterns = []
                if (
                    "include-paths" in self.config
                    and "patterns" in self.config["include-paths"]
                ):
                    self.include_patterns = self.config["include-paths"]["patterns"]

                self.exclude_patterns = []
                if (
                    "exclude-paths" in self.config
                    and "patterns" in self.config["exclude-paths"]
                ):
                    self.exclude_patterns = self.config["exclude-paths"]["patterns"]

            except Exception as e:
                logger.warning(f"Failed to load .rassdb-config.toml: {e}")
        else:
            # No config file, use defaults
            self.config = None
            self.include_extensions = self.code_extensions
            self.exclude_extensions = set()
            self.include_patterns = []
            self.exclude_patterns = []

    def _should_include_file(self, file_path: Path, base_path: Path) -> bool:
        """Determine if a file should be included based on config rules.

        Priority: INCLUDE always wins over EXCLUDE

        Args:
            file_path: The file path to check.
            base_path: The base directory path.

        Returns:
            True if file should be included.
        """
        relative_path = str(file_path.relative_to(base_path))
        file_ext = file_path.suffix.lower()

        # Check if file matches any include pattern - INCLUDES ALWAYS WIN
        for pattern in self.include_patterns:
            if fnmatch.fnmatch(relative_path, pattern):
                return True

        # Check if extension is explicitly included - INCLUDES ALWAYS WIN
        if file_ext in self.include_extensions:
            return True

        # Now check excludes (only if not explicitly included)
        # Check exclude patterns
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(relative_path, pattern):
                return False

        # Check exclude extensions
        if file_ext in self.exclude_extensions:
            return False

        # If we have a config file and the extension isn't in include list, exclude it
        if (
            self.config
            and self.include_extensions
            and file_ext not in self.include_extensions
        ):
            return False

        # Default: include if no config or extension is in default list
        return file_ext in self.code_extensions

    def should_index_file(
        self,
        file_path: Path,
        gitignore_func: Optional[Callable[[str], bool]] = None,
        max_file_size: int = 1024 * 1024,  # 1MB
        base_path: Optional[Path] = None,
    ) -> bool:
        """Check if a file should be indexed.

        Args:
            file_path: Path to the file.
            gitignore_func: Function to check gitignore rules.
            max_file_size: Maximum file size in bytes.
            base_path: Base directory path for relative path calculations.

        Returns:
            True if the file should be indexed.
        """
        # If we have RASSDB config and base_path, use the new logic
        if self.use_rassdb_config and self.config and base_path:
            # Check config-based rules first
            if not self._should_include_file(file_path, base_path):
                return False
        else:
            # Legacy behavior: just check extension against defaults
            if file_path.suffix.lower() not in self.code_extensions:
                return False

        # Check default ignore patterns (from DEFAULT_IGNORE_PATTERNS)
        for pattern in self.ignore_patterns:
            if pattern.startswith("*"):
                if file_path.name.endswith(pattern[1:]):
                    return False
            elif pattern in str(file_path):
                return False

        # Check gitignore if available (works in conjunction with our rules)
        if gitignore_func and gitignore_func(str(file_path)):
            return False

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
        """Create text for embedding that focuses on code content and semantics.

        Since we're using Nomic Embed Code model, we send just the raw code
        as it's specifically trained to understand code structure and semantics.
        Per best practices, we avoid adding any metadata like language identifiers,
        chunk type labels, or parent class information.

        Args:
            chunk: The code chunk.
            language: Programming language.
            file_path: Relative file path.

        Returns:
            Text to be embedded (just the raw code).
        """
        # Return only the raw code - no metadata, no language identifiers
        return chunk.content

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
                embedding = self.model.encode(embedding_text, normalize_embeddings=True)

                # Add chunk metadata
                metadata = chunk.metadata.copy()
                metadata["file_name"] = file_path.name

                # Store in database
                self.vector_store.add_code_chunk(
                    file_path=relative_path_str,
                    content=chunk.content,
                    embedding=embedding,
                    language=language,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    chunk_type=chunk.chunk_type,
                    metadata=metadata,
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
        use_gitignore: bool = True,
        show_progress: bool = True,
        clear_existing: bool = False,
    ) -> None:
        """Index all code files in a directory.

        Args:
            directory: Path to the directory to index.
            use_gitignore: Whether to respect .gitignore files.
            show_progress: Whether to show progress bar.
            clear_existing: Whether to clear existing data before indexing.
        """
        base_path = Path(directory).resolve()

        if not base_path.exists():
            raise ValueError(f"Directory {directory} does not exist")

        logger.info(f"Indexing directory: {base_path}")

        if clear_existing:
            # Clear all existing data
            self.vector_store.conn.execute("DELETE FROM code_chunks")
            self.vector_store.conn.execute("DELETE FROM vec_embeddings")
            self.vector_store.conn.execute("DELETE FROM file_metadata")
            self.vector_store.conn.commit()
            logger.info("✓ Cleared existing database")

        # Load RASSDB configuration files if enabled
        if self.use_rassdb_config:
            self._load_rassdb_config(base_path)

        # Load gitignore if available
        gitignore_func = None
        if use_gitignore:
            gitignore_path = base_path / ".gitignore"
            if gitignore_path.exists():
                gitignore_func = gitignore_parser.parse_gitignore(gitignore_path)
                logger.info("✓ Using .gitignore file")

        # Collect all files to index
        files_to_index = []
        for file_path in base_path.rglob("*"):
            if file_path.is_file() and self.should_index_file(
                file_path, gitignore_func, base_path=base_path
            ):
                files_to_index.append(file_path)

        logger.info(f"Found {len(files_to_index)} files to index")

        # Index files with progress bar
        total_chunks = 0
        iterator = files_to_index

        if show_progress:
            iterator = tqdm(files_to_index, desc="Indexing files")

        for file_path in iterator:
            if show_progress:
                iterator.set_description(f"Indexing {file_path.name}")
            chunks = self.index_file(file_path, base_path)
            total_chunks += chunks

        logger.info(
            f"\n✓ Indexed {total_chunks} code chunks from {len(files_to_index)} files"
        )

        # Print statistics
        stats = self.vector_store.get_statistics()
        logger.info("\nDatabase Statistics:")
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
