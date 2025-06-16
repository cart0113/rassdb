"""Codebase indexing functionality for building RAG databases.

This module provides the main indexing functionality to process codebases
and store their embeddings in the vector database.
"""

import logging
from pathlib import Path
from typing import Set, Optional, Callable, Any
from tqdm import tqdm
import gitignore_parser
from sentence_transformers import SentenceTransformer
import numpy as np
import numpy.typing as npt

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
        ".py", ".pyi",  # Python
        ".js", ".jsx", ".mjs",  # JavaScript
        ".ts", ".tsx",  # TypeScript
        ".java",  # Java
        ".cpp", ".cc", ".cxx", ".c++", ".hpp", ".h",  # C/C++
        ".c",  # C
        ".rs",  # Rust
        ".go",  # Go
        ".rb",  # Ruby
        ".php",  # PHP
        ".swift",  # Swift
        ".kt", ".kts",  # Kotlin
        ".scala",  # Scala
        ".r", ".R",  # R
        ".m", ".mm",  # Objective-C
        ".sh", ".bash", ".zsh", ".fish",  # Shell
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
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        embedding_dim: int = 768,
        code_extensions: Optional[Set[str]] = None,
        ignore_patterns: Optional[Set[str]] = None,
    ) -> None:
        """Initialize the codebase indexer.
        
        Args:
            db_path: Path to the database file.
            model_name: Name of the sentence transformer model.
            embedding_dim: Dimension of embeddings.
            code_extensions: Set of file extensions to index (uses defaults if None).
            ignore_patterns: Set of patterns to ignore (uses defaults if None).
        """
        self.vector_store = VectorStore(db_path, embedding_dim)
        self.parser = CodeParser()
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.code_extensions = code_extensions or self.DEFAULT_CODE_EXTENSIONS
        self.ignore_patterns = ignore_patterns or self.DEFAULT_IGNORE_PATTERNS
    
    def _init_embedding_model(self) -> None:
        """Initialize the embedding model lazily."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
            logger.info("✓ Embedding model loaded")
    
    def should_index_file(
        self,
        file_path: Path,
        gitignore_func: Optional[Callable[[str], bool]] = None,
        max_file_size: int = 1024 * 1024,  # 1MB
    ) -> bool:
        """Check if a file should be indexed.
        
        Args:
            file_path: Path to the file.
            gitignore_func: Function to check gitignore rules.
            max_file_size: Maximum file size in bytes.
            
        Returns:
            True if the file should be indexed.
        """
        # Check extension
        if file_path.suffix.lower() not in self.code_extensions:
            return False
        
        # Check ignore patterns
        for pattern in self.ignore_patterns:
            if pattern.startswith("*"):
                if file_path.name.endswith(pattern[1:]):
                    return False
            elif pattern in str(file_path):
                return False
        
        # Check gitignore if available
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
        
        Args:
            chunk: The code chunk.
            language: Programming language.
            file_path: Relative file path.
            
        Returns:
            Text to be embedded.
        """
        parts = []
        
        # Add language as context
        if language:
            parts.append(f"Language: {language}")
        
        # Add type context
        if chunk.chunk_type:
            parts.append(f"Type: {chunk.chunk_type}")
        
        # Add name if available (function/class name)
        if chunk.name:
            parts.append(f"Name: {chunk.name}")
        
        # Add file context (just the filename, not full path)
        parts.append(f"File: {Path(file_path).name}")
        
        # Add the actual code content - this should be the main focus
        parts.append("Code:")
        parts.append(chunk.content)
        
        return "\n".join(parts)
    
    def index_file(
        self, file_path: Path, base_path: Path, max_chunk_size: int = 5000
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
            relative_path = file_path.relative_to(base_path)
            language = self.parser.detect_language(str(file_path))
            
            # Ensure embedding model is loaded
            self._init_embedding_model()
            
            for chunk in chunks:
                # Skip empty chunks
                if not chunk.content.strip():
                    continue
                
                # Create focused embedding text
                embedding_text = self.create_embedding_text(
                    chunk, language, str(relative_path)
                )
                
                # Generate embedding
                embedding = self.model.encode(
                    embedding_text, normalize_embeddings=True
                )
                
                # Add chunk metadata
                metadata = chunk.metadata.copy()
                metadata["file_name"] = file_path.name
                
                # Store in database
                self.vector_store.add_code_chunk(
                    file_path=str(relative_path),
                    content=chunk.content,
                    embedding=embedding,
                    language=language,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    chunk_type=chunk.chunk_type,
                    metadata=metadata,
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
            self.vector_store.conn.commit()
            logger.info("✓ Cleared existing database")
        
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
                file_path, gitignore_func
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