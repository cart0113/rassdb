"""SQLite vector store using sqlite-vec for efficient similarity search.

This module provides a vector store implementation using SQLite with the sqlite-vec
extension for storing and searching code embeddings.
"""

import sqlite3
import json
import struct
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from contextlib import contextmanager

import numpy as np
import numpy.typing as npt
import sqlite_vec


class VectorStore:
    """Vector store for code embeddings using SQLite and sqlite-vec.
    
    This class manages storage and retrieval of code chunks with their embeddings,
    supporting efficient similarity search using the sqlite-vec extension.
    
    Attributes:
        db_path: Path to the SQLite database file.
        embedding_dim: Dimension of the embedding vectors (default: 768 for nomic-embed).
    """
    
    def __init__(self, db_path: Union[str, Path] = "code_rag.db", embedding_dim: int = 768) -> None:
        """Initialize the vector store.
        
        Args:
            db_path: Path to the SQLite database file.
            embedding_dim: Dimension of the embedding vectors.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()
    
    @property
    def conn(self) -> sqlite3.Connection:
        """Get the database connection, creating it if necessary."""
        if self._conn is None:
            self._init_db()
        return self._conn
    
    def _init_db(self) -> None:
        """Initialize SQLite database with vector extension."""
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        
        # Load sqlite-vec extension
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        
        # Create tables
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS code_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                content TEXT NOT NULL,
                language TEXT,
                start_line INTEGER,
                end_line INTEGER,
                chunk_type TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_file_path ON code_chunks(file_path);
            CREATE INDEX IF NOT EXISTS idx_language ON code_chunks(language);
            CREATE INDEX IF NOT EXISTS idx_chunk_type ON code_chunks(chunk_type);
        """)
        
        # Create vector table with proper configuration
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='vec_embeddings'
        """)
        
        if not cursor.fetchone():
            self._conn.execute(f"""
                CREATE VIRTUAL TABLE vec_embeddings USING vec0(
                    embedding float[{self.embedding_dim}]
                );
            """)
        
        self._conn.commit()
    
    def _serialize_embedding(self, embedding: npt.NDArray[np.float32]) -> bytes:
        """Serialize numpy array to bytes for sqlite-vec.
        
        Args:
            embedding: Numpy array of embeddings.
            
        Returns:
            Binary representation of the embedding.
        """
        embedding = embedding.astype(np.float32)
        return struct.pack(f"{len(embedding)}f", *embedding)
    
    def _deserialize_embedding(self, blob: bytes) -> npt.NDArray[np.float32]:
        """Deserialize bytes to numpy array.
        
        Args:
            blob: Binary data from database.
            
        Returns:
            Numpy array of embeddings.
        """
        values = struct.unpack(f"{self.embedding_dim}f", blob)
        return np.array(values, dtype=np.float32)
    
    def add_code_chunk(
        self,
        file_path: str,
        content: str,
        embedding: npt.NDArray[np.float32],
        language: Optional[str] = None,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        chunk_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add a code chunk with its embedding to the database.
        
        Args:
            file_path: Path to the source file.
            content: The code content.
            embedding: Embedding vector for the code.
            language: Programming language of the code.
            start_line: Starting line number in the source file.
            end_line: Ending line number in the source file.
            chunk_type: Type of code chunk (e.g., 'function', 'class').
            metadata: Additional metadata as a dictionary.
            
        Returns:
            The ID of the inserted chunk.
        """
        cursor = self.conn.cursor()
        
        # Insert code chunk
        cursor.execute(
            """
            INSERT INTO code_chunks (file_path, content, language, start_line, 
                                   end_line, chunk_type, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_path,
                content,
                language,
                start_line,
                end_line,
                chunk_type,
                json.dumps(metadata) if metadata else None,
            ),
        )
        
        chunk_id = cursor.lastrowid
        
        # Serialize embedding for sqlite-vec
        embedding_bytes = self._serialize_embedding(embedding)
        
        # Insert into vector table
        cursor.execute(
            """
            INSERT INTO vec_embeddings (rowid, embedding)
            VALUES (?, ?)
            """,
            (chunk_id, embedding_bytes),
        )
        
        self.conn.commit()
        return chunk_id
    
    def search_similar(
        self,
        query_embedding: npt.NDArray[np.float32],
        limit: int = 10,
        language: Optional[str] = None,
        file_pattern: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar code chunks using vector similarity.
        
        Args:
            query_embedding: The query embedding vector.
            limit: Maximum number of results to return.
            language: Filter by programming language.
            file_pattern: Filter by file path pattern.
            
        Returns:
            List of dictionaries containing similar code chunks with distance scores.
        """
        cursor = self.conn.cursor()
        
        # Build WHERE clause for filters
        where_clauses = []
        params = []
        
        if language:
            where_clauses.append("c.language = ?")
            params.append(language)
        
        if file_pattern:
            where_clauses.append("c.file_path LIKE ?")
            params.append(f"%{file_pattern}%")
        
        where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        
        # Serialize query embedding
        query_bytes = self._serialize_embedding(query_embedding)
        
        # Search using sqlite-vec
        query = f"""
            SELECT 
                c.id,
                c.file_path,
                c.content,
                c.language,
                c.start_line,
                c.end_line,
                c.chunk_type,
                c.metadata,
                vec_distance_l2(v.embedding, ?) as distance
            FROM vec_embeddings v
            JOIN code_chunks c ON v.rowid = c.id
            {where_clause}
            ORDER BY distance ASC
            LIMIT ?
        """
        
        cursor.execute(query, [query_bytes] + params + [limit])
        
        results = []
        for row in cursor.fetchall():
            result = dict(row)
            if result["metadata"]:
                result["metadata"] = json.loads(result["metadata"])
            results.append(result)
        
        return results
    
    def search_literal(
        self,
        query: str,
        limit: int = 50,
        language: Optional[str] = None,
        file_pattern: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for code chunks containing literal text.
        
        Args:
            query: The text to search for.
            limit: Maximum number of results to return.
            language: Filter by programming language.
            file_pattern: Filter by file path pattern.
            
        Returns:
            List of dictionaries containing matching code chunks.
        """
        cursor = self.conn.cursor()
        
        # Build WHERE clause
        where_clauses = ["c.content LIKE ?"]
        params = [f"%{query}%"]
        
        if language:
            where_clauses.append("c.language = ?")
            params.append(language)
        
        if file_pattern:
            where_clauses.append("c.file_path LIKE ?")
            params.append(f"%{file_pattern}%")
        
        where_clause = f"WHERE {' AND '.join(where_clauses)}"
        
        query_sql = f"""
            SELECT 
                c.id,
                c.file_path,
                c.content,
                c.language,
                c.start_line,
                c.end_line,
                c.chunk_type,
                c.metadata
            FROM code_chunks c
            {where_clause}
            ORDER BY c.file_path, c.start_line
            LIMIT ?
        """
        
        cursor.execute(query_sql, params + [limit])
        
        results = []
        for row in cursor.fetchall():
            result = dict(row)
            if result["metadata"]:
                result["metadata"] = json.loads(result["metadata"])
            results.append(result)
        
        return results
    
    def get_chunk_by_id(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific code chunk by ID.
        
        Args:
            chunk_id: The ID of the chunk to retrieve.
            
        Returns:
            Dictionary containing the chunk data, or None if not found.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM code_chunks WHERE id = ?",
            (chunk_id,),
        )
        
        row = cursor.fetchone()
        if row:
            result = dict(row)
            if result["metadata"]:
                result["metadata"] = json.loads(result["metadata"])
            return result
        return None
    
    def delete_by_file(self, file_path: str) -> None:
        """Delete all chunks from a specific file.
        
        Args:
            file_path: Path of the file whose chunks should be deleted.
        """
        cursor = self.conn.cursor()
        
        # Get IDs to delete
        cursor.execute(
            "SELECT id FROM code_chunks WHERE file_path = ?",
            (file_path,),
        )
        ids_to_delete = [row[0] for row in cursor.fetchall()]
        
        # Delete from vector table
        for chunk_id in ids_to_delete:
            cursor.execute(
                "DELETE FROM vec_embeddings WHERE rowid = ?",
                (chunk_id,),
            )
        
        # Delete from chunks table
        cursor.execute(
            "DELETE FROM code_chunks WHERE file_path = ?",
            (file_path,),
        )
        
        self.conn.commit()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Dictionary containing various statistics about the database.
        """
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Total chunks
        cursor.execute("SELECT COUNT(*) FROM code_chunks")
        stats["total_chunks"] = cursor.fetchone()[0]
        
        # Chunks by language
        cursor.execute("""
            SELECT language, COUNT(*) as count 
            FROM code_chunks 
            WHERE language IS NOT NULL
            GROUP BY language
        """)
        stats["by_language"] = dict(cursor.fetchall())
        
        # Chunks by type
        cursor.execute("""
            SELECT chunk_type, COUNT(*) as count 
            FROM code_chunks 
            WHERE chunk_type IS NOT NULL
            GROUP BY chunk_type
        """)
        stats["by_type"] = dict(cursor.fetchall())
        
        # Number of unique files
        cursor.execute("SELECT COUNT(DISTINCT file_path) FROM code_chunks")
        stats["unique_files"] = cursor.fetchone()[0]
        
        return stats
    
    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def __enter__(self) -> "VectorStore":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()