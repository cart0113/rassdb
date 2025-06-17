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

    def __init__(
        self, db_path: Union[str, Path] = "code_rag.db", embedding_dim: int = 768
    ) -> None:
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

        # Add REGEXP function
        def regexp(pattern, string):
            if pattern is None or string is None:
                return False
            import re

            try:
                return re.search(pattern, string) is not None
            except re.error:
                return False

        self._conn.create_function("REGEXP", 2, regexp)

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
            
            CREATE TABLE IF NOT EXISTS file_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL UNIQUE,
                mtime REAL NOT NULL,
                ctime REAL NOT NULL,
                size INTEGER NOT NULL,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_file_metadata_path ON file_metadata(file_path);
        """)

        # Create FTS5 virtual table for full-text search
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='code_chunks_fts'
        """)

        if not cursor.fetchone():
            # Create FTS5 table with advanced tokenization
            self._conn.execute("""
                CREATE VIRTUAL TABLE code_chunks_fts USING fts5(
                    content,
                    file_path UNINDEXED,
                    chunk_type UNINDEXED,
                    language UNINDEXED,
                    content=code_chunks,
                    content_rowid=id,
                    tokenize='unicode61 remove_diacritics 2'
                );
            """)

            # Create triggers to keep FTS5 table in sync
            self._conn.executescript("""
                CREATE TRIGGER IF NOT EXISTS code_chunks_ai AFTER INSERT ON code_chunks BEGIN
                    INSERT INTO code_chunks_fts(rowid, content, file_path, chunk_type, language)
                    VALUES (new.id, new.content, new.file_path, new.chunk_type, new.language);
                END;
                
                CREATE TRIGGER IF NOT EXISTS code_chunks_au AFTER UPDATE ON code_chunks BEGIN
                    UPDATE code_chunks_fts 
                    SET content = new.content, 
                        file_path = new.file_path,
                        chunk_type = new.chunk_type,
                        language = new.language
                    WHERE rowid = new.id;
                END;
                
                CREATE TRIGGER IF NOT EXISTS code_chunks_ad AFTER DELETE ON code_chunks BEGIN
                    DELETE FROM code_chunks_fts WHERE rowid = old.id;
                END;
            """)

            # Populate FTS5 table with existing data
            self._conn.execute("""
                INSERT INTO code_chunks_fts(rowid, content, file_path, chunk_type, language)
                SELECT id, content, file_path, chunk_type, language FROM code_chunks;
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
            where_clauses.append("c.file_path REGEXP ?")
            params.append(file_pattern)

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
                vec_distance_cosine(v.embedding, ?) as distance
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

    def search_lexical(
        self,
        query: str,
        limit: int = 50,
        language: Optional[str] = None,
        file_pattern: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for code chunks using FTS5 with BM25 ranking.

        Args:
            query: The text to search for. Supports FTS5 query syntax:
                   - "term1 term2" for AND search
                   - "term1 OR term2" for OR search
                   - "\"exact phrase\"" for phrase search
                   - "term*" for prefix search
                   - "term1 NEAR(term2, 10)" for proximity search
            limit: Maximum number of results to return.
            language: Filter by programming language.
            file_pattern: Filter by file path pattern.

        Returns:
            List of dictionaries containing matching code chunks with BM25 scores.
        """
        cursor = self.conn.cursor()

        # Prepare FTS5 query - escape special characters if needed
        fts_query = query

        # Build WHERE clauses for additional filters
        where_clauses = []
        params = []

        if language:
            where_clauses.append("c.language = ?")
            params.append(language)

        if file_pattern:
            where_clauses.append("c.file_path REGEXP ?")
            params.append(file_pattern)

        where_clause = ""
        if where_clauses:
            where_clause = "AND " + " AND ".join(where_clauses)

        # Use FTS5 with BM25 ranking
        query_sql = f"""
            SELECT 
                c.id,
                c.file_path,
                c.content,
                c.language,
                c.start_line,
                c.end_line,
                c.chunk_type,
                c.metadata,
                -fts.rank as score,
                snippet(code_chunks_fts, 0, '<match>', '</match>', '...', 32) as snippet,
                highlight(code_chunks_fts, 0, '<match>', '</match>') as highlighted_content
            FROM code_chunks c
            JOIN code_chunks_fts fts ON c.id = fts.rowid
            WHERE code_chunks_fts MATCH ?
            {where_clause}
            ORDER BY rank
            LIMIT ?
        """

        try:
            cursor.execute(query_sql, [fts_query] + params + [limit])
        except sqlite3.OperationalError as e:
            # Fallback to simple token search if FTS5 query syntax is invalid
            if "fts5: syntax error" in str(e):
                # Escape special characters and retry with phrase search
                fts_query = '"' + query.replace('"', '""') + '"'
                cursor.execute(query_sql, [fts_query] + params + [limit])
            else:
                raise

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            if result["metadata"]:
                result["metadata"] = json.loads(result["metadata"])

            # Calculate a normalized similarity score from BM25 rank
            # BM25 scores are negative, with closer to 0 being better
            # Normalize to 0-1 range where 1 is best match
            bm25_score = result.pop("score", 0)
            result["similarity"] = 1.0 / (1.0 + abs(bm25_score))

            # Clean up snippet and highlighted content
            result["snippet"] = result.get("snippet", "").strip()

            # Remove highlighted content if not needed to save space
            result.pop("highlighted_content", None)

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

        # Delete from file metadata
        cursor.execute(
            "DELETE FROM file_metadata WHERE file_path = ?",
            (file_path,),
        )

        self.conn.commit()

    def has_file_changed(
        self, file_path: str, mtime: float, ctime: float, size: int
    ) -> bool:
        """Check if a file has changed since it was last indexed.

        Args:
            file_path: Path to the file.
            mtime: Current modification time of the file.
            ctime: Current change time of the file.
            size: Current size of the file in bytes.

        Returns:
            True if the file has changed or is not in the database, False otherwise.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT mtime, ctime, size 
            FROM file_metadata 
            WHERE file_path = ?
            """,
            (file_path,),
        )

        row = cursor.fetchone()
        if not row:
            # File not in database, needs indexing
            return True

        # Check if any of the metadata has changed
        stored_mtime, stored_ctime, stored_size = row
        return (
            abs(stored_mtime - mtime) > 0.001  # Float comparison with small epsilon
            or abs(stored_ctime - ctime) > 0.001
            or stored_size != size
        )

    def update_file_metadata(
        self, file_path: str, mtime: float, ctime: float, size: int
    ) -> None:
        """Update or insert file metadata after indexing.

        Args:
            file_path: Path to the file.
            mtime: Modification time of the file.
            ctime: Change time of the file.
            size: Size of the file in bytes.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO file_metadata (file_path, mtime, ctime, size)
            VALUES (?, ?, ?, ?)
            """,
            (file_path, mtime, ctime, size),
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
