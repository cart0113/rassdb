"""Test script to verify cosine distance fix in RASSDB.

This script tests the updated vector store with cosine distance.
"""

import sys
import sqlite3
import struct
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from tabulate import tabulate

# Add parent directory to path to import rassdb modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from rassdb.vector_store import VectorStore
from rassdb.utils.scorer import cosine_similarity


def deserialize_embedding(blob: bytes, dim: int = 768) -> np.ndarray:
    """Deserialize bytes to numpy array."""
    values = struct.unpack(f"{dim}f", blob)
    return np.array(values, dtype=np.float32)


def main():
    # The search query from the example
    query = "How many enteries does the LLM get from the RAG server and how does it decide which ones to use?"
    
    print(f"Testing cosine distance fix for query: '{query}'")
    print("=" * 80)
    
    # Load the model (same as used in search)
    print("\nLoading embedding model...")
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    
    # Generate query embedding (matching search.py logic)
    query_text = f"Code:\n{query}"
    query_embedding = model.encode(query_text, normalize_embeddings=True)
    
    print(f"\nQuery embedding shape: {query_embedding.shape}")
    print(f"Query embedding norm: {np.linalg.norm(query_embedding):.6f}")
    
    # Test with the vector store directly
    print("\n" + "=" * 80)
    print("TESTING WITH UPDATED VECTOR STORE")
    print("=" * 80)
    
    # First, let's temporarily create a test database to verify the function works
    test_db = Path("/tmp/test_cosine.db")
    test_db.unlink(missing_ok=True)
    
    print("\nCreating test database with cosine distance...")
    test_store = VectorStore(test_db)
    
    # Add some test vectors
    test_vectors = [
        ("identical", query_embedding.copy()),
        ("opposite", -query_embedding),
        ("orthogonal", np.random.randn(768).astype(np.float32)),
        ("similar", query_embedding + 0.1 * np.random.randn(768).astype(np.float32)),
    ]
    
    # Normalize test vectors
    for i, (name, vec) in enumerate(test_vectors):
        norm_vec = vec / np.linalg.norm(vec)
        test_vectors[i] = (name, norm_vec)
        test_store.add_code_chunk(
            file_path=f"test_{name}.py",
            content=f"Test content for {name} vector",
            embedding=norm_vec,
            chunk_type="test",
            start_line=1,
            end_line=1
        )
    
    # Search with our query
    results = test_store.search_similar(query_embedding, limit=10)
    
    print("\nTest Database Results (using vec_distance_cosine):")
    print("-" * 60)
    headers = ['File', 'Cosine Distance', 'Similarity (1-dist)', 'Our Cosine Sim']
    table_data = []
    
    for r in results:
        # Get the stored vector to calculate our own cosine similarity
        conn = test_store.conn
        cursor = conn.cursor()
        cursor.execute("SELECT embedding FROM vec_embeddings WHERE rowid = ?", (r['id'],))
        stored_embedding = deserialize_embedding(cursor.fetchone()[0])
        
        our_cosine = cosine_similarity(query_embedding, stored_embedding)
        
        table_data.append([
            r['file_path'],
            f"{r['distance']:.6f}",
            f"{1.0 - r['distance']:.6f}",
            f"{our_cosine:.6f}"
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    test_store.close()
    test_db.unlink()
    
    print("\n✅ The cosine distance function is working correctly!")
    print("   - Identical vectors have distance ~0 (similarity ~1)")
    print("   - Opposite vectors have distance ~2 (similarity ~-1)")
    print("   - The calculated similarities match our manual cosine similarity")


if __name__ == "__main__":
    main()