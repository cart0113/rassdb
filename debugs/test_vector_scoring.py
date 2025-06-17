"""Debug script to analyze vector scoring issues in RASSDB.

This script:
1. Forms a vector for a search string
2. Retrieves snippets from the database
3. Recomputes similarity scores using our scorer
4. Compares with database scores
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
from rassdb.utils.scorer import (
    cosine_similarity,
    euclidean_distance,
    l2_to_similarity_score,
    debug_vector_similarity,
    normalize_vector,
)


def deserialize_embedding(blob: bytes, dim: int = 768) -> np.ndarray:
    """Deserialize bytes to numpy array."""
    values = struct.unpack(f"{dim}f", blob)
    return np.array(values, dtype=np.float32)


def main():
    # The search query from the example
    query = "How many enteries does the LLM get from the RAG server and how does it decide which ones to use?"
    
    # Path to the database
    db_path = Path("/Users/ajcarter/workspace/GIT_RASSDB/example-chat-bot/.rassdb/example-chat-bot-nomic-embed-text-v1.5.rassdb")
    
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return
    
    print(f"Analyzing vector scoring for query: '{query}'")
    print("=" * 80)
    
    # Load the model (same as used in search)
    print("\nLoading embedding model...")
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    
    # Generate query embedding (matching search.py logic)
    query_text = f"Code:\n{query}"
    query_embedding = model.encode(query_text, normalize_embeddings=True)
    
    print(f"\nQuery embedding shape: {query_embedding.shape}")
    print(f"Query embedding norm: {np.linalg.norm(query_embedding):.6f}")
    print(f"Query is normalized: {abs(np.linalg.norm(query_embedding) - 1.0) < 1e-6}")
    
    # Connect to database and run raw SQL query
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    
    # Load sqlite-vec extension
    conn.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    
    # Serialize query embedding for sqlite-vec
    query_bytes = struct.pack(f"{len(query_embedding)}f", *query_embedding)
    
    # Run the exact same query as vector_store.py
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            c.id,
            c.file_path,
            c.content,
            c.language,
            c.start_line,
            c.end_line,
            c.chunk_type,
            v.embedding,
            vec_distance_l2(v.embedding, ?) as distance
        FROM vec_embeddings v
        JOIN code_chunks c ON v.rowid = c.id
        ORDER BY distance ASC
        LIMIT 10
    """, [query_bytes])
    
    results = []
    print("\n" + "=" * 80)
    print("TOP 10 RESULTS ANALYSIS")
    print("=" * 80)
    
    for i, row in enumerate(cursor.fetchall()):
        # Get the stored embedding
        stored_embedding = deserialize_embedding(row['embedding'])
        
        # Calculate various metrics
        debug_info = debug_vector_similarity(
            query_embedding, 
            stored_embedding,
            "Query",
            f"Result_{i+1}"
        )
        
        # Database calculations
        db_l2_distance = row['distance']
        db_similarity = 1.0 / (1.0 + db_l2_distance)  # As done in search.py
        
        # Our calculations
        our_cosine_sim = debug_info['similarities']['cosine_similarity']
        our_l2_dist = debug_info['similarities']['l2_distance']
        our_l2_to_sim = debug_info['similarities']['l2_to_similarity_score']
        
        result = {
            'rank': i + 1,
            'file': row['file_path'],
            'lines': f"{row['start_line']}-{row['end_line']}",
            'type': row['chunk_type'],
            'db_l2_dist': db_l2_distance,
            'db_similarity': db_similarity,
            'our_l2_dist': our_l2_dist,
            'our_cosine_sim': our_cosine_sim,
            'stored_vec_norm': debug_info['vector_info'][f'Result_{i+1}_norm'],
            'stored_vec_normalized': debug_info['vector_info'][f'Result_{i+1}_is_normalized'],
        }
        results.append(result)
        
        # Print detailed analysis for first few results
        if i < 3:
            print(f"\nResult #{i+1}: {row['file_path']} (lines {row['start_line']}-{row['end_line']})")
            print(f"Content preview: {row['content'][:100]}...")
            print(f"\nVector Analysis:")
            print(f"  Stored vector norm: {result['stored_vec_norm']:.6f}")
            print(f"  Stored vector is normalized: {result['stored_vec_normalized']}")
            print(f"\nDistance/Similarity Metrics:")
            print(f"  Database L2 distance: {db_l2_distance:.6f}")
            print(f"  Database similarity (1/(1+L2)): {db_similarity:.6f}")
            print(f"  Our L2 distance: {our_l2_dist:.6f}")
            print(f"  Our cosine similarity: {our_cosine_sim:.6f}")
            print(f"  Difference in L2: {abs(db_l2_distance - our_l2_dist):.6f}")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    
    headers = ['#', 'File', 'Lines', 'DB L2', 'DB Sim', 'Our L2', 'Cosine Sim', 'Vec Normalized']
    table_data = []
    
    for r in results:
        table_data.append([
            r['rank'],
            r['file'][-30:],  # Last 30 chars of filename
            r['lines'],
            f"{r['db_l2_dist']:.4f}",
            f"{r['db_similarity']:.4f}",
            f"{r['our_l2_dist']:.4f}",
            f"{r['our_cosine_sim']:.4f}",
            r['stored_vec_normalized']
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Check if vectors are normalized
    print("\n" + "=" * 80)
    print("NORMALIZATION CHECK")
    print("=" * 80)
    
    # Sample more vectors to check normalization
    cursor.execute("""
        SELECT v.embedding
        FROM vec_embeddings v
        LIMIT 100
    """)
    
    norms = []
    for row in cursor.fetchall():
        embedding = deserialize_embedding(row['embedding'])
        norms.append(np.linalg.norm(embedding))
    
    norms = np.array(norms)
    print(f"\nVector norms statistics (sample of 100):")
    print(f"  Mean: {np.mean(norms):.6f}")
    print(f"  Std: {np.std(norms):.6f}")
    print(f"  Min: {np.min(norms):.6f}")
    print(f"  Max: {np.max(norms):.6f}")
    print(f"  All normalized (within 1e-5): {np.all(np.abs(norms - 1.0) < 1e-5)}")
    
    # Analysis of the scoring issue
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("\n1. The database is using L2 (Euclidean) distance, not cosine distance.")
    print("2. Even though vectors are normalized, L2 distance != cosine distance.")
    print("3. For normalized vectors: cosine_distance = L2_distance^2 / 2")
    print("4. The similarity conversion (1/(1+L2)) is arbitrary and doesn't reflect true cosine similarity.")
    print("\nRecommendation: If you want true cosine similarity ranking, either:")
    print("  a) Use a vector database that supports cosine distance natively")
    print("  b) Store dot products directly (for normalized vectors, dot product = cosine similarity)")
    print("  c) Convert the L2 distances to cosine similarities for ranking")
    
    conn.close()


if __name__ == "__main__":
    main()