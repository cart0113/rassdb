#!/usr/bin/env python-main
"""Debug script to test embeddings and cosine similarity"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sentence_transformers import SentenceTransformer
from rassdb.vector_store import VectorStore
from rassdb.embedding_strategies import get_embedding_strategy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# The query that's not working
query = "How does the javascript call out to the python rag server?"

# Database path
db_path = ".rassdb/example-chat-bot-CodeRankEmbed.rassdb"

print(f"Opening database: {db_path}")
vector_store = VectorStore(db_path)

# Get model name from DB
conn = vector_store.conn
cursor = conn.cursor()
cursor.execute("SELECT model_name, embedding_dimensions FROM vec_embeddings_info WHERE distance_metric = 'cosine'")
model_info = cursor.fetchone()
model_name = model_info[0] if model_info else "nomic-ai/CodeRankEmbed"
print(f"Model from DB: {model_name}")

# Load model and strategy
print("\nLoading embedding model...")
model = SentenceTransformer(model_name, trust_remote_code=True)
strategy = get_embedding_strategy(model_name)

# Prepare query with proper prefix
query_text = strategy.prepare_query(query)
print(f"\nOriginal query: {query}")
print(f"Prepared query: {query_text}")

# Generate query embedding
query_embedding = model.encode(query_text, normalize_embeddings=True)
print(f"\nQuery embedding shape: {query_embedding.shape}")
print(f"Query embedding norm: {np.linalg.norm(query_embedding)}")
print(f"Query embedding sample: {query_embedding[:10]}")

# Test with some relevant code chunks
test_code = """// Call RASSDB search
const pythonProcess = spawn(RASSDB_SEARCH, [
    query,
    '--semantic',
    '--limit', limit.toString(),
    '--db', path.join(__dirname, '.rassdb', 'example-chat-bot-nomic-embed-text-v1.5.rassdb'),
    '--format', 'json'
]);"""

print(f"\n\nTest code chunk:")
print(test_code)

# Encode the code chunk (no prefix for code)
code_embedding = model.encode(test_code, normalize_embeddings=True)
print(f"\nCode embedding shape: {code_embedding.shape}")
print(f"Code embedding norm: {np.linalg.norm(code_embedding)}")

# Calculate similarity
similarity = cosine_similarity([query_embedding], [code_embedding])[0][0]
print(f"\nCosine similarity: {similarity:.4f}")

# Now let's search the actual database
print("\n\nSearching database...")
results = vector_store.search_similar(query_embedding, limit=10)

print(f"\nFound {len(results)} results:")
for i, result in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"File: {result['file_path']}:{result['start_line']}-{result['end_line']}")
    print(f"Type: {result['chunk_type']}, Language: {result['language']}")
    print(f"Distance: {result['distance']:.4f}, Similarity: {1 - result['distance']:.4f}")
    print(f"Content preview: {result['content'][:200]}...")

# Let's also manually check a specific chunk
print("\n\nManually checking server.js chunks...")
cursor.execute("""
    SELECT id, file_path, start_line, end_line, content 
    FROM code_chunks 
    WHERE file_path LIKE '%server.js%' 
    AND content LIKE '%spawn%RASSDB%'
    LIMIT 3
""")

for row in cursor.fetchall():
    chunk_id, file_path, start_line, end_line, content = row
    print(f"\nChunk ID {chunk_id}: {file_path}:{start_line}-{end_line}")
    print(f"Content preview: {content[:200]}...")
    
    # Get embedding for this chunk
    cursor.execute("SELECT embedding FROM vec_embeddings WHERE rowid = ?", (chunk_id,))
    embedding_data = cursor.fetchone()
    
    if embedding_data:
        # Decode the embedding
        chunk_embedding = np.frombuffer(embedding_data[0], dtype=np.float32)
        print(f"Embedding shape: {chunk_embedding.shape}")
        
        # Calculate similarity
        sim = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
        print(f"Cosine similarity: {sim:.4f}")
    else:
        print("NO EMBEDDING FOUND!")

vector_store.close()