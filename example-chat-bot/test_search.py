#!/usr/bin/env python-main
"""Test the actual search functionality"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3

# The query
query = "How does the javascript call out to the python rag server?"

# Load model
model_name = "nomic-ai/CodeRankEmbed"
print(f"Loading model: {model_name}")
model = SentenceTransformer(model_name, trust_remote_code=True)

# Test embeddings for different queries
test_queries = [
    query,
    f"Represent this query for searching relevant code: {query}",
    "spawn RASSDB python",
    "Represent this query for searching relevant code: spawn RASSDB python",
    "javascript spawn python process"
]

# Also test some code snippets
test_codes = [
    "const pythonProcess = spawn(RASSDB_SEARCH, [query, '--semantic'])",
    "app.post('/api/search', async (req, res) => {",
    "// Call RASSDB search",
    """pythonProcess.stdout.on('data', (data) => {
    output += data.toString();
});"""
]

print("\nQuery embeddings:")
query_embeddings = []
for q in test_queries:
    emb = model.encode(q, normalize_embeddings=True)
    query_embeddings.append(emb)
    print(f"  '{q[:50]}...' -> norm: {np.linalg.norm(emb):.4f}")

print("\nCode embeddings:")
code_embeddings = []
for c in test_codes:
    emb = model.encode(c, normalize_embeddings=True)
    code_embeddings.append(emb)
    print(f"  '{c[:50]}...' -> norm: {np.linalg.norm(emb):.4f}")

print("\nSimilarity matrix (queries vs codes):")
print("          ", end="")
for i in range(len(test_codes)):
    print(f"Code{i:<8}", end="")
print()

for i, q_emb in enumerate(query_embeddings):
    print(f"Query{i}: ", end="")
    for c_emb in code_embeddings:
        sim = cosine_similarity([q_emb], [c_emb])[0][0]
        print(f"{sim:8.4f}", end="")
    print()

# Now check what's actually in the database
print("\n\nChecking database contents...")
db_path = ".rassdb/example-chat-bot-CodeRankEmbed.rassdb"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Find relevant chunks
cursor.execute("""
    SELECT id, file_path, start_line, end_line, content
    FROM code_chunks
    WHERE file_path LIKE '%server.js%'
    AND (content LIKE '%spawn%' OR content LIKE '%RASSDB%' OR content LIKE '%api/search%')
    ORDER BY start_line
    LIMIT 10
""")

relevant_chunks = cursor.fetchall()
print(f"\nFound {len(relevant_chunks)} relevant chunks in server.js")

# Check if they have embeddings
for chunk in relevant_chunks:
    chunk_id = chunk[0]
    cursor.execute("SELECT COUNT(*) FROM vec_embeddings WHERE rowid = ?", (chunk_id,))
    has_embedding = cursor.fetchone()[0] > 0
    print(f"Chunk {chunk_id} ({chunk[1]}:{chunk[2]}-{chunk[3]}): has_embedding={has_embedding}")
    if not has_embedding:
        print(f"  Content: {chunk[4][:100]}...")

conn.close()