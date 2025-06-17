#!/usr/bin/env python-main
"""Final test to understand the embedding behavior"""

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the model
model_name = "nomic-ai/CodeRankEmbed"
print(f"Loading {model_name}...")
model = SentenceTransformer(model_name, trust_remote_code=True)

# The actual code from chunk ID 28
actual_code = """app.post('/api/search', async (req, res) => {
    const { query, limit = 5 } = req.body;
    
    try {
        // Call RASSDB search
        const pythonProcess = spawn(RASSDB_SEARCH, [
            query,
            '--semantic',
            '--limit', limit.toString(),
            '--db', path.join(__dirname, '.rassdb', 'example-chat-bot-nomic-embed-text-v1.5.rassdb'),
            '--format', 'json'
        ]);"""

# Various queries to test
queries = [
    # Original query
    "How does the javascript call out to the python rag server?",
    
    # More specific variations
    "javascript spawn python process",
    "spawn RASSDB_SEARCH",
    "node.js spawn python subprocess",
    "express endpoint spawn python",
    
    # Different phrasings
    "how to execute python from javascript",
    "javascript execute python script",
    "node spawn child process python",
    
    # Very specific
    "app.post spawn pythonProcess",
    "const pythonProcess = spawn"
]

print("\nEncoding code chunk...")
code_embedding = model.encode(actual_code, normalize_embeddings=True)

print("\nTesting different queries:")
print("-" * 80)

best_score = 0
best_query = ""

for query in queries:
    # Test with and without the prefix
    query_with_prefix = f"Represent this query for searching relevant code: {query}"
    
    # Encode both versions
    emb_no_prefix = model.encode(query, normalize_embeddings=True)
    emb_with_prefix = model.encode(query_with_prefix, normalize_embeddings=True)
    
    # Calculate similarities
    sim_no_prefix = cosine_similarity([emb_no_prefix], [code_embedding])[0][0]
    sim_with_prefix = cosine_similarity([emb_with_prefix], [code_embedding])[0][0]
    
    print(f"\nQuery: {query}")
    print(f"  Without prefix: {sim_no_prefix:.4f}")
    print(f"  With prefix:    {sim_with_prefix:.4f}")
    
    if sim_with_prefix > best_score:
        best_score = sim_with_prefix
        best_query = query

print("\n" + "=" * 80)
print(f"Best query: '{best_query}' with score: {best_score:.4f}")

# Also test what kind of queries would get high scores
print("\n\nTesting code-like queries:")
test_codes = [
    "spawn(RASSDB_SEARCH",
    "pythonProcess = spawn",
    "app.post('/api/search'",
    "async (req, res) =>",
    "path.join(__dirname, '.rassdb')"
]

for test in test_codes:
    test_emb = model.encode(test, normalize_embeddings=True)
    sim = cosine_similarity([test_emb], [code_embedding])[0][0]
    print(f"'{test}': {sim:.4f}")