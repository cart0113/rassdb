#!/usr/bin/env python-main
"""Test the new nomic-embed-text-v1.5 model"""

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the new model
model_name = "nomic-ai/nomic-embed-text-v1.5"
print(f"Loading {model_name}...")
model = SentenceTransformer(model_name, trust_remote_code=True)

# The query
query = "How does the javascript call out to the python rag server?"

# The actual relevant code
code_with_metadata = """# Function: post
# Type: function
# File: server.js

app.post('/api/search', async (req, res) => {
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

# Also test without metadata
code_without_metadata = """app.post('/api/search', async (req, res) => {
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

print("\nEncoding query and code...")
query_embedding = model.encode(query, normalize_embeddings=True)
code_with_meta_embedding = model.encode(code_with_metadata, normalize_embeddings=True)
code_without_meta_embedding = model.encode(code_without_metadata, normalize_embeddings=True)

# Calculate similarities
sim_with_meta = cosine_similarity([query_embedding], [code_with_meta_embedding])[0][0]
sim_without_meta = cosine_similarity([query_embedding], [code_without_meta_embedding])[0][0]

print(f"\nQuery: '{query}'")
print(f"\nCosine similarity with metadata:    {sim_with_meta:.4f}")
print(f"Cosine similarity without metadata: {sim_without_meta:.4f}")
print(f"\nDifference: {sim_with_meta - sim_without_meta:.4f}")

# Test with other queries
print("\n\nTesting various queries:")
test_queries = [
    "javascript spawn python process",
    "node.js call python script",
    "express endpoint execute python",
    "How do I run Python from JavaScript?",
    "RASSDB search API"
]

for test_query in test_queries:
    test_emb = model.encode(test_query, normalize_embeddings=True)
    sim = cosine_similarity([test_emb], [code_with_meta_embedding])[0][0]
    print(f"'{test_query}': {sim:.4f}")

print("\n\nComparing with CodeRankEmbed scores:")
print("CodeRankEmbed best score was 0.6218 for 'spawn RASSDB_SEARCH'")
print(f"Nomic-embed-text-v1.5 score: {sim_with_meta:.4f} for the original query")