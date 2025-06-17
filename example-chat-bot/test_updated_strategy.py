#!/usr/bin/env python-main
"""Test the updated nomic-embed-text-v1.5 strategy with task prefixes"""

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the model
model_name = "nomic-ai/nomic-embed-text-v1.5"
print(f"Loading {model_name}...")
model = SentenceTransformer(model_name, trust_remote_code=True)

# The query with proper prefix
query = "search_query: How does the javascript call out to the python rag server?"

# Different code formatting approaches
code_examples = {
    "With task prefix and context": """search_document: # function post

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
        ]);""",
    
    "With task prefix only": """search_document: app.post('/api/search', async (req, res) => {
    const { query, limit = 5 } = req.body;
    
    try {
        // Call RASSDB search
        const pythonProcess = spawn(RASSDB_SEARCH, [
            query,
            '--semantic',
            '--limit', limit.toString(),
            '--db', path.join(__dirname, '.rassdb', 'example-chat-bot-nomic-embed-text-v1.5.rassdb'),
            '--format', 'json'
        ]);""",
    
    "Without any prefix": """app.post('/api/search', async (req, res) => {
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
}

print("\nEncoding query and code examples...")
query_embedding = model.encode(query, normalize_embeddings=True)

print(f"\nQuery: '{query}'")
print("\nSimilarity scores:")
print("-" * 60)

for name, code in code_examples.items():
    code_embedding = model.encode(code, normalize_embeddings=True)
    similarity = cosine_similarity([query_embedding], [code_embedding])[0][0]
    print(f"{name}: {similarity:.4f}")

# Test with different queries
print("\n\nTesting various queries (all with search_query prefix):")
test_queries = [
    "search_query: javascript spawn python process",
    "search_query: node.js call python script",
    "search_query: express endpoint execute python",
    "search_query: How do I run Python from JavaScript?",
    "search_query: RASSDB search API"
]

# Use the best code format
best_code = code_examples["With task prefix and context"]

for test_query in test_queries:
    test_emb = model.encode(test_query, normalize_embeddings=True)
    sim = cosine_similarity([test_emb], [model.encode(best_code, normalize_embeddings=True)])[0][0]
    print(f"'{test_query[14:]}': {sim:.4f}")  # Skip the "search_query: " prefix in display