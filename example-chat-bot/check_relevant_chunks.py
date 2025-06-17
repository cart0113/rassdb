#!/usr/bin/env python-main
"""Check if relevant chunks exist in the database"""

import sqlite3
import json

db_path = ".rassdb/example-chat-bot-CodeRankEmbed.rassdb"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Search for the actual spawn code
print("Searching for chunks with spawn/RASSDB code...")
cursor.execute("""
    SELECT id, file_path, start_line, end_line, chunk_type, content
    FROM code_chunks
    WHERE content LIKE '%spawn%' 
    AND content LIKE '%RASSDB%'
    ORDER BY file_path, start_line
""")

chunks = cursor.fetchall()
print(f"\nFound {len(chunks)} chunks with spawn/RASSDB:")

for chunk in chunks:
    print(f"\nChunk ID: {chunk[0]}")
    print(f"File: {chunk[1]}:{chunk[2]}-{chunk[3]}")
    print(f"Type: {chunk[4]}")
    print(f"Content preview: {chunk[5][:200]}...")

# Specifically look for the RAG search endpoint
print("\n\nSearching for RAG search endpoint...")
cursor.execute("""
    SELECT id, file_path, start_line, end_line, chunk_type, content
    FROM code_chunks
    WHERE file_path LIKE '%server.js%'
    AND (content LIKE '%app.post%/api/search%' OR content LIKE '%RAG search endpoint%')
""")

endpoint_chunks = cursor.fetchall()
print(f"\nFound {len(endpoint_chunks)} endpoint chunks:")

for chunk in endpoint_chunks:
    print(f"\nChunk ID: {chunk[0]}")
    print(f"File: {chunk[1]}:{chunk[2]}-{chunk[3]}")
    print(f"Type: {chunk[4]}")
    print(f"Content: {chunk[5][:300]}...")

# Check what chunks exist for server.js
print("\n\nAll server.js chunks:")
cursor.execute("""
    SELECT id, start_line, end_line, chunk_type, LENGTH(content) as len
    FROM code_chunks
    WHERE file_path LIKE '%server.js%'
    ORDER BY start_line
""")

server_chunks = cursor.fetchall()
print(f"\nTotal server.js chunks: {len(server_chunks)}")
for chunk in server_chunks:
    print(f"  ID {chunk[0]}: lines {chunk[1]}-{chunk[2]} ({chunk[3]}, {chunk[4]} chars)")

conn.close()