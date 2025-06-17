#!/usr/bin/env python-main
"""Check the exact schema of the database"""

import sqlite3

db_path = ".rassdb/example-chat-bot-CodeRankEmbed.rassdb"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check vec_embeddings_info schema
print("Schema for vec_embeddings_info:")
cursor.execute("PRAGMA table_info(vec_embeddings_info)")
for col in cursor.fetchall():
    print(f"  {col}")

# Get the actual data
print("\nData in vec_embeddings_info:")
cursor.execute("SELECT * FROM vec_embeddings_info")
for row in cursor.fetchall():
    print(f"  {row}")

conn.close()