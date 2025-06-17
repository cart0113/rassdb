#!/usr/bin/env python-main
"""Debug script to test embeddings and cosine similarity"""

import sqlite3

# Database path - using the one from command line (CodeRankEmbed)
db_path = ".rassdb/example-chat-bot-CodeRankEmbed.rassdb"

print(f"Opening database: {db_path}")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# First, let's see what tables exist
print("\nDatabase tables:")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
for table in tables:
    print(f"  - {table[0]}")

# Check the schema of each table
for table in tables:
    table_name = table[0]
    print(f"\nSchema for table '{table_name}':")
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")

conn.close()