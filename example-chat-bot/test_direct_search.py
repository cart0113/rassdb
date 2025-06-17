#!/usr/bin/env python-main
"""Test search by directly using the CLI"""

import subprocess
import json

query = "How does the javascript call out to the python rag server?"

# Run the search command
cmd = [
    "../bin/rassdb-search",
    query,
    "--semantic",
    "--limit", "10",
    "--db", ".rassdb/example-chat-bot-CodeRankEmbed.rassdb",
    "--format", "json"
]

print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)

print("\nSTDOUT:")
print(result.stdout)

print("\nSTDERR:")
print(result.stderr)

print("\nReturn code:", result.returncode)