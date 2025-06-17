#!/usr/bin/env python3
"""Test script for GGUF model integration."""

import sys
from pathlib import Path

# Test 1: Check if llama-cpp-python is installed
print("Testing GGUF integration for RASSDB...")
print("-" * 60)

try:
    import llama_cpp

    print("✓ llama-cpp-python is installed")
except ImportError:
    print("✗ llama-cpp-python is NOT installed")
    print("  Install with: pip install llama-cpp-python")
    sys.exit(1)

# Test 2: Check if GGUF model file exists
hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
model_dir = hf_cache / "models--nomic-ai--nomic-embed-code" / "blobs"
gguf_path = model_dir / "nomic-embed-code.Q8_0.gguf"

if gguf_path.exists():
    print(f"✓ GGUF model found at: {gguf_path}")
    print(f"  File size: {gguf_path.stat().st_size / 1024**3:.2f} GB")
else:
    print(f"✗ GGUF model NOT found at: {gguf_path}")
    print("  Run download_nomic_code_gguf.py to download it")
    sys.exit(1)

# Test 3: Try to load the model
print("\nTesting model loading...")
try:
    from rassdb.gguf_embeddings import NomicEmbedCodeGGUF

    model = NomicEmbedCodeGGUF()
    print("✓ GGUF model loaded successfully")

    # Test 4: Generate a test embedding
    print("\nTesting embedding generation...")
    test_code = """def hello_world():
    print("Hello, World!")
"""

    embedding = model.encode(test_code)
    print(f"✓ Generated embedding with shape: {embedding.shape}")
    print(f"  Embedding dimension: {embedding.shape[1]}")
    print(f"  First few values: {embedding[0][:5]}")

    # Test 5: Test query encoding
    print("\nTesting query encoding...")
    query = "function that prints hello world"
    query_embedding = model.encode_queries(query)
    print(f"✓ Generated query embedding with shape: {query_embedding.shape}")

    # Calculate similarity
    import numpy as np

    similarity = np.dot(embedding[0], query_embedding[0])
    print(f"  Similarity between code and query: {similarity:.4f}")

except Exception as e:
    print(f"✗ Error loading or using model: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 6: Test embedding strategy
print("\nTesting embedding strategy...")
try:
    from rassdb.embedding_strategies import get_embedding_strategy

    strategy = get_embedding_strategy("nomic-ai/nomic-embed-code-gguf")
    print(f"✓ Got embedding strategy: {strategy.__class__.__name__}")

    # Test the prepare_query method
    prepared_query = strategy.prepare_query("find hello world function")
    print(f"  Prepared query: '{prepared_query}'")

except Exception as e:
    print(f"✗ Error with embedding strategy: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! GGUF integration is working correctly.")
print("\nTo use this model by default, add to ~/.rassdb-config.toml:")
print("[embedding-model]")
print('name = "nomic-ai/nomic-embed-code-gguf"')
