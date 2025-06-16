#!/usr/bin/env python-main
"""Download and setup embedding model for RASSDB.

This script downloads the nomic-embed-text model used for generating
code embeddings in RASSDB.
"""

import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer


def download_embedding_model(model_name: str = "nomic-ai/nomic-embed-text-v1.5") -> bool:
    """Download the embedding model.
    
    Args:
        model_name: Name of the model to download.
        
    Returns:
        True if successful, False otherwise.
    """
    print(f"Downloading {model_name} model...")
    
    try:
        # This will download the model if not already present
        model = SentenceTransformer(model_name, trust_remote_code=True)
        
        # Test the model
        test_text = "def hello_world():\n    print('Hello, World!')"
        embedding = model.encode(test_text)
        
        print(f"✓ Model downloaded successfully!")
        print(f"✓ Model device: {model.device}")
        print(f"✓ Embedding dimension: {len(embedding)}")
        print(f"✓ Test embedding generated successfully")
        
        # Get cache location
        cache_dir = Path.home() / ".cache" / "torch" / "sentence_transformers"
        print(f"✓ Model cached at: {cache_dir}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error downloading model: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download embedding model for RASSDB")
    parser.add_argument(
        "--model",
        default="nomic-ai/nomic-embed-text-v1.5",
        help="Model name to download"
    )
    args = parser.parse_args()
    
    success = download_embedding_model(args.model)
    sys.exit(0 if success else 1)