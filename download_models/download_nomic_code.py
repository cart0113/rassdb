#!/usr/bin/env python3
"""Download Nomic Embed Code model (nomic-ai/nomic-embed-code) - Large model."""

from sentence_transformers import SentenceTransformer


def download_nomic_code():
    """Download and test Nomic Embed Code model."""
    model_name = "nomic-ai/nomic-embed-code"
    
    print(f"Downloading: {model_name}")
    print("=" * 60)
    print("WARNING: This is a 7B parameter model (~14GB). Ensure you have enough disk space.")
    
    try:
        model = SentenceTransformer(model_name, trust_remote_code=True)
        dim = model.get_sentence_embedding_dimension()
        print(f"✓ Model downloaded! Embedding dimension: {dim}")
        
        # Test encoding
        test_text = "def hello(): return 'world'"
        embedding = model.encode(test_text)
        print(f"✓ Test encoding successful! Shape: {embedding.shape}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {model_name}: {e}")
        return False


if __name__ == "__main__":
    response = input("\nThis will download a ~14GB model. Proceed? (y/N): ")
    if response.lower() != "y":
        print("Download cancelled.")
        exit(0)
    
    success = download_nomic_code()
    if success:
        print("\nNomic Embed Code downloaded successfully!")
    else:
        print("\nFailed to download Nomic Embed Code.")
        exit(1)