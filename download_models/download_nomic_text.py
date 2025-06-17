#!/usr/bin/env python3
"""Download Nomic Embed Text v1.5 model (nomic-ai/nomic-embed-text-v1.5)."""

from sentence_transformers import SentenceTransformer


def download_nomic_text():
    """Download and test Nomic Embed Text model."""
    model_name = "nomic-ai/nomic-embed-text-v1.5"
    
    print(f"Downloading: {model_name}")
    print("=" * 60)
    
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
    success = download_nomic_text()
    if success:
        print("\nNomic Embed Text v1.5 downloaded successfully!")
    else:
        print("\nFailed to download Nomic Embed Text v1.5.")
        exit(1)