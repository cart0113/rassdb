#!/usr/bin/env python3
"""Download Qodo-Embed-1.5B model (Qodo/Qodo-Embed-1-1.5B)."""

from sentence_transformers import SentenceTransformer


def download_qodo_embed():
    """Download and test Qodo-Embed model."""
    model_name = "Qodo/Qodo-Embed-1-1.5B"
    
    print(f"Downloading: {model_name}")
    print("=" * 60)
    print("Note: This model is ~6.2GB and may take a while to download.")
    
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
    success = download_qodo_embed()
    if success:
        print("\nQodo-Embed downloaded successfully!")
    else:
        print("\nFailed to download Qodo-Embed.")
        exit(1)