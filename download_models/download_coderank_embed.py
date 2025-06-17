#!/usr/bin/env python3
"""Download CodeRankEmbed model (nomic-ai/CodeRankEmbed)."""

from sentence_transformers import SentenceTransformer


def download_coderank_embed():
    """Download and test CodeRankEmbed model."""
    model_name = "nomic-ai/CodeRankEmbed"
    
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
    success = download_coderank_embed()
    if success:
        print("\nCodeRankEmbed downloaded successfully!")
    else:
        print("\nFailed to download CodeRankEmbed.")
        exit(1)