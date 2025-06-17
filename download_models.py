#!/usr/bin/env python3
"""Download required embedding models for RASSDB.

This script downloads the 3 supported code embedding models:
- CodeBERT (microsoft/codebert-base)
- Qodo-Embed-1.5B (Qodo/Qodo-Embed-1-1.5B)
- CodeRankEmbed (nomic-ai/CodeRankEmbed)

Note: Nomic Embed Code is not downloaded due to size constraints.
"""

import sys
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel


def download_model(model_name):
    """Download and test a model."""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print("=" * 60)

    try:
        if model_name == "microsoft/codebert-base":
            # CodeBERT needs special handling
            print("Downloading CodeBERT tokenizer and model...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            print(f"✓ CodeBERT downloaded! Hidden size: {model.config.hidden_size}")
        else:
            # Sentence transformer models
            model = SentenceTransformer(model_name, trust_remote_code=True)
            dim = model.get_sentence_embedding_dimension()
            print(f"✓ Model downloaded! Embedding dimension: {dim}")

            # Test encoding
            test_text = "def hello(): return 'world'"
            embedding = model.encode(test_text)
            print(f"✓ Test encoding successful! Shape: {embedding.shape}")

    except Exception as e:
        print(f"✗ Error downloading {model_name}: {e}")
        return False

    return True


def main():
    """Download all configured models."""
    models = [
        "microsoft/codebert-base",  # CodeBERT (requires special handling)
        "Qodo/Qodo-Embed-1-1.5B",  # State-of-the-art code embeddings (6.2GB)
        "nomic-ai/CodeRankEmbed",  # Code ranking embeddings
    ]

    print("RASSDB Model Downloader")
    print("=" * 60)
    print("This will download the following 3 models:")
    for model in models:
        print(f"  - {model}")
    print("\nNote: Nomic Embed Code (nomic-ai/nomic-embed-code) is not downloaded")
    print("due to size constraints but is still supported by RASSDB.")
    print("\nQodo-Embed-1-1.5B is ~6.2GB and may take a while to download.")

    response = input("\nProceed with download? (y/N): ")
    if response.lower() != "y":
        print("Download cancelled.")
        return

    success_count = 0
    for model_name in models:
        if download_model(model_name):
            success_count += 1

    print(f"\n{'='*60}")
    print(
        f"Download complete! {success_count}/{len(models)} models downloaded successfully."
    )

    if success_count < len(models):
        print("\nSome models failed to download. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
