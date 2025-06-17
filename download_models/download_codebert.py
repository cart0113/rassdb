#!/usr/bin/env python3
"""Download CodeBERT model (microsoft/codebert-base)."""

from transformers import AutoTokenizer, AutoModel


def download_codebert():
    """Download and test CodeBERT model."""
    model_name = "microsoft/codebert-base"
    
    print(f"Downloading: {model_name}")
    print("=" * 60)
    
    try:
        print("Downloading CodeBERT tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        print(f"✓ CodeBERT downloaded! Hidden size: {model.config.hidden_size}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {model_name}: {e}")
        return False


if __name__ == "__main__":
    success = download_codebert()
    if success:
        print("\nCodeBERT downloaded successfully!")
    else:
        print("\nFailed to download CodeBERT.")
        exit(1)