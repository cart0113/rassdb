#!/usr/bin/env python3
"""Download Nomic Embed Code GGUF model (8-bit quantized version)."""

import os
import requests
from pathlib import Path


def download_nomic_code_gguf():
    """Download the 8-bit quantized GGUF version of Nomic Embed Code."""
    repo_id = "nomic-ai/nomic-embed-code-GGUF"
    filename = "nomic-embed-code.Q8_0.gguf"
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"

    # Use HuggingFace cache directory
    hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = hf_cache_dir / "models--nomic-ai--nomic-embed-code" / "blobs"
    model_dir.mkdir(parents=True, exist_ok=True)

    output_path = model_dir / filename

    print(f"Downloading: {repo_id}")
    print(f"File: {filename}")
    print("=" * 60)
    print("This is the 8-bit quantized version (~7GB vs ~26GB for full precision)")
    print("URL:", url)

    try:
        # Download with streaming to show progress
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 8192
        downloaded = 0

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(
                            f"\rProgress: {percent:.1f}% ({downloaded / 1024**3:.2f} GB / {total_size / 1024**3:.2f} GB)",
                            end="",
                        )

        print(f"\n✓ Model downloaded to: {output_path}")
        print(f"✓ File size: {output_path.stat().st_size / 1024**3:.2f} GB")
        return True

    except requests.exceptions.RequestException as e:
        print(f"\n✗ Error downloading model: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    response = input(
        "\nThis will download the 8-bit quantized GGUF model (~7GB). Proceed? (y/N): "
    )
    if response.lower() != "y":
        print("Download cancelled.")
        exit(0)

    success = download_nomic_code_gguf()
    if success:
        print("\nNomic Embed Code GGUF downloaded successfully!")
        print(
            "\nNote: To use GGUF models, you'll need a compatible inference engine like llama.cpp or similar."
        )
    else:
        print("\nFailed to download Nomic Embed Code GGUF.")
        exit(1)
