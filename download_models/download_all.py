#!/usr/bin/env python3
"""Download all RASSDB embedding models."""

import subprocess
import sys
from pathlib import Path


STANDARD_MODELS = [
    ("CodeBERT", "download_codebert.py"),
    ("Qodo-Embed-1.5B (~6.2GB)", "download_qodo_embed.py"),
    ("CodeRankEmbed", "download_coderank_embed.py"),
    ("Nomic Embed Text v1.5", "download_nomic_text.py"),
]

LARGE_MODELS = [
    ("Nomic Embed Code (~14GB)", "download_nomic_code.py"),
]


def run_download_script(script_name):
    """Run a download script and return success status."""
    script_path = Path(__file__).parent / script_name
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        return False


def main():
    """Download all configured models."""
    print("RASSDB Model Downloader")
    print("=" * 60)
    print("This will download the following models:")
    for name, _ in STANDARD_MODELS:
        print(f"  - {name}")
    print("\nLarge models (optional):")
    for name, _ in LARGE_MODELS:
        print(f"  - {name}")
    
    response = input("\nProceed with standard models download? (y/N): ")
    if response.lower() != "y":
        print("Download cancelled.")
        return
    
    download_large = False
    response = input("\nAlso download large models? (y/N): ")
    if response.lower() == "y":
        download_large = True
    
    # Download standard models
    success_count = 0
    total_models = STANDARD_MODELS[:]
    if download_large:
        total_models.extend(LARGE_MODELS)
    
    for name, script in total_models:
        print(f"\n{'='*60}")
        print(f"Downloading {name}...")
        print("=" * 60)
        if run_download_script(script):
            success_count += 1
        else:
            print(f"Failed to download {name}")
    
    print(f"\n{'='*60}")
    print(f"Download complete! {success_count}/{len(total_models)} models downloaded successfully.")
    
    if success_count < len(total_models):
        print("\nSome models failed to download. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()