# Embedding Server Setup

This directory contains scripts for setting up and managing the embedding model used by RASSDB.

## Downloading the Model

To download the embedding model:

```bash
python-main download_model.py
```

This will download the `nomic-ai/nomic-embed-text-v1.5` model to your local cache.

To use a different model:

```bash
python-main download_model.py --model "your-model-name"
```

## Model Information

- **Default Model**: nomic-ai/nomic-embed-text-v1.5
- **Embedding Dimension**: 768
- **Cache Location**: `~/.cache/torch/sentence_transformers/`

## Notes

The embedding model is loaded on-demand when indexing or searching. You don't need to run a separate embedding server - the model is embedded directly in the RASSDB processes.