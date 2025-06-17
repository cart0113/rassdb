"""GGUF model support for RASSDB using llama-cpp-python.

This module provides wrapper classes for GGUF embedding models
that implement the same interface as SentenceTransformer models.
"""

import os
import logging
import numpy as np
from typing import List, Union, Optional
from pathlib import Path
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Try to import llama-cpp-python
try:
    from llama_cpp import Llama

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not installed. GGUF models will not be available.")


class GGUFEmbedding:
    """Wrapper for GGUF embedding models using llama-cpp-python.

    This class implements the same interface as SentenceTransformer
    but uses GGUF models via llama-cpp-python.
    """

    def __init__(self, model_path: str, n_ctx: int = 8192, n_gpu_layers: int = -1):
        """Initialize GGUF embedding model.

        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context window size (default: 8192 for nomic models)
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is required for GGUF models. "
                "Install with: pip install llama-cpp-python"
            )

        self.model_path = model_path
        self.n_ctx = n_ctx

        # Initialize the model
        logger.info(f"Loading GGUF model from: {model_path}")

        # Import pooling type for proper embedding mode
        try:
            from llama_cpp import LLAMA_POOLING_TYPE_MEAN

            pooling_type = LLAMA_POOLING_TYPE_MEAN
        except ImportError:
            # Fallback for older versions
            pooling_type = 1  # MEAN pooling

        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            embedding=True,  # Enable embedding mode
            pooling_type=pooling_type,  # Use mean pooling for embeddings
            n_batch=512,  # Increase batch size
            verbose=False,
        )
        logger.info("✓ GGUF model loaded successfully")

        # Get actual embedding dimension from model
        try:
            test_embedding = self.model.embed("test")
            self._embedding_dim = len(test_embedding)
            logger.info(f"  Embedding dimension: {self._embedding_dim}")
        except RuntimeError as e:
            if "llama_decode returned -3" in str(e):
                logger.warning(
                    "Initial embedding test failed, likely due to model initialization. Assuming 3584 dimensions."
                )
                self._embedding_dim = 3584
            else:
                raise

    def encode(
        self,
        sentences: Union[str, List[str]],
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> npt.NDArray[np.float32]:
        """Generate embeddings for sentences using GGUF model.

        Args:
            sentences: Single sentence or list of sentences to embed
            normalize_embeddings: Whether to normalize the embeddings
            batch_size: Batch size (not used, kept for compatibility)
            show_progress_bar: Not implemented for GGUF
            convert_to_numpy: Whether to return numpy array (always True)

        Returns:
            Numpy array of embeddings
        """
        # Convert single string to list
        if isinstance(sentences, str):
            sentences = [sentences]

        all_embeddings = []

        # Process each sentence
        for sentence in sentences:
            try:
                # Get embedding from llama.cpp
                embedding = self.model.embed(sentence)
                all_embeddings.append(embedding)
            except RuntimeError as e:
                if "llama_decode returned -3" in str(e):
                    # Try with truncated text if context is full
                    logger.warning(f"Text too long, truncating to fit context window")
                    # Rough estimate: ~3 chars per token
                    max_chars = (self.n_ctx - 100) * 3
                    truncated = sentence[:max_chars]
                    embedding = self.model.embed(truncated)
                    all_embeddings.append(embedding)
                else:
                    raise

        # Convert to numpy array
        embeddings_array = np.array(all_embeddings, dtype=np.float32)

        # Normalize if requested
        if normalize_embeddings:
            # L2 normalization
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            embeddings_array = embeddings_array / np.maximum(norms, 1e-10)

        return embeddings_array

    def encode_queries(
        self, queries: Union[str, List[str]], **kwargs
    ) -> npt.NDArray[np.float32]:
        """Encode queries - for GGUF models, same as encode."""
        return self.encode(queries, **kwargs)

    def encode_corpus(
        self, corpus: Union[str, List[str]], **kwargs
    ) -> npt.NDArray[np.float32]:
        """Encode corpus/documents - for GGUF models, same as encode."""
        return self.encode(corpus, **kwargs)

    def get_sentence_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self._embedding_dim


class NomicEmbedCodeGGUF(GGUFEmbedding):
    """Specialized wrapper for nomic-embed-code GGUF model."""

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-code-gguf",
        trust_remote_code: bool = True,
    ):
        """Initialize Nomic Embed Code GGUF model.

        Args:
            model_name: Model identifier (used for strategy selection)
            trust_remote_code: Included for compatibility
        """
        # Find the GGUF file in HuggingFace cache
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        model_dir = hf_cache / "models--nomic-ai--nomic-embed-code" / "blobs"

        # Look for the Q8_0 GGUF file
        gguf_path = model_dir / "nomic-embed-code.Q8_0.gguf"

        if not gguf_path.exists():
            # Try to find any GGUF file in the directory
            gguf_files = list(model_dir.glob("*.gguf"))
            if gguf_files:
                gguf_path = gguf_files[0]
                logger.info(f"Found GGUF file: {gguf_path.name}")
            else:
                raise FileNotFoundError(
                    f"No GGUF model found in {model_dir}. "
                    "Please run download_nomic_code_gguf.py first."
                )

        # Initialize with parameters suitable for this Qwen2-based model
        # Use smaller context to avoid KV cache issues
        super().__init__(
            model_path=str(gguf_path),
            n_ctx=2048,  # Start with smaller context to avoid errors
            n_gpu_layers=-1,  # Use GPU if available
        )

        self.model_name = model_name
        # Embedding dimension will be set by parent class from model


def get_gguf_embedding_model(model_name: str) -> Optional[GGUFEmbedding]:
    """Factory function to get appropriate GGUF embedding model.

    Args:
        model_name: The model identifier

    Returns:
        GGUF embedding model instance or None if not a GGUF model
    """
    if model_name in ["nomic-ai/nomic-embed-code-gguf", "nomic-embed-code-gguf"]:
        return NomicEmbedCodeGGUF(model_name)

    return None
