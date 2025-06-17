"""Cloud-based embedding models for RASSDB.

This module provides wrapper classes for cloud-based embedding services
that implement the same interface as local SentenceTransformer models.
"""

import os
import logging
import requests
import numpy as np
from typing import List, Union, Optional, Dict, Any
import numpy.typing as npt

logger = logging.getLogger(__name__)


class NomicCloudEmbedding:
    """Wrapper for Nomic Cloud API embeddings.
    
    This class implements the same interface as SentenceTransformer
    but uses the Nomic Cloud API for embedding generation.
    """
    
    def __init__(self, model_name: str = "nomic-embed-code", trust_remote_code: bool = True):
        """Initialize Nomic Cloud embedding model.
        
        Args:
            model_name: The model to use (currently only supports variations of nomic-embed-code)
            trust_remote_code: Included for compatibility with SentenceTransformer
        """
        self.api_key = os.environ.get("NOMIC_API_KEY")
        if not self.api_key:
            raise ValueError("NOMIC_API_KEY environment variable not set")
            
        self.model_name = model_name
        self.api_url = "https://api-atlas.nomic.ai/v1/embedding/text"
        self._embedding_dim = 768  # Default for nomic-embed-code
        
        # Validate we can reach the API
        self._validate_api_access()
        
    def _validate_api_access(self):
        """Validate API key and access."""
        try:
            # Test with a simple request
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "texts": ["test"],
                    "model": "nomic-embed-text-v1.5",
                    "task_type": "search_document"
                },
                timeout=10
            )
            
            if response.status_code == 401:
                raise ValueError("Invalid NOMIC_API_KEY - authentication failed")
            elif response.status_code != 200:
                raise ValueError(f"Nomic API error: {response.status_code} - {response.text}")
                
            # Get actual embedding dimension from response
            data = response.json()
            if "embeddings" in data and len(data["embeddings"]) > 0:
                self._embedding_dim = len(data["embeddings"][0])
                
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to connect to Nomic API: {e}")
    
    def encode(
        self, 
        sentences: Union[str, List[str]], 
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        task_type: str = "search_document"
    ) -> npt.NDArray[np.float32]:
        """Generate embeddings for sentences using Nomic Cloud API.
        
        Args:
            sentences: Single sentence or list of sentences to embed
            normalize_embeddings: Whether to normalize the embeddings
            batch_size: Batch size for API requests (max 100 for Nomic)
            show_progress_bar: Ignored for cloud API
            task_type: Task type for embedding (search_document, search_query, etc.)
            
        Returns:
            Numpy array of embeddings
        """
        # Convert single string to list
        if isinstance(sentences, str):
            sentences = [sentences]
            
        # Nomic API has a max batch size
        batch_size = min(batch_size, 100)
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            # Prepare request
            request_data = {
                "texts": batch,
                "model": "nomic-embed-code",
                "task_type": task_type,
                "long_text_mode": "truncate",
                "max_tokens_per_text": 8192
            }
            
            # Make API request
            try:
                response = requests.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=request_data,
                    timeout=30
                )
                
                if response.status_code != 200:
                    raise RuntimeError(f"Nomic API error: {response.status_code} - {response.text}")
                    
                data = response.json()
                embeddings = data["embeddings"]
                
                # Log token usage if available
                if "usage" in data:
                    logger.debug(f"Nomic API usage: {data['usage']}")
                    
                all_embeddings.extend(embeddings)
                
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Failed to generate embeddings: {e}")
        
        # Convert to numpy array
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        
        # Normalize if requested
        if normalize_embeddings:
            # L2 normalization
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            embeddings_array = embeddings_array / np.maximum(norms, 1e-10)
            
        return embeddings_array
    
    def encode_queries(self, queries: Union[str, List[str]], **kwargs) -> npt.NDArray[np.float32]:
        """Encode queries with search_query task type."""
        return self.encode(queries, task_type="search_query", **kwargs)
    
    def encode_corpus(self, corpus: Union[str, List[str]], **kwargs) -> npt.NDArray[np.float32]:
        """Encode corpus/documents with search_document task type."""
        return self.encode(corpus, task_type="search_document", **kwargs)
        
    def get_sentence_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self._embedding_dim


def get_cloud_embedding_model(model_name: str) -> Union[NomicCloudEmbedding, None]:
    """Factory function to get appropriate cloud embedding model.
    
    Args:
        model_name: The model identifier
        
    Returns:
        Cloud embedding model instance or None if not a cloud model
    """
    if model_name == "nomic-cloud/nomic-embed-code":
        return NomicCloudEmbedding(model_name="nomic-embed-code")
    
    return None