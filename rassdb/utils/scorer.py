"""Vector similarity scoring utilities implemented from first principles.

This module provides implementations of various distance and similarity metrics
for vector comparison, particularly useful for debugging embedding search issues.
"""

import numpy as np
import numpy.typing as npt
from typing import Tuple, List, Optional


def cosine_similarity(
    vec1: npt.NDArray[np.float32], vec2: npt.NDArray[np.float32]
) -> float:
    """Calculate cosine similarity between two vectors from first principles.

    Cosine similarity = (A · B) / (||A|| * ||B||)

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score between -1 and 1 (1 = identical direction)
    """
    # Calculate dot product
    dot_product = np.dot(vec1, vec2)

    # Calculate magnitudes (L2 norms)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0

    # Calculate cosine similarity
    return float(dot_product / (norm1 * norm2))


def cosine_distance(
    vec1: npt.NDArray[np.float32], vec2: npt.NDArray[np.float32]
) -> float:
    """Calculate cosine distance between two vectors.

    Cosine distance = 1 - cosine_similarity

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine distance between 0 and 2 (0 = identical)
    """
    return 1.0 - cosine_similarity(vec1, vec2)


def euclidean_distance(
    vec1: npt.NDArray[np.float32], vec2: npt.NDArray[np.float32]
) -> float:
    """Calculate Euclidean (L2) distance between two vectors.

    L2 distance = sqrt(sum((a_i - b_i)^2))

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Euclidean distance (>= 0)
    """
    diff = vec1 - vec2
    return float(np.sqrt(np.sum(diff * diff)))


def euclidean_distance_squared(
    vec1: npt.NDArray[np.float32], vec2: npt.NDArray[np.float32]
) -> float:
    """Calculate squared Euclidean distance (saves sqrt computation).

    L2^2 distance = sum((a_i - b_i)^2)

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Squared Euclidean distance (>= 0)
    """
    diff = vec1 - vec2
    return float(np.sum(diff * diff))


def normalize_vector(vec: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Normalize a vector to unit length.

    Args:
        vec: Input vector

    Returns:
        Normalized vector with L2 norm = 1
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def l2_to_similarity_score(l2_distance: float) -> float:
    """Convert L2 distance to a similarity score.

    Uses the formula: similarity = 1 / (1 + distance)
    This maps distance [0, inf) to similarity (1, 0]

    Args:
        l2_distance: L2 (Euclidean) distance

    Returns:
        Similarity score between 0 and 1
    """
    return 1.0 / (1.0 + l2_distance)


def cosine_similarity_from_normalized_l2(l2_distance: float) -> float:
    """Convert L2 distance between normalized vectors to cosine similarity.

    For normalized vectors (unit length):
    ||a - b||^2 = ||a||^2 + ||b||^2 - 2(a·b) = 2 - 2(a·b)
    Since ||a|| = ||b|| = 1 for normalized vectors

    Therefore: cosine_similarity = 1 - (l2_distance^2 / 2)

    Args:
        l2_distance: L2 distance between normalized vectors

    Returns:
        Cosine similarity
    """
    return 1.0 - (l2_distance**2) / 2.0


def debug_vector_similarity(
    vec1: npt.NDArray[np.float32],
    vec2: npt.NDArray[np.float32],
    name1: str = "Vector 1",
    name2: str = "Vector 2",
) -> dict:
    """Debug helper to compare vectors with multiple metrics.

    Args:
        vec1: First vector
        vec2: Second vector
        name1: Name for first vector
        name2: Name for second vector

    Returns:
        Dictionary with various similarity/distance metrics
    """
    # Check if vectors are normalized
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # Calculate metrics
    cos_sim = cosine_similarity(vec1, vec2)
    cos_dist = cosine_distance(vec1, vec2)
    l2_dist = euclidean_distance(vec1, vec2)
    l2_dist_sq = euclidean_distance_squared(vec1, vec2)

    # Normalize vectors and recalculate
    vec1_norm = normalize_vector(vec1)
    vec2_norm = normalize_vector(vec2)
    l2_dist_norm = euclidean_distance(vec1_norm, vec2_norm)
    cos_sim_from_l2 = cosine_similarity_from_normalized_l2(l2_dist_norm)

    return {
        "vector_info": {
            f"{name1}_norm": float(norm1),
            f"{name2}_norm": float(norm2),
            f"{name1}_is_normalized": bool(abs(norm1 - 1.0) < 1e-6),
            f"{name2}_is_normalized": bool(abs(norm2 - 1.0) < 1e-6),
        },
        "similarities": {
            "cosine_similarity": cos_sim,
            "cosine_distance": cos_dist,
            "l2_distance": l2_dist,
            "l2_distance_squared": l2_dist_sq,
            "l2_distance_normalized": l2_dist_norm,
            "cosine_sim_from_normalized_l2": cos_sim_from_l2,
            "l2_to_similarity_score": l2_to_similarity_score(l2_dist),
        },
        "dimension": len(vec1),
    }


def rank_by_similarity(
    query_vec: npt.NDArray[np.float32],
    candidate_vecs: List[npt.NDArray[np.float32]],
    metric: str = "cosine",
) -> List[Tuple[int, float]]:
    """Rank candidate vectors by similarity to query vector.

    Args:
        query_vec: Query vector
        candidate_vecs: List of candidate vectors
        metric: Similarity metric to use ('cosine', 'l2', 'l2_squared')

    Returns:
        List of (index, score) tuples sorted by similarity (highest first)
    """
    scores = []

    for i, candidate in enumerate(candidate_vecs):
        if metric == "cosine":
            score = cosine_similarity(query_vec, candidate)
        elif metric == "l2":
            # Convert distance to similarity (lower distance = higher similarity)
            score = -euclidean_distance(query_vec, candidate)
        elif metric == "l2_squared":
            score = -euclidean_distance_squared(query_vec, candidate)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        scores.append((i, score))

    # Sort by score (highest first)
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores
