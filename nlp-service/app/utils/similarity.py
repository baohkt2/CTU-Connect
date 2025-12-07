"""
Similarity calculation utilities
"""

import numpy as np
from typing import List


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Cosine similarity score (0-1)
    """
    try:
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)
        
        # Calculate dot product
        similarity = np.dot(vec1_norm, vec2_norm)
        
        # Clip to [0, 1]
        return max(0.0, min(1.0, float(similarity)))
        
    except Exception as e:
        print(f"Cosine similarity error: {e}")
        return 0.0


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate Euclidean distance between two vectors"""
    try:
        distance = np.linalg.norm(vec1 - vec2)
        return float(distance)
    except Exception as e:
        print(f"Euclidean distance error: {e}")
        return float('inf')


def manhattan_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate Manhattan distance between two vectors"""
    try:
        distance = np.sum(np.abs(vec1 - vec2))
        return float(distance)
    except Exception as e:
        print(f"Manhattan distance error: {e}")
        return float('inf')


def batch_cosine_similarity(query_vec: np.ndarray, vectors: List[np.ndarray]) -> np.ndarray:
    """
    Calculate cosine similarity between query vector and multiple vectors
    
    Args:
        query_vec: Query vector
        vectors: List of vectors to compare
    
    Returns:
        Array of similarity scores
    """
    try:
        # Stack vectors
        matrix = np.vstack(vectors)
        
        # Normalize
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
        
        # Calculate similarities
        similarities = np.dot(matrix_norm, query_norm)
        
        # Clip to [0, 1]
        similarities = np.clip(similarities, 0.0, 1.0)
        
        return similarities
        
    except Exception as e:
        print(f"Batch cosine similarity error: {e}")
        return np.zeros(len(vectors))
