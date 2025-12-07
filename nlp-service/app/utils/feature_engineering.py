"""
Feature engineering utilities
"""

from typing import Dict, Any, List
import numpy as np


def extract_features(
    post: Dict[str, Any],
    user_academic: Dict[str, Any],
    user_history: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Extract features for ranking model
    
    Args:
        post: Candidate post
        user_academic: User academic profile
        user_history: User interaction history
    
    Returns:
        Dictionary of features
    """
    features = {}
    
    # Post features
    features["content_length"] = len(post.get("content", ""))
    features["hashtags_count"] = len(post.get("hashtags", []))
    features["likes_count"] = post.get("likesCount", 0)
    features["comments_count"] = post.get("commentsCount", 0)
    features["shares_count"] = post.get("sharesCount", 0)
    
    # Academic matching features
    features["same_major"] = 1.0 if post.get("authorMajor") == user_academic.get("major") else 0.0
    features["same_faculty"] = 1.0 if post.get("authorFaculty") == user_academic.get("faculty") else 0.0
    features["same_batch"] = 1.0 if post.get("authorBatch") == user_academic.get("batch") else 0.0
    
    # User history features
    features["user_interaction_count"] = len(user_history)
    features["user_likes_count"] = sum(h.get("liked", 0) for h in user_history)
    features["user_comments_count"] = sum(h.get("commented", 0) for h in user_history)
    
    # Engagement rate
    total_engagement = features["likes_count"] + features["comments_count"] + features["shares_count"]
    features["engagement_rate"] = np.log1p(total_engagement)
    
    return features


def normalize_features(features: Dict[str, float]) -> Dict[str, float]:
    """Normalize features to [0, 1] range"""
    normalized = {}
    
    # Features that need log transformation
    log_features = ["content_length", "engagement_rate", "user_interaction_count"]
    
    for key, value in features.items():
        if key in log_features:
            normalized[key] = np.log1p(value) / 10.0  # Scale down
        else:
            normalized[key] = min(1.0, value)  # Clip to 1.0
    
    return normalized


def encode_categorical(value: str, categories: List[str]) -> List[float]:
    """One-hot encode categorical value"""
    encoding = [0.0] * len(categories)
    if value in categories:
        idx = categories.index(value)
        encoding[idx] = 1.0
    return encoding
