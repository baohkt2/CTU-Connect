"""
Services module for CTU Connect Recommendation System
"""

from .prediction_service import PredictionService
from .user_similarity_service import UserSimilarityService, get_user_similarity_service

__all__ = [
    'PredictionService',
    'UserSimilarityService', 
    'get_user_similarity_service'
]
