"""
User Similarity Service for Friend Recommendations
Uses PhoBERT embeddings to compute user-to-user similarity
Implements hybrid scoring for friend ranking
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class UserSimilarityService:
    """
    Service for computing user similarities and ranking friend candidates
    Uses PhoBERT embeddings combined with multiple signals
    """
    
    # Scoring weights for hybrid ranking
    WEIGHT_CONTENT_SIMILARITY = 0.30
    WEIGHT_MUTUAL_FRIENDS = 0.25
    WEIGHT_ACADEMIC = 0.20
    WEIGHT_ACTIVITY = 0.15
    WEIGHT_RECENCY = 0.10
    
    def __init__(self, inference_engine=None):
        """
        Initialize with optional inference engine
        
        Args:
            inference_engine: PhoBERT inference engine instance
        """
        self.inference_engine = inference_engine
        logger.info("UserSimilarityService initialized")
    
    def set_inference_engine(self, inference_engine):
        """Set the inference engine"""
        self.inference_engine = inference_engine
    
    def generate_user_embedding(self, user_data: Dict) -> np.ndarray:
        """
        Generate embedding for a user profile
        
        Args:
            user_data: Dictionary containing user profile information
                - major: User's major
                - faculty: User's faculty
                - bio: User bio
                - interests: List of interests
                - skills: List of skills
                
        Returns:
            768-dimensional embedding vector
        """
        if self.inference_engine is None:
            raise RuntimeError("Inference engine not initialized")
        
        return self.inference_engine.encode_user_profile(user_data)
    
    def generate_user_embeddings_batch(self, users_data: List[Dict]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple users
        
        Args:
            users_data: List of user data dictionaries
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for user_data in users_data:
            try:
                emb = self.generate_user_embedding(user_data)
                embeddings.append(emb)
            except Exception as e:
                logger.error(f"Error generating embedding for user: {e}")
                # Return zero vector as fallback
                embeddings.append(np.zeros(768))
        return embeddings
    
    def compute_user_similarity(self, 
                                user1_embedding: np.ndarray, 
                                user2_embedding: np.ndarray) -> float:
        """
        Compute cosine similarity between two user embeddings
        
        Args:
            user1_embedding: First user's embedding
            user2_embedding: Second user's embedding
            
        Returns:
            Cosine similarity score (0-1)
        """
        if self.inference_engine is None:
            raise RuntimeError("Inference engine not initialized")
        
        return self.inference_engine.compute_similarity(user1_embedding, user2_embedding)
    
    def compute_batch_similarity(self,
                                 query_embedding: np.ndarray,
                                 candidate_embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Compute similarity between one user and multiple candidates
        
        Args:
            query_embedding: Query user's embedding
            candidate_embeddings: List of candidate embeddings
            
        Returns:
            Array of similarity scores
        """
        if len(candidate_embeddings) == 0:
            return np.array([])
        
        candidates = np.array(candidate_embeddings)
        
        if self.inference_engine is None:
            raise RuntimeError("Inference engine not initialized")
        
        return self.inference_engine.compute_batch_similarity(query_embedding, candidates)
    
    def calculate_hybrid_score(self,
                               content_similarity: float,
                               mutual_friends_score: float = 0.0,
                               academic_score: float = 0.0,
                               activity_score: float = 0.0,
                               recency_score: float = 0.0) -> float:
        """
        Calculate hybrid relevance score using multiple signals
        
        Weights:
        - Content Similarity (PhoBERT): 30%
        - Mutual Friends: 25%
        - Academic Connection: 20%
        - Activity Score: 15%
        - Recency: 10%
        
        Args:
            content_similarity: PhoBERT embedding similarity (0-1)
            mutual_friends_score: Normalized mutual friends score (0-1)
            academic_score: Academic connection score (0-1)
            activity_score: User activity score (0-1)
            recency_score: How recently active (0-1)
            
        Returns:
            Final hybrid score (0-1)
        """
        return (
            content_similarity * self.WEIGHT_CONTENT_SIMILARITY +
            mutual_friends_score * self.WEIGHT_MUTUAL_FRIENDS +
            academic_score * self.WEIGHT_ACADEMIC +
            activity_score * self.WEIGHT_ACTIVITY +
            recency_score * self.WEIGHT_RECENCY
        )
    
    def rank_friend_candidates(self,
                               current_user_embedding: np.ndarray,
                               candidate_embeddings: List[np.ndarray],
                               candidate_ids: List[str],
                               additional_scores: Optional[Dict[str, Dict[str, float]]] = None,
                               top_k: int = 20) -> List[Dict]:
        """
        Rank friend candidates using hybrid scoring
        
        Args:
            current_user_embedding: Current user's PhoBERT embedding
            candidate_embeddings: List of candidate user embeddings
            candidate_ids: List of candidate user IDs
            additional_scores: Dict of additional scores per user
                {
                    "user_id": {
                        "mutual_friends_score": 0.5,
                        "academic_score": 0.8,
                        "activity_score": 0.6,
                        "recency_score": 0.9
                    }
                }
            top_k: Number of top candidates to return
            
        Returns:
            List of ranked candidates with scores
        """
        if len(candidate_embeddings) == 0:
            return []
        
        # Compute content similarities
        content_similarities = self.compute_batch_similarity(
            current_user_embedding, 
            candidate_embeddings
        )
        
        results = []
        
        for idx, (cand_id, content_sim) in enumerate(zip(candidate_ids, content_similarities)):
            # Get additional scores
            additional = additional_scores.get(cand_id, {}) if additional_scores else {}
            mutual_score = additional.get('mutual_friends_score', 0.0)
            academic_score = additional.get('academic_score', 0.0)
            activity_score = additional.get('activity_score', 0.0)
            recency_score = additional.get('recency_score', 0.0)
            
            # Calculate hybrid score
            final_score = self.calculate_hybrid_score(
                content_similarity=float(content_sim),
                mutual_friends_score=mutual_score,
                academic_score=academic_score,
                activity_score=activity_score,
                recency_score=recency_score
            )
            
            results.append({
                'user_id': cand_id,
                'final_score': float(final_score),
                'content_similarity': float(content_sim),
                'mutual_friends_score': float(mutual_score),
                'academic_score': float(academic_score),
                'activity_score': float(activity_score),
                'recency_score': float(recency_score)
            })
        
        # Sort by final score descending
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Return top K
        return results[:top_k]
    
    def determine_suggestion_type(self, scores: Dict) -> str:
        """
        Determine the primary suggestion reason based on scores
        
        Args:
            scores: Dictionary with all score components
            
        Returns:
            Suggestion type string
        """
        content_sim = scores.get('content_similarity', 0)
        mutual_friends = scores.get('mutual_friends_score', 0)
        academic = scores.get('academic_score', 0)
        activity = scores.get('activity_score', 0)
        
        # Find the dominant signal
        signals = [
            ('CONTENT_SIMILARITY', content_sim * self.WEIGHT_CONTENT_SIMILARITY),
            ('MUTUAL_FRIENDS', mutual_friends * self.WEIGHT_MUTUAL_FRIENDS),
            ('ACADEMIC_CONNECTION', academic * self.WEIGHT_ACADEMIC),
            ('ACTIVITY_BASED', activity * self.WEIGHT_ACTIVITY)
        ]
        
        # Get the signal with highest weighted contribution
        dominant = max(signals, key=lambda x: x[1])
        return dominant[0]
    
    def generate_suggestion_reason(self, 
                                   scores: Dict,
                                   user_info: Optional[Dict] = None) -> str:
        """
        Generate human-readable suggestion reason
        
        Args:
            scores: Dictionary with all score components
            user_info: Optional user info for context
            
        Returns:
            Suggestion reason string
        """
        reasons = []
        
        # Mutual friends
        mutual_count = scores.get('mutual_friends_count', 0)
        if mutual_count > 0:
            reasons.append(f"{mutual_count} bạn chung")
        
        # Academic connection
        if user_info:
            if user_info.get('same_major'):
                reasons.append(f"Cùng ngành {user_info.get('major_name', '')}")
            elif user_info.get('same_faculty'):
                reasons.append(f"Cùng khoa {user_info.get('faculty_name', '')}")
            if user_info.get('same_batch'):
                reasons.append(f"Cùng khóa {user_info.get('batch_year', '')}")
        
        # Content similarity
        if scores.get('content_similarity', 0) > 0.7:
            reasons.append("Gợi ý cho bạn")
        
        # Activity
        if scores.get('activity_score', 0) > 0.6:
            reasons.append("Hoạt động tích cực")
        
        return " • ".join(reasons) if reasons else "Gợi ý cho bạn"


# Singleton instance
_user_similarity_service = None


def get_user_similarity_service(inference_engine=None) -> UserSimilarityService:
    """
    Get or create singleton UserSimilarityService instance
    
    Args:
        inference_engine: Optional inference engine to set
        
    Returns:
        UserSimilarityService instance
    """
    global _user_similarity_service
    
    if _user_similarity_service is None:
        _user_similarity_service = UserSimilarityService(inference_engine)
    elif inference_engine is not None:
        _user_similarity_service.set_inference_engine(inference_engine)
    
    return _user_similarity_service
