"""
User Embedding Service
Handles user profile embedding generation with caching
Follows the architecture: Redis (24h cache) -> PostgreSQL (persistent) -> Generate new
"""

import numpy as np
from typing import List, Dict, Optional
import logging
import hashlib
import json

logger = logging.getLogger(__name__)


class UserEmbeddingService:
    """
    Service for managing user profile embeddings
    - Generate embeddings from user profiles
    - Cache in Redis (fast access, 24h TTL)
    - Store in PostgreSQL (persistent backup)
    """
    
    REDIS_KEY_PREFIX = "user_embedding:"
    CACHE_TTL_SECONDS = 86400  # 24 hours
    
    def __init__(self, inference_engine=None):
        """
        Initialize with inference engine
        
        Args:
            inference_engine: PhoBERT inference engine
        """
        self.inference_engine = inference_engine
        self.embedding_dimension = 768  # PhoBERT output dimension
        logger.info("[UserEmbedding] Service initialized")
    
    def set_inference_engine(self, inference_engine):
        """Set the inference engine"""
        self.inference_engine = inference_engine
    
    def _generate_cache_key(self, user_id: str) -> str:
        """Generate Redis cache key for user embedding"""
        return f"{self.REDIS_KEY_PREFIX}{user_id}"
    
    def _hash_user_profile(self, user_data: Dict) -> str:
        """
        Generate hash of user profile data to detect changes
        
        Args:
            user_data: User profile dictionary
            
        Returns:
            MD5 hash of profile data
        """
        # Create stable string representation
        profile_str = json.dumps({
            'major': user_data.get('major', ''),
            'faculty': user_data.get('faculty', ''),
            'courses': sorted(user_data.get('courses', [])),
            'skills': sorted(user_data.get('skills', [])),
            'bio': user_data.get('bio', '')[:500]  # Limit bio length
        }, sort_keys=True)
        
        return hashlib.md5(profile_str.encode()).hexdigest()
    
    def generate_embedding(self, user_data: Dict) -> np.ndarray:
        """
        Generate embedding for user profile using PhoBERT
        
        Args:
            user_data: Dictionary with user profile:
                - major: User's major
                - faculty: User's faculty  
                - courses: List of courses
                - skills: List of skills
                - bio: User bio text
                
        Returns:
            768-dimensional embedding vector
        """
        if self.inference_engine is None:
            logger.warning("[UserEmbedding] Inference engine not initialized, returning zero vector")
            return np.zeros(self.embedding_dimension, dtype=np.float32)
        
        try:
            # Use inference engine to encode user profile
            embedding = self.inference_engine.encode_user_profile(user_data)
            
            # Validate embedding
            if embedding is None or len(embedding) != self.embedding_dimension:
                logger.error(f"[UserEmbedding] Invalid embedding dimension: {len(embedding) if embedding is not None else 0}")
                return np.zeros(self.embedding_dimension, dtype=np.float32)
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"[UserEmbedding] Error generating embedding: {e}")
            return np.zeros(self.embedding_dimension, dtype=np.float32)
    
    def generate_embeddings_batch(self, users_data: List[Dict]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple users (batch processing)
        
        Args:
            users_data: List of user profile dictionaries
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for user_data in users_data:
            try:
                emb = self.generate_embedding(user_data)
                embeddings.append(emb)
            except Exception as e:
                logger.error(f"[UserEmbedding] Batch error for user: {e}")
                embeddings.append(np.zeros(self.embedding_dimension, dtype=np.float32))
        
        return embeddings
    
    def get_embedding(self, 
                     user_id: str, 
                     user_data: Dict,
                     redis_client=None,
                     force_regenerate: bool = False) -> Dict:
        """
        Get user embedding with caching strategy:
        1. Check Redis cache (if available)
        2. Generate new if cache miss or force_regenerate
        3. Cache result in Redis
        
        Args:
            user_id: User ID
            user_data: User profile data
            redis_client: Optional Redis client for caching
            force_regenerate: Force regeneration even if cached
            
        Returns:
            Dict with:
                - embedding: numpy array
                - source: 'cache' | 'generated'
                - profile_hash: hash of profile data
        """
        profile_hash = self._hash_user_profile(user_data)
        
        # Try Redis cache first (if not force regenerate)
        if not force_regenerate and redis_client is not None:
            try:
                cache_key = self._generate_cache_key(user_id)
                cached = redis_client.get(cache_key)
                
                if cached:
                    cached_data = json.loads(cached)
                    cached_hash = cached_data.get('profile_hash')
                    
                    # Check if profile changed
                    if cached_hash == profile_hash:
                        embedding_list = cached_data.get('embedding')
                        if embedding_list and len(embedding_list) == self.embedding_dimension:
                            logger.debug(f"[UserEmbedding] Cache HIT for user {user_id}")
                            return {
                                'embedding': np.array(embedding_list, dtype=np.float32),
                                'source': 'cache',
                                'profile_hash': profile_hash
                            }
                    else:
                        logger.debug(f"[UserEmbedding] Profile changed for user {user_id}, regenerating")
                        
            except Exception as e:
                logger.warning(f"[UserEmbedding] Redis cache read error: {e}")
        
        # Cache miss or error - generate new embedding
        logger.debug(f"[UserEmbedding] Cache MISS for user {user_id}, generating...")
        embedding = self.generate_embedding(user_data)
        
        # Cache in Redis (if available)
        if redis_client is not None:
            try:
                cache_key = self._generate_cache_key(user_id)
                cache_data = {
                    'embedding': embedding.tolist(),
                    'profile_hash': profile_hash,
                    'user_id': user_id
                }
                redis_client.setex(
                    cache_key,
                    self.CACHE_TTL_SECONDS,
                    json.dumps(cache_data)
                )
                logger.debug(f"[UserEmbedding] Cached for user {user_id} (TTL: 24h)")
            except Exception as e:
                logger.warning(f"[UserEmbedding] Redis cache write error: {e}")
        
        return {
            'embedding': embedding,
            'source': 'generated',
            'profile_hash': profile_hash
        }
    
    def get_embeddings_batch(self,
                            user_ids: List[str],
                            users_data: List[Dict],
                            redis_client=None) -> List[Dict]:
        """
        Get embeddings for multiple users with caching
        
        Args:
            user_ids: List of user IDs
            users_data: List of user profile data (same order as user_ids)
            redis_client: Optional Redis client
            
        Returns:
            List of dicts with embedding info
        """
        if len(user_ids) != len(users_data):
            raise ValueError("user_ids and users_data must have same length")
        
        results = []
        for user_id, user_data in zip(user_ids, users_data):
            result = self.get_embedding(user_id, user_data, redis_client)
            results.append(result)
        
        return results
    
    def invalidate_cache(self, user_id: str, redis_client=None) -> bool:
        """
        Invalidate cached embedding for a user
        
        Args:
            user_id: User ID
            redis_client: Redis client
            
        Returns:
            True if invalidated, False if not found or error
        """
        if redis_client is None:
            return False
        
        try:
            cache_key = self._generate_cache_key(user_id)
            deleted = redis_client.delete(cache_key)
            logger.info(f"[UserEmbedding] Invalidated cache for user {user_id}: {deleted > 0}")
            return deleted > 0
        except Exception as e:
            logger.error(f"[UserEmbedding] Cache invalidation error: {e}")
            return False


# Singleton instance
_user_embedding_service = None


def get_user_embedding_service(inference_engine=None) -> UserEmbeddingService:
    """
    Get or create singleton UserEmbeddingService
    
    Args:
        inference_engine: Optional inference engine to set
        
    Returns:
        UserEmbeddingService instance
    """
    global _user_embedding_service
    
    if _user_embedding_service is None:
        _user_embedding_service = UserEmbeddingService(inference_engine)
    elif inference_engine is not None:
        _user_embedding_service.set_inference_engine(inference_engine)
    
    return _user_embedding_service
