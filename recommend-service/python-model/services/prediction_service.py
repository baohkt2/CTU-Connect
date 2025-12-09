"""
Main prediction service for ML-based recommendations
"""

import os
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import torch

from config import settings
from models.schemas import RankedPost
from utils.similarity import cosine_similarity
from utils.feature_engineering import extract_features

logger = logging.getLogger(__name__)


class PredictionService:
    """Main prediction service for recommendation ranking"""
    
    def __init__(self):
        self.model_path = settings.MODEL_PATH
        self.embedding_dimension = settings.EMBEDDING_DIMENSION
        self.academic_categories = settings.ACADEMIC_CATEGORIES
        self.model_version = "1.0.0"
        
        # Model placeholders
        self.vectorizer = None
        self.post_encoder = None
        self.academic_encoder = None
        self.ranking_model = None
        self.phobert_model = None
        self.phobert_tokenizer = None
        
        # Performance metrics
        self.metrics = {
            "predictions_count": 0,
            "total_latency_ms": 0,
            "cache_hits": 0
        }
        
        # Initialize models
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models from disk"""
        logger.info("Loading ML models...")
        
        try:
            # Try to load pre-trained models if they exist
            if os.path.exists(settings.VECTORIZER_PATH):
                with open(settings.VECTORIZER_PATH, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                logger.info("✅ Vectorizer loaded")
            else:
                logger.warning(f"⚠️ Vectorizer not found at {settings.VECTORIZER_PATH}")
            
            if os.path.exists(settings.RANKING_MODEL_PATH):
                with open(settings.RANKING_MODEL_PATH, 'rb') as f:
                    self.ranking_model = pickle.load(f)
                logger.info("✅ Ranking model loaded")
            else:
                logger.warning(f"⚠️ Ranking model not found at {settings.RANKING_MODEL_PATH}")
            
            # Load PhoBERT model (for embedding generation)
            try:
                from transformers import AutoModel, AutoTokenizer
                self.phobert_tokenizer = AutoTokenizer.from_pretrained(settings.PHOBERT_MODEL_NAME)
                self.phobert_model = AutoModel.from_pretrained(settings.PHOBERT_MODEL_NAME)
                self.phobert_model.eval()
                logger.info("✅ PhoBERT model loaded")
            except Exception as e:
                logger.warning(f"⚠️ PhoBERT loading failed: {e}")
            
            logger.info("✅ All models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Model loading error: {e}")
            logger.warning("Service will use fallback methods")
    
    def is_ready(self) -> bool:
        """Check if service is ready"""
        # Service is ready if at least PhoBERT is loaded
        return self.phobert_model is not None
    
    def get_model_version(self) -> str:
        """Get current model version"""
        return self.model_version
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        avg_latency = 0
        if self.metrics["predictions_count"] > 0:
            avg_latency = self.metrics["total_latency_ms"] / self.metrics["predictions_count"]
        
        cache_hit_rate = 0
        if self.metrics["predictions_count"] > 0:
            cache_hit_rate = self.metrics["cache_hits"] / self.metrics["predictions_count"]
        
        return {
            "predictions_count": self.metrics["predictions_count"],
            "avg_latency_ms": round(avg_latency, 2),
            "cache_hit_rate": round(cache_hit_rate, 4)
        }
    
    async def predict(
        self,
        user_academic: Dict[str, Any],
        user_history: List[Dict[str, Any]],
        candidate_posts: List[Dict[str, Any]],
        top_k: int = 20
    ) -> List[RankedPost]:
        """
        Main prediction method
        
        Steps:
        1. Generate embeddings for user profile and posts
        2. Calculate content similarity scores
        3. Calculate implicit feedback scores from history
        4. Calculate academic scores
        5. Calculate popularity scores
        6. Combine scores and rank
        """
        start_time = datetime.now()
        
        try:
            ranked_posts = []
            
            # Generate user profile embedding
            user_embedding = await self._generate_user_embedding(user_academic, user_history)
            
            # Process each candidate post
            for post in candidate_posts:
                try:
                    post_id = post.get("postId", "unknown")
                    content = post.get("content", "")
                    
                    if not content:
                        logger.warning(f"Post {post_id} has no content, skipping")
                        continue
                    
                    # Generate post embedding
                    post_embedding = await self.generate_embedding(content)
                    
                    if post_embedding is None:
                        logger.warning(f"Failed to generate embedding for post {post_id}, skipping")
                        continue
                    
                    # Validate post embedding
                    if post_embedding.size != self.embedding_dimension:
                        logger.error(f"Invalid post embedding size for {post_id}: {post_embedding.size}")
                        continue
                    
                    # Calculate scores (with None handling) - ensure all return float
                    content_sim = self._calculate_content_similarity(user_embedding, post_embedding)
                    implicit_fb = self._calculate_implicit_feedback(post, user_history)
                    academic_score = await self._calculate_academic_score(post, user_academic)
                    popularity = self._calculate_popularity_score(post)
                    
                    # Validate all scores are numbers
                    if any(score is None or not isinstance(score, (int, float)) 
                          for score in [content_sim, implicit_fb, academic_score, popularity]):
                        logger.error(f"Invalid scores for post {post_id}, skipping")
                        continue
                    
                    # Debug log
                    logger.debug(f"Post {post_id} scores: sim={content_sim:.3f}, fb={implicit_fb:.3f}, " +
                                f"acad={academic_score:.3f}, pop={popularity:.3f}")
                    
                    # Combine scores with weights - ensure float multiplication
                    final_score = (
                        float(settings.WEIGHT_CONTENT_SIMILARITY) * float(content_sim) +
                        float(settings.WEIGHT_IMPLICIT_FEEDBACK) * float(implicit_fb) +
                        float(settings.WEIGHT_ACADEMIC_SCORE) * float(academic_score) +
                        float(settings.WEIGHT_POPULARITY) * float(popularity)
                    )
                    
                    # Clip score to [0, 1]
                    final_score = max(0.0, min(1.0, float(final_score)))
                    
                    ranked_posts.append(RankedPost(
                        postId=post_id,
                        score=round(final_score, 4),
                        contentSimilarity=round(content_sim, 4),
                        implicitFeedback=round(implicit_fb, 4),
                        academicScore=round(academic_score, 4),
                        popularityScore=round(popularity, 4)
                    ))
                    
                except Exception as e:
                    logger.error(f"Error processing post {post.get('postId', 'unknown')}: {e}")
                    continue
            
            # Sort by score descending
            ranked_posts.sort(key=lambda x: x.score, reverse=True)
            
            # Take top K and add rank
            result = ranked_posts[:top_k]
            for idx, post in enumerate(result):
                post.rank = idx + 1
            
            # Update metrics
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics["predictions_count"] += 1
            self.metrics["total_latency_ms"] += latency_ms
            
            logger.info(f"Prediction completed: {len(result)} posts ranked in {latency_ms:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            # Return fallback: sort by popularity
            return self._fallback_ranking(candidate_posts, top_k)
    
    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using PhoBERT"""
        if not self.phobert_model:
            # Fallback: random embedding
            logger.warning("PhoBERT not loaded, using random embedding")
            return np.random.rand(self.embedding_dimension).astype(np.float32)
        
        try:
            # Tokenize
            inputs = self.phobert_tokenizer(
                text,
                return_tensors="pt",
                max_length=settings.MAX_SEQUENCE_LENGTH,
                truncation=True,
                padding=True
            )
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.phobert_model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return np.random.rand(self.embedding_dimension).astype(np.float32)
    
    async def classify_academic(self, content: str) -> Dict[str, Any]:
        """Classify if content is academic"""
        # Simple heuristic-based classifier (can be replaced with trained model)
        academic_keywords = [
            "nghiên cứu", "học thuật", "hội thảo", "seminar", "workshop",
            "luận văn", "luận án", "đề tài", "học bổng", "scholarship",
            "tuyển sinh", "đào tạo", "khóa học", "giảng viên", "giáo sư"
        ]
        
        content_lower = content.lower()
        score = sum(1 for keyword in academic_keywords if keyword in content_lower)
        confidence = min(score / 5.0, 1.0)  # Normalize
        
        is_academic = confidence >= settings.ACADEMIC_CONFIDENCE_THRESHOLD
        
        return {
            "is_academic": is_academic,
            "confidence": round(confidence, 4),
            "category": "academic" if is_academic else "general"
        }
    
    async def reload_models(self):
        """Reload models from disk (hot reload)"""
        logger.info("Reloading models...")
        self._load_models()
        logger.info("Models reloaded successfully")
    
    # Private helper methods
    
    async def _generate_user_embedding(
        self,
        user_academic: Dict[str, Any],
        user_history: List[Dict[str, Any]]
    ) -> Optional[np.ndarray]:
        """Generate user profile embedding"""
        try:
            user_text_parts = []
            
            # Combine user academic info into text
            major = user_academic.get('major', '')
            faculty = user_academic.get('faculty', '')
            degree = user_academic.get('degree', '')
            batch = user_academic.get('batch', '')
            
            if major:
                user_text_parts.append(major)
            if faculty:
                user_text_parts.append(faculty)
            if degree:
                user_text_parts.append(degree)
            if batch:
                user_text_parts.append(str(batch))
            
            user_text = " ".join(user_text_parts).strip()
            
            # If no academic info, generate from history
            if not user_text and user_history:
                # Use most recent interactions
                recent_content = " ".join([
                    h.get("content", "")[:100]  # First 100 chars
                    for h in user_history[-5:]  # Last 5 interactions
                    if h.get("content")
                ])
                user_text = recent_content.strip()
            
            # Fallback: default text based on user type
            if not user_text:
                user_text = "sinh viên đại học cần tư vấn tuyển sinh"  # Generic Vietnamese student text
                logger.warning("No user info available, using default Vietnamese text")
            
            logger.debug(f"Generating user embedding for: {user_text[:80]}...")
            
            # Generate base embedding
            embedding = await self.generate_embedding(user_text)
            
            if embedding is None:
                logger.error("Failed to generate user embedding even with fallback text")
                # Return zero embedding as last resort
                return np.zeros(self.embedding_dimension, dtype=np.float32)
            
            # Validate embedding
            if embedding.size != self.embedding_dimension:
                logger.error(f"Invalid embedding size: {embedding.size}, expected: {self.embedding_dimension}")
                return np.zeros(self.embedding_dimension, dtype=np.float32)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating user embedding: {e}", exc_info=True)
            # Return zero embedding on error
            return np.zeros(self.embedding_dimension, dtype=np.float32)
    
    def _calculate_content_similarity(
        self,
        user_embedding: Optional[np.ndarray],
        post_embedding: Optional[np.ndarray]
    ) -> float:
        """Calculate cosine similarity between user and post embeddings"""
        # Handle None embeddings
        if user_embedding is None or post_embedding is None:
            logger.warning("One or both embeddings are None, returning default similarity")
            return 0.3  # Lower default for cold start
        
        # Handle empty embeddings
        if user_embedding.size == 0 or post_embedding.size == 0:
            logger.warning("One or both embeddings are empty, returning default similarity")
            return 0.3
        
        # Check if embeddings have correct shape
        if len(user_embedding.shape) == 0 or len(post_embedding.shape) == 0:
            logger.warning("Invalid embedding shape, returning default similarity")
            return 0.3
        
        try:
            similarity = cosine_similarity(user_embedding, post_embedding)
            # Ensure valid range
            if np.isnan(similarity) or np.isinf(similarity):
                logger.warning("Invalid similarity value (NaN/Inf), returning default")
                return 0.3
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.3
    
    def _calculate_implicit_feedback(
        self,
        post: Dict[str, Any],
        user_history: List[Dict[str, Any]]
    ) -> float:
        """Calculate implicit feedback score based on user history"""
        # Check if user has interacted with similar content
        
        if not user_history or len(user_history) == 0:
            return 0.5  # Neutral score for new users
        
        try:
            # Calculate average interaction quality
            total_interactions = len(user_history)
            positive_interactions = sum(
                1 for h in user_history
                if (h.get("liked", 0) > 0 or h.get("commented", 0) > 0 or 
                    h.get("action") in ["LIKE", "COMMENT", "SHARE", "SAVE"])
            )
            
            score = positive_interactions / total_interactions if total_interactions > 0 else 0.5
            return float(score)
        except Exception as e:
            logger.error(f"Error calculating implicit feedback: {e}")
            return 0.5
    
    async def _calculate_academic_score(
        self,
        post: Dict[str, Any],
        user_academic: Dict[str, Any]
    ) -> float:
        """Calculate academic relevance score"""
        try:
            # Classify post
            content = post.get("content", "")
            if not content:
                return 0.0
            
            classification = await self.classify_academic(content)
            academic_score = float(classification.get("confidence", 0.0))
            
            # Boost if same major/faculty
            boost = 0.0
            if post.get("authorMajor") and post.get("authorMajor") == user_academic.get("major"):
                boost += 0.2
            if post.get("authorFaculty") and post.get("authorFaculty") == user_academic.get("faculty"):
                boost += 0.1
            
            final_score = min(1.0, academic_score + boost)
            return float(final_score)
        except Exception as e:
            logger.error(f"Error calculating academic score: {e}")
            return 0.0
    
    def _calculate_popularity_score(self, post: Dict[str, Any]) -> float:
        """Calculate popularity score based on engagement"""
        try:
            likes = int(post.get("likesCount", 0) or 0)
            comments = int(post.get("commentsCount", 0) or 0)
            shares = int(post.get("sharesCount", 0) or 0)
            
            # Weighted sum
            engagement = likes * 1.0 + comments * 2.0 + shares * 3.0
            
            # Normalize using log scale
            normalized = np.log1p(engagement) / 10.0
            
            score = min(1.0, float(normalized))
            return score
        except Exception as e:
            logger.error(f"Error calculating popularity score: {e}")
            return 0.0
    
    def _fallback_ranking(
        self,
        candidate_posts: List[Dict[str, Any]],
        top_k: int
    ) -> List[RankedPost]:
        """Fallback ranking using popularity only"""
        logger.warning("Using fallback ranking (popularity-based)")
        
        ranked = []
        for post in candidate_posts:
            popularity = self._calculate_popularity_score(post)
            ranked.append(RankedPost(
                postId=post["postId"],
                score=popularity,
                popularityScore=popularity
            ))
        
        ranked.sort(key=lambda x: x.score, reverse=True)
        result = ranked[:top_k]
        
        for idx, post in enumerate(result):
            post.rank = idx + 1
        
        return result
