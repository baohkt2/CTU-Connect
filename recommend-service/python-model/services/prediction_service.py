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
from services.academic_classifier_service import get_academic_classifier, AcademicClassifierService

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
        
        # Academic classifier (fine-tuned PhoBERT)
        self.academic_classifier: Optional[AcademicClassifierService] = None
        self.use_ml_classifier = settings.USE_ML_ACADEMIC_CLASSIFIER
        
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
                logger.info("[OK] Vectorizer loaded")
            else:
                logger.warning(f"[WARN] Vectorizer not found at {settings.VECTORIZER_PATH}")
            
            if os.path.exists(settings.RANKING_MODEL_PATH):
                with open(settings.RANKING_MODEL_PATH, 'rb') as f:
                    self.ranking_model = pickle.load(f)
                logger.info("[OK] Ranking model loaded")
            else:
                logger.warning(f"[WARN] Ranking model not found at {settings.RANKING_MODEL_PATH}")
            
            # Load PhoBERT model (for embedding generation)
            try:
                from transformers import AutoModel, AutoTokenizer
                self.phobert_tokenizer = AutoTokenizer.from_pretrained(settings.PHOBERT_MODEL_NAME)
                self.phobert_model = AutoModel.from_pretrained(settings.PHOBERT_MODEL_NAME)
                self.phobert_model.eval()
                logger.info("[OK] PhoBERT model loaded")
            except Exception as e:
                logger.warning(f"[WARN] PhoBERT loading failed: {e}")
            
            # Load Academic Classifier (fine-tuned PhoBERT for academic classification)
            if self.use_ml_classifier:
                try:
                    self.academic_classifier = get_academic_classifier(
                        settings.ACADEMIC_CLASSIFIER_MODEL_PATH
                    )
                    if self.academic_classifier.is_ready():
                        logger.info("[OK] Academic classifier loaded (ML-based)")
                    else:
                        logger.warning("[WARN] Academic classifier loaded but not ready, using fallback")
                except Exception as e:
                    logger.warning(f"[WARN] Academic classifier loading failed: {e}, using heuristic fallback")
                    self.academic_classifier = None
            else:
                logger.info("[INFO] ML academic classifier disabled, using heuristic")
            
            logger.info("[OK] All models loaded successfully")
            
        except Exception as e:
            logger.error(f"[ERROR] Model loading error: {e}")
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
            
            # Validate user embedding
            if user_embedding is None or user_embedding.size == 0:
                logger.warning("Invalid user embedding, generating default")
                user_embedding = np.zeros(self.embedding_dimension, dtype=np.float32)
            
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
                    
                    if post_embedding is None or post_embedding.size == 0:
                        logger.warning(f"Failed to generate valid embedding for post {post_id}, skipping")
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
                    
                    # Ensure all scores are valid floats (not None, not NaN, not Inf)
                    content_sim = 0.5 if (content_sim is None or np.isnan(content_sim) or np.isinf(content_sim)) else float(content_sim)
                    implicit_fb = 0.5 if (implicit_fb is None or np.isnan(implicit_fb) or np.isinf(implicit_fb)) else float(implicit_fb)
                    academic_score = 0.0 if (academic_score is None or np.isnan(academic_score) or np.isinf(academic_score)) else float(academic_score)
                    popularity = 0.0 if (popularity is None or np.isnan(popularity) or np.isinf(popularity)) else float(popularity)
                    
                    # Add randomness boost for diversity (especially important for new users)
                    diversity_boost = np.random.uniform(0.0, 0.15)  # Random boost 0-15%
                    
                    # Debug log with detailed score info
                    logger.debug(f"Post {post_id} scores: " +
                                f"content={content_sim}, " +
                                f"implicit={implicit_fb}, " +
                                f"academic={academic_score}, " +
                                f"popularity={popularity}, " +
                                f"diversity={diversity_boost}")
                    
                    # Combine scores with weights - ensure float multiplication
                    final_score = (
                        float(settings.WEIGHT_CONTENT_SIMILARITY) * float(content_sim) +
                        float(settings.WEIGHT_IMPLICIT_FEEDBACK) * float(implicit_fb) +
                        float(settings.WEIGHT_ACADEMIC_SCORE) * float(academic_score) +
                        float(settings.WEIGHT_POPULARITY) * float(popularity) +
                        diversity_boost  # Add diversity for better distribution
                    )
                    
                    # Clip score to [0, 1]
                    final_score = max(0.0, min(1.0, float(final_score)))
                    
                    logger.info(f"[OK] Post {post_id} final score: {final_score:.4f}")
                    
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
        """
        Classify if content is academic using ML model or fallback heuristic
        
        Uses fine-tuned PhoBERT classifier if available, otherwise falls back
        to keyword-based heuristic classification.
        """
        # Try ML classifier first
        if self.academic_classifier and self.academic_classifier.is_ready():
            try:
                result = self.academic_classifier.predict(content)
                logger.debug(f"[ML] ML classification: {result['label']} ({result['confidence']:.2%})")
                return {
                    "is_academic": result["is_academic"],
                    "confidence": result["confidence"],
                    "category": "academic" if result["is_academic"] else "general",
                    "probabilities": result.get("probabilities", {}),
                    "method": "ml_classifier"
                }
            except Exception as e:
                logger.warning(f"[WARN] ML classifier failed, using fallback: {e}")
        
        # Fallback: Heuristic-based classifier
        return self._heuristic_classify_academic(content)
    
    def _heuristic_classify_academic(self, content: str) -> Dict[str, Any]:
        """Fallback heuristic-based classifier using keywords"""
        academic_keywords = [
            "nghiên cứu", "học thuật", "hội thảo", "seminar", "workshop",
            "luận văn", "luận án", "đề tài", "học bổng", "scholarship",
            "tuyển sinh", "đào tạo", "khóa học", "giảng viên", "giáo sư",
            "báo cáo", "thuyết trình", "đề cương", "tài liệu", "giáo trình",
            "thực tập", "internship", "research", "paper", "thesis",
            "academic", "conference", "journal", "publication"
        ]
        
        content_lower = content.lower()
        score = sum(1 for keyword in academic_keywords if keyword in content_lower)
        confidence = min(score / 5.0, 1.0)  # Normalize
        
        is_academic = confidence >= settings.ACADEMIC_CONFIDENCE_THRESHOLD
        
        return {
            "is_academic": is_academic,
            "confidence": round(confidence, 4),
            "category": "academic" if is_academic else "general",
            "method": "heuristic"
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
    ) -> np.ndarray:
        """
        Generate user profile embedding with improved fallback strategy
        Always returns a valid embedding (never None)
        """
        try:
            user_text_parts = []
            
            # Extract user academic info - handle various field naming conventions
            user_id = user_academic.get('userId') or user_academic.get('user_id', '')
            major = user_academic.get('major', '')
            faculty = user_academic.get('faculty', '')
            degree = user_academic.get('degree', '')
            batch = user_academic.get('batch', '')
            
            # Build user profile text from academic info
            if major:
                user_text_parts.append(f"chuyên ngành {major}")
            if faculty:
                user_text_parts.append(f"khoa {faculty}")
            if degree:
                user_text_parts.append(f"bậc {degree}")
            if batch:
                user_text_parts.append(f"khóa {batch}")
            
            user_text = " ".join(user_text_parts).strip()
            logger.debug(f"User academic text: {user_text}")
            
            # Strategy 2: If no academic info, try to extract from history
            if not user_text and user_history and len(user_history) > 0:
                logger.info(f"No academic info for user {user_id}, extracting from {len(user_history)} history items")
                
                # Collect content from recent interactions
                history_texts = []
                for h in user_history[-10:]:  # Last 10 interactions
                    content = h.get("content", "")
                    if content:
                        # Take first 80 chars to avoid too long text
                        history_texts.append(content[:80])
                
                if history_texts:
                    user_text = " ".join(history_texts).strip()
                    logger.debug(f"Generated user text from history: {user_text[:100]}...")
            
            # Strategy 3: Fallback to generic Vietnamese student profile
            if not user_text or len(user_text) < 5:
                user_text = "sinh viên đại học Cần Thơ quan tâm học tập và hoạt động sinh viên"
                logger.info(f"Using default profile text for user {user_id}")
            
            # Ensure text is not too long
            if len(user_text) > 500:
                user_text = user_text[:500]
            
            logger.info(f"Generating embedding for user {user_id} with text length: {len(user_text)}")
            
            # Generate embedding using PhoBERT
            embedding = await self.generate_embedding(user_text)
            
            # Final validation
            if embedding is None or embedding.size == 0:
                logger.warning(f"Failed to generate embedding, returning default for user {user_id}")
                # Return a small random embedding (better than zeros for cold start)
                embedding = np.random.normal(0, 0.1, self.embedding_dimension).astype(np.float32)
            
            # Validate size
            if embedding.size != self.embedding_dimension:
                logger.error(f"Invalid embedding size: {embedding.size}, expected: {self.embedding_dimension}")
                embedding = np.random.normal(0, 0.1, self.embedding_dimension).astype(np.float32)
            
            # Normalize embedding to unit length (helps with cosine similarity)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            logger.debug(f"Generated user embedding: shape={embedding.shape}, norm={np.linalg.norm(embedding):.4f}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating user embedding: {e}", exc_info=True)
            # Return normalized random embedding as last resort (better than zeros)
            embedding = np.random.normal(0, 0.1, self.embedding_dimension).astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
    
    def _calculate_content_similarity(
        self,
        user_embedding: Optional[np.ndarray],
        post_embedding: Optional[np.ndarray]
    ) -> float:
        """Calculate cosine similarity between user and post embeddings"""
        # Handle None embeddings
        if user_embedding is None or post_embedding is None:
            logger.warning("One or both embeddings are None, returning cold-start score")
            return 0.5  # Neutral default for cold start
        
        # Handle empty embeddings
        if user_embedding.size == 0 or post_embedding.size == 0:
            logger.warning("One or both embeddings are empty, returning cold-start score")
            return 0.5
        
        # Check if embeddings have correct shape
        if len(user_embedding.shape) == 0 or len(post_embedding.shape) == 0:
            logger.warning("Invalid embedding shape, returning cold-start score")
            return 0.5
        
        # Check dimension mismatch
        if user_embedding.shape[0] != post_embedding.shape[0]:
            logger.error(f"Embedding dimension mismatch: user={user_embedding.shape[0]}, post={post_embedding.shape[0]}")
            return 0.5
        
        try:
            # Calculate cosine similarity
            similarity = cosine_similarity(user_embedding, post_embedding)
            
            # Ensure valid range
            if np.isnan(similarity) or np.isinf(similarity):
                logger.warning("Invalid similarity value (NaN/Inf), returning cold-start score")
                return 0.5
            
            # Clip to [0, 1] range
            similarity = max(0.0, min(1.0, float(similarity)))
            
            # If similarity is too low, boost it slightly for better UX
            # (avoid showing only zero-score items to new users)
            if similarity < 0.1:
                similarity = 0.5  # Give it a neutral chance
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.5  # Neutral default on error
    
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
