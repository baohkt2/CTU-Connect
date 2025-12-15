"""
CTU Connect Recommendation Service - Python AI Inference Layer
Unified FastAPI service for embedding generation, similarity computation, and ML predictions
According to ARCHITECTURE.md
"""

import sys
import os

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        except (AttributeError, ValueError):
            pass
    if not isinstance(sys.stderr, io.TextIOWrapper) or sys.stderr.encoding != 'utf-8':
        try:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        except (AttributeError, ValueError):
            pass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import logging
from datetime import datetime
from inference import get_inference_engine

# Configure logging
os.makedirs('logs', exist_ok=True)

# Create handlers with UTF-8 encoding
file_handler = logging.FileHandler(
    f'logs/python-service-{datetime.now().strftime("%Y%m%d")}.log',
    encoding='utf-8'
)
stream_handler = logging.StreamHandler()

# For Windows, ensure stream handler uses UTF-8 with error replacement
if sys.platform == 'win32':
    stream_handler.setStream(
        open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace', buffering=1)
    )

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[file_handler, stream_handler],
    force=True
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CTU Connect Recommendation - AI Inference Service",
    description="PhoBERT-based AI inference: embeddings, similarity, predictions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine
inference_engine = None


@app.on_event("startup")
async def startup_event():
    """Initialize inference engine on startup"""
    global inference_engine
    try:
        logger.info("🚀 Starting CTU Connect AI Inference Service...")
        logger.info("📦 Loading PhoBERT model...")
        inference_engine = get_inference_engine()
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.warning("⚠️  Service will start in fallback mode")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Shutting down AI Inference Service...")


# ==================== Pydantic Models ====================

class PostEmbeddingRequest(BaseModel):
    post_id: str = Field(..., description="Post ID")
    content: str = Field(..., description="Post content")
    title: Optional[str] = Field("", description="Post title")


class PostEmbeddingBatchRequest(BaseModel):
    posts: List[PostEmbeddingRequest] = Field(..., description="List of posts")


class UserEmbeddingRequest(BaseModel):
    user_id: Optional[str] = Field(None, alias="userId", description="User ID")
    userId: Optional[str] = Field(None, description="User ID (camelCase from Java)")
    major: Optional[str] = Field(None, description="User's major")
    faculty: Optional[str] = Field(None, description="User's faculty")
    courses: Optional[List[str]] = Field([], description="List of courses")
    skills: Optional[List[str]] = Field([], description="List of skills")
    bio: Optional[str] = Field(None, description="User bio")
    
    @property
    def effective_user_id(self) -> str:
        """Get user_id from either field"""
        return self.user_id or self.userId or "unknown"
    
    class Config:
        populate_by_name = True


class EmbeddingResponse(BaseModel):
    id: str = Field(..., description="Post or User ID")
    embedding: List[float] = Field(..., description="Embedding vector")
    dimension: int = Field(..., description="Embedding dimension")


class BatchEmbeddingResponse(BaseModel):
    embeddings: List[EmbeddingResponse] = Field(..., description="List of embeddings")
    count: int = Field(..., description="Number of embeddings")


class SimilarityRequest(BaseModel):
    embedding1: List[float] = Field(..., description="First embedding")
    embedding2: List[float] = Field(..., description="Second embedding")


class SimilarityResponse(BaseModel):
    similarity: float = Field(..., description="Cosine similarity score")


class BatchSimilarityRequest(BaseModel):
    query_embedding: List[float] = Field(..., description="Query embedding")
    candidate_embeddings: List[List[float]] = Field(..., description="Candidate embeddings")


class BatchSimilarityResponse(BaseModel):
    similarities: List[float] = Field(..., description="Similarity scores")
    count: int = Field(..., description="Number of candidates")


# ==================== Core API Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "CTU Connect AI Inference Service",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if inference_engine else "degraded",
        "model_loaded": inference_engine is not None,
        "service": "CTU Connect AI Inference",
        "timestamp": datetime.now().isoformat()
    }


# ==================== Embedding Endpoints ====================

@app.post("/embed/post", response_model=EmbeddingResponse)
async def embed_post(request: PostEmbeddingRequest):
    """Generate embedding for a single post"""
    try:
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        embedding = inference_engine.encode_post(
            post_content=request.content,
            post_title=request.title
        )
        
        return EmbeddingResponse(
            id=request.post_id,
            embedding=embedding.tolist(),
            dimension=len(embedding)
        )
    except Exception as e:
        logger.error(f"Error generating post embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/post/batch", response_model=BatchEmbeddingResponse)
async def embed_post_batch(request: PostEmbeddingBatchRequest):
    """Generate embeddings for multiple posts"""
    try:
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        texts = [f"{post.title} {post.content}".strip() for post in request.posts]
        embeddings = inference_engine.encode_batch(texts)
        
        results = [
            EmbeddingResponse(
                id=post.post_id,
                embedding=emb.tolist(),
                dimension=len(emb)
            )
            for post, emb in zip(request.posts, embeddings)
        ]
        
        return BatchEmbeddingResponse(embeddings=results, count=len(results))
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/user", response_model=EmbeddingResponse)
async def embed_user(request: UserEmbeddingRequest):
    """Generate embedding for a user profile"""
    try:
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        user_data = {
            'major': request.major,
            'faculty': request.faculty,
            'courses': request.courses or [],
            'skills': request.skills or [],
            'bio': request.bio
        }
        
        embedding = inference_engine.encode_user_profile(user_data)
        
        return EmbeddingResponse(
            id=request.user_id,
            embedding=embedding.tolist(),
            dimension=len(embedding)
        )
    except Exception as e:
        logger.error(f"Error generating user embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Similarity Endpoints ====================

@app.post("/similarity", response_model=SimilarityResponse)
async def compute_similarity(request: SimilarityRequest):
    """Compute cosine similarity between two embeddings"""
    try:
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        emb1 = np.array(request.embedding1)
        emb2 = np.array(request.embedding2)
        similarity = inference_engine.compute_similarity(emb1, emb2)
        
        return SimilarityResponse(similarity=float(similarity))
    except Exception as e:
        logger.error(f"Error computing similarity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similarity/batch", response_model=BatchSimilarityResponse)
async def compute_batch_similarity(request: BatchSimilarityRequest):
    """Compute similarity between query and multiple candidates"""
    try:
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        query = np.array(request.query_embedding)
        candidates = np.array(request.candidate_embeddings)
        similarities = inference_engine.compute_batch_similarity(query, candidates)
        
        return BatchSimilarityResponse(
            similarities=similarities.tolist(),
            count=len(similarities)
        )
    except Exception as e:
        logger.error(f"Error computing batch similarity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Friend Recommendation Endpoints (NEW) ====================

class UserBatchEmbeddingRequest(BaseModel):
    """Request for batch user embedding generation"""
    users: List[UserEmbeddingRequest] = Field(..., description="List of users")


class FriendCandidateScore(BaseModel):
    """Additional scores for a friend candidate"""
    mutual_friends_score: float = Field(0.0, description="Mutual friends score (0-1)")
    academic_score: float = Field(0.0, description="Academic connection score (0-1)")
    activity_score: float = Field(0.0, description="Activity score (0-1)")
    recency_score: float = Field(0.0, description="Recency score (0-1)")
    mutual_friends_count: Optional[int] = Field(0, description="Number of mutual friends")


class UserProfileData(BaseModel):
    """User profile data for friend ranking - matches Java FriendRankingRequest.UserProfileData"""
    userId: Optional[str] = Field(None, description="User ID (camelCase from Java)")
    user_id: Optional[str] = Field(None, description="User ID (snake_case)")
    major: Optional[str] = Field(None, description="User's major")
    faculty: Optional[str] = Field(None, description="User's faculty")
    courses: Optional[List[str]] = Field([], description="List of courses")
    skills: Optional[List[str]] = Field([], description="List of skills")
    bio: Optional[str] = Field(None, description="User bio")
    
    @property
    def effective_user_id(self) -> str:
        """Get user_id from either field"""
        return self.userId or self.user_id or "unknown"
    
    class Config:
        populate_by_name = True


class FriendRankingRequest(BaseModel):
    """Request for friend candidate ranking - matches Java FriendRankingRequest"""
    currentUser: Optional[UserProfileData] = Field(None, alias="current_user", description="Current user profile")
    current_user: Optional[UserProfileData] = Field(None, description="Current user profile (snake_case)")
    candidates: List[UserProfileData] = Field(..., description="Candidate users")
    additionalScores: Optional[Dict[str, FriendCandidateScore]] = Field(
        None, alias="additional_scores", description="Additional scores per candidate user_id"
    )
    additional_scores: Optional[Dict[str, FriendCandidateScore]] = Field(
        None, description="Additional scores per candidate user_id (snake_case)"
    )
    topK: int = Field(20, alias="top_k", description="Number of top candidates to return")
    top_k: int = Field(20, description="Number of top candidates to return (snake_case)")
    
    @property
    def effective_current_user(self) -> Optional[UserProfileData]:
        return self.currentUser or self.current_user
    
    @property
    def effective_additional_scores(self) -> Optional[Dict[str, FriendCandidateScore]]:
        return self.additionalScores or self.additional_scores
    
    @property
    def effective_top_k(self) -> int:
        return self.topK or self.top_k or 20
    
    class Config:
        populate_by_name = True


class RankedFriend(BaseModel):
    """Ranked friend candidate result"""
    user_id: str
    final_score: float
    content_similarity: float
    mutual_friends_score: float
    academic_score: float
    activity_score: float
    recency_score: float
    suggestion_type: str
    suggestion_reason: Optional[str] = None


class FriendRankingResponse(BaseModel):
    """Response for friend ranking"""
    rankings: List[RankedFriend]
    count: int
    model_version: str = "phobert-v1"


@app.post("/embed/user/batch", response_model=BatchEmbeddingResponse)
async def embed_users_batch(request: UserBatchEmbeddingRequest):
    """Generate embeddings for multiple user profiles"""
    try:
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        results = []
        for user in request.users:
            user_data = {
                'major': user.major,
                'faculty': user.faculty,
                'courses': user.courses or [],
                'skills': user.skills or [],
                'bio': user.bio
            }
            
            embedding = inference_engine.encode_user_profile(user_data)
            
            results.append(EmbeddingResponse(
                id=user.user_id,
                embedding=embedding.tolist(),
                dimension=len(embedding)
            ))
        
        return BatchEmbeddingResponse(embeddings=results, count=len(results))
    
    except Exception as e:
        logger.error(f"Error generating batch user embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/friends/rank", response_model=FriendRankingResponse)
async def rank_friend_candidates(request: FriendRankingRequest):
    """
    Rank friend candidates using hybrid scoring
    
    Scoring weights:
    - Content Similarity (PhoBERT): 30%
    - Mutual Friends: 25%
    - Academic Connection: 20%
    - Activity Score: 15%
    - Recency: 10%
    """
    try:
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        from services.user_similarity_service import get_user_similarity_service
        
        # Get or create similarity service
        similarity_service = get_user_similarity_service(inference_engine)
        
        # Get current user from either field name
        current_user = request.effective_current_user
        if current_user is None:
            logger.error("No current user provided in request")
            raise HTTPException(status_code=400, detail="current_user or currentUser is required")
        
        # Generate current user embedding
        current_user_data = {
            'major': current_user.major,
            'faculty': current_user.faculty,
            'courses': current_user.courses or [],
            'skills': current_user.skills or [],
            'bio': current_user.bio
        }
        current_user_embedding = similarity_service.generate_user_embedding(current_user_data)
        
        # Generate candidate embeddings
        candidate_embeddings = []
        candidate_ids = []
        candidate_infos = {}
        
        for candidate in request.candidates:
            candidate_data = {
                'major': candidate.major,
                'faculty': candidate.faculty,
                'courses': candidate.courses or [],
                'skills': candidate.skills or [],
                'bio': candidate.bio
            }
            
            embedding = similarity_service.generate_user_embedding(candidate_data)
            candidate_embeddings.append(embedding)
            candidate_id = candidate.effective_user_id
            candidate_ids.append(candidate_id)
            
            # Store info for reason generation
            candidate_infos[candidate_id] = {
                'major_name': candidate.major,
                'faculty_name': candidate.faculty,
                'same_major': candidate.major == current_user.major if candidate.major and current_user.major else False,
                'same_faculty': candidate.faculty == current_user.faculty if candidate.faculty and current_user.faculty else False
            }
        
        # Convert additional scores to dict format
        additional_scores_dict = None
        additional_scores = request.effective_additional_scores
        if additional_scores:
            additional_scores_dict = {
                user_id: {
                    'mutual_friends_score': scores.mutual_friends_score,
                    'academic_score': scores.academic_score,
                    'activity_score': scores.activity_score,
                    'recency_score': scores.recency_score,
                    'mutual_friends_count': scores.mutual_friends_count
                }
                for user_id, scores in additional_scores.items()
            }
        
        # Rank candidates
        ranked_results = similarity_service.rank_friend_candidates(
            current_user_embedding=current_user_embedding,
            candidate_embeddings=candidate_embeddings,
            candidate_ids=candidate_ids,
            additional_scores=additional_scores_dict,
            top_k=request.effective_top_k
        )
        
        # Build response
        rankings = []
        for result in ranked_results:
            user_id = result['user_id']
            
            # Determine suggestion type and reason
            suggestion_type = similarity_service.determine_suggestion_type(result)
            
            # Get user info for reason generation
            user_info = candidate_infos.get(user_id, {})
            if additional_scores_dict and user_id in additional_scores_dict:
                user_info['mutual_friends_count'] = additional_scores_dict[user_id].get('mutual_friends_count', 0)
            
            scores_with_count = {**result}
            if additional_scores_dict and user_id in additional_scores_dict:
                scores_with_count['mutual_friends_count'] = additional_scores_dict[user_id].get('mutual_friends_count', 0)
            
            suggestion_reason = similarity_service.generate_suggestion_reason(
                scores=scores_with_count,
                user_info=user_info
            )
            
            rankings.append(RankedFriend(
                user_id=user_id,
                final_score=result['final_score'],
                content_similarity=result['content_similarity'],
                mutual_friends_score=result['mutual_friends_score'],
                academic_score=result['academic_score'],
                activity_score=result['activity_score'],
                recency_score=result['recency_score'],
                suggestion_type=suggestion_type,
                suggestion_reason=suggestion_reason
            ))
        
        logger.info(f"🤝 Ranked {len(rankings)} friend candidates for user {current_user.effective_user_id}")
        
        return FriendRankingResponse(
            rankings=rankings,
            count=len(rankings)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ranking friend candidates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similarity/users/batch", response_model=BatchSimilarityResponse)
async def compute_user_similarity_batch(request: BatchSimilarityRequest):
    """
    Compute similarity between a query user embedding and multiple candidate user embeddings
    Specialized endpoint for user-to-user similarity (same as /similarity/batch but semantic naming)
    """
    try:
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        query = np.array(request.query_embedding)
        candidates = np.array(request.candidate_embeddings)
        similarities = inference_engine.compute_batch_similarity(query, candidates)
        
        return BatchSimilarityResponse(
            similarities=similarities.tolist(),
            count=len(similarities)
        )
    except Exception as e:
        logger.error(f"Error computing user batch similarity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Academic Classification Endpoints ====================

class AcademicClassifyRequest(BaseModel):
    content: str = Field(..., description="Text content to classify")


class AcademicClassifyBatchRequest(BaseModel):
    contents: List[str] = Field(..., description="List of text contents to classify")


class AcademicClassifyResponse(BaseModel):
    is_academic: bool
    confidence: float
    label: str
    category: str
    probabilities: Optional[Dict[str, float]] = None
    method: str = "unknown"


@app.post("/classify/academic", response_model=AcademicClassifyResponse)
async def classify_academic(request: AcademicClassifyRequest):
    """
    Classify if content is academic using fine-tuned PhoBERT model
    Returns classification result with confidence score
    """
    try:
        from services.academic_classifier_service import get_academic_classifier
        
        classifier = get_academic_classifier()
        result = classifier.predict(request.content)
        
        return AcademicClassifyResponse(
            is_academic=result["is_academic"],
            confidence=result["confidence"],
            label=result["label"],
            category="academic" if result["is_academic"] else "general",
            probabilities=result.get("probabilities"),
            method="ml_classifier" if not result.get("fallback") else "heuristic"
        )
    except Exception as e:
        logger.error(f"Error classifying content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/academic/batch")
async def classify_academic_batch(request: AcademicClassifyBatchRequest):
    """
    Batch classify multiple contents as academic/non-academic
    """
    try:
        from services.academic_classifier_service import get_academic_classifier
        
        classifier = get_academic_classifier()
        results = classifier.batch_predict(request.contents)
        
        return {
            "results": results,
            "count": len(results),
            "classifier_ready": classifier.is_ready()
        }
    except Exception as e:
        logger.error(f"Error batch classifying content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classify/academic/info")
async def get_classifier_info():
    """Get information about the academic classifier"""
    try:
        from services.academic_classifier_service import get_academic_classifier
        
        classifier = get_academic_classifier()
        return classifier.get_info()
    except Exception as e:
        logger.error(f"Error getting classifier info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Include additional ML routes from api.routes if they exist
try:
    from api.routes import router as ml_router
    app.include_router(ml_router, prefix="/api")
    logger.info("✅ ML prediction routes loaded")
except ImportError:
    logger.warning("⚠️  ML prediction routes not found, using embedding-only mode")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

