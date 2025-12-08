"""
FastAPI Server for PhoBERT Inference
Provides REST API endpoints for embedding generation
"""

import sys
import os

# Fix encoding for Windows console (only if not already wrapped)
if sys.platform == 'win32':
    import io
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        except (AttributeError, ValueError):
            pass  # Already wrapped or can't wrap
    
    if not isinstance(sys.stderr, io.TextIOWrapper) or sys.stderr.encoding != 'utf-8':
        try:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        except (AttributeError, ValueError):
            pass  # Already wrapped or can't wrap

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import logging
from inference import get_inference_engine

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CTU Connect Recommendation - Inference API",
    description="PhoBERT-based embedding generation for posts and users",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize inference engine
inference_engine = None


@app.on_event("startup")
async def startup_event():
    """Initialize inference engine on startup"""
    global inference_engine
    try:
        logger.info("Loading PhoBERT model...")
        inference_engine = get_inference_engine()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


# Request/Response Models
class PostEmbeddingRequest(BaseModel):
    """Request model for post embedding"""
    post_id: str = Field(..., description="Post ID")
    content: str = Field(..., description="Post content")
    title: Optional[str] = Field("", description="Post title")


class PostEmbeddingBatchRequest(BaseModel):
    """Request model for batch post embedding"""
    posts: List[PostEmbeddingRequest] = Field(..., description="List of posts")


class UserEmbeddingRequest(BaseModel):
    """Request model for user embedding"""
    user_id: str = Field(..., description="User ID")
    major: Optional[str] = Field(None, description="User's major")
    faculty: Optional[str] = Field(None, description="User's faculty")
    courses: Optional[List[str]] = Field([], description="List of courses")
    skills: Optional[List[str]] = Field([], description="List of skills")
    bio: Optional[str] = Field(None, description="User bio")


class EmbeddingResponse(BaseModel):
    """Response model for single embedding"""
    id: str = Field(..., description="Post or User ID")
    embedding: List[float] = Field(..., description="Embedding vector")
    dimension: int = Field(..., description="Embedding dimension")


class BatchEmbeddingResponse(BaseModel):
    """Response model for batch embeddings"""
    embeddings: List[EmbeddingResponse] = Field(..., description="List of embeddings")
    count: int = Field(..., description="Number of embeddings")


class SimilarityRequest(BaseModel):
    """Request model for similarity computation"""
    embedding1: List[float] = Field(..., description="First embedding")
    embedding2: List[float] = Field(..., description="Second embedding")


class SimilarityResponse(BaseModel):
    """Response model for similarity"""
    similarity: float = Field(..., description="Cosine similarity score")


class BatchSimilarityRequest(BaseModel):
    """Request model for batch similarity"""
    query_embedding: List[float] = Field(..., description="Query embedding")
    candidate_embeddings: List[List[float]] = Field(..., description="Candidate embeddings")


class BatchSimilarityResponse(BaseModel):
    """Response model for batch similarity"""
    similarities: List[float] = Field(..., description="Similarity scores")
    count: int = Field(..., description="Number of candidates")


# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": inference_engine is not None,
        "service": "CTU Connect Recommendation Inference"
    }


@app.post("/embed/post", response_model=EmbeddingResponse)
async def embed_post(request: PostEmbeddingRequest):
    """
    Generate embedding for a single post
    """
    try:
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Generate embedding
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
    """
    Generate embeddings for multiple posts
    """
    try:
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Prepare texts
        texts = [
            f"{post.title} {post.content}".strip()
            for post in request.posts
        ]
        
        # Generate embeddings
        embeddings = inference_engine.encode_batch(texts)
        
        # Format response
        results = [
            EmbeddingResponse(
                id=post.post_id,
                embedding=emb.tolist(),
                dimension=len(emb)
            )
            for post, emb in zip(request.posts, embeddings)
        ]
        
        return BatchEmbeddingResponse(
            embeddings=results,
            count=len(results)
        )
    
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/user", response_model=EmbeddingResponse)
async def embed_user(request: UserEmbeddingRequest):
    """
    Generate embedding for a user profile
    """
    try:
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Prepare user data
        user_data = {
            'major': request.major,
            'faculty': request.faculty,
            'courses': request.courses or [],
            'skills': request.skills or [],
            'bio': request.bio
        }
        
        # Generate embedding
        embedding = inference_engine.encode_user_profile(user_data)
        
        return EmbeddingResponse(
            id=request.user_id,
            embedding=embedding.tolist(),
            dimension=len(embedding)
        )
    
    except Exception as e:
        logger.error(f"Error generating user embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similarity", response_model=SimilarityResponse)
async def compute_similarity(request: SimilarityRequest):
    """
    Compute cosine similarity between two embeddings
    """
    try:
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Convert to numpy arrays
        emb1 = np.array(request.embedding1)
        emb2 = np.array(request.embedding2)
        
        # Compute similarity
        similarity = inference_engine.compute_similarity(emb1, emb2)
        
        return SimilarityResponse(similarity=similarity)
    
    except Exception as e:
        logger.error(f"Error computing similarity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similarity/batch", response_model=BatchSimilarityResponse)
async def compute_batch_similarity(request: BatchSimilarityRequest):
    """
    Compute similarity between query and multiple candidates
    """
    try:
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Convert to numpy arrays
        query = np.array(request.query_embedding)
        candidates = np.array(request.candidate_embeddings)
        
        # Compute similarities
        similarities = inference_engine.compute_batch_similarity(query, candidates)
        
        return BatchSimilarityResponse(
            similarities=similarities.tolist(),
            count=len(similarities)
        )
    
    except Exception as e:
        logger.error(f"Error computing batch similarity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
