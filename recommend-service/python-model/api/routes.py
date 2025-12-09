"""
API routes for recommendation ML service
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from services.prediction_service import PredictionService
from models.schemas import PredictionRequest, PredictionResponse, RankedPost

logger = logging.getLogger(__name__)

router = APIRouter()

# Global singleton instance
_prediction_service_instance = None

# Dependency to get prediction service
def get_prediction_service() -> PredictionService:
    """Get prediction service instance - singleton pattern"""
    global _prediction_service_instance
    
    if _prediction_service_instance is None:
        try:
            logger.info("Initializing prediction service singleton...")
            _prediction_service_instance = PredictionService()
            logger.info("✅ Prediction service initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize prediction service: {e}")
            raise HTTPException(status_code=503, detail=f"Prediction service not available: {str(e)}")
    
    return _prediction_service_instance


@router.post("/model/predict", response_model=PredictionResponse)
async def predict_recommendations(
    request: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Main prediction endpoint for recommendation ranking
    
    **Input:**
    - userAcademic: User's academic profile (major, faculty, degree, batch)
    - userHistory: User's interaction history
    - candidatePosts: List of candidate posts to rank
    - topK: Number of top recommendations to return
    
    **Output:**
    - rankedPosts: Ranked list of posts with scores
    - modelVersion: Version of the model used
    - processingTimeMs: Processing time in milliseconds
    """
    start_time = datetime.now()

    try:
        logger.info(f"🎯 Prediction request for user: {request.userAcademic.userId}, candidates: {len(request.candidatePosts)}")
        logger.debug(f"   User academic: major={request.userAcademic.major}, faculty={request.userAcademic.faculty}")
        logger.debug(f"   User history: {len(request.userHistory)} interactions")
        logger.debug(f"   TopK requested: {request.topK}")
        
        # Log first candidate for debugging
        if request.candidatePosts:
            first_post = request.candidatePosts[0]
            logger.debug(f"   Sample post: id={first_post.postId}, contentLength={len(first_post.content) if first_post.content else 0}, likes={first_post.likeCount}")
        
        # Log first history item if present
        if request.userHistory:
            first_history = request.userHistory[0]
            logger.debug(f"   Sample history: postId={first_history.postId}, liked={first_history.liked}, timestamp={first_history.timestamp}")

        # Validate request
        if not request.candidatePosts:
            logger.error("❌ No candidate posts provided")
            raise HTTPException(status_code=400, detail="No candidate posts provided")

        if request.topK <= 0 or request.topK > 100:
            logger.error(f"❌ Invalid topK: {request.topK}")
            raise HTTPException(status_code=400, detail="topK must be between 1 and 100")
        
        # Get predictions from service
        ranked_posts = await service.predict(
            user_academic=request.userAcademic.model_dump(),
            user_history=[h.model_dump() for h in request.userHistory],
            candidate_posts=[p.model_dump() for p in request.candidatePosts],
            top_k=request.topK
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"Prediction completed: {len(ranked_posts)} posts ranked in {processing_time:.2f}ms")
        
        return PredictionResponse(
            rankedPosts=ranked_posts,
            modelVersion=service.get_model_version(),
            processingTimeMs=int(processing_time),
            timestamp=datetime.now().isoformat()
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal prediction error")


@router.post("/model/embed")
async def embed_text(
    text: str,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Generate embedding for text content
    
    **Input:**
    - text: Text to embed
    
    **Output:**
    - embedding: 768-dimensional vector
    - dimension: Embedding dimension
    """
    try:
        embedding = await service.generate_embedding(text)
        
        return {
            "embedding": embedding.tolist(),
            "dimension": len(embedding),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Embedding error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Embedding generation failed")


@router.post("/model/classify/academic")
async def classify_academic(
    content: str,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Classify if content is academic
    
    **Input:**
    - content: Post content to classify
    
    **Output:**
    - isAcademic: Boolean
    - confidence: Confidence score
    - category: Academic category if applicable
    """
    try:
        result = await service.classify_academic(content)
        
        return {
            "isAcademic": result["is_academic"],
            "confidence": result["confidence"],
            "category": result.get("category"),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Classification error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Classification failed")


@router.get("/model/info")
async def get_model_info(service: PredictionService = Depends(get_prediction_service)):
    """Get model information"""
    return {
        "modelVersion": service.get_model_version(),
        "embeddingDimension": service.embedding_dimension,
        "modelPath": service.model_path,
        "isReady": service.is_ready(),
        "academicCategories": service.academic_categories,
        "timestamp": datetime.now().isoformat()
    }


@router.post("/model/reload")
async def reload_models(service: PredictionService = Depends(get_prediction_service)):
    """
    Reload models from disk (for hot reload after retraining)
    """
    try:
        await service.reload_models()
        return {
            "status": "success",
            "message": "Models reloaded successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Model reload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Model reload failed")
