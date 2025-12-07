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

# Dependency to get prediction service
def get_prediction_service() -> PredictionService:
    """Get prediction service instance"""
    from app import prediction_service
    if not prediction_service:
        raise HTTPException(status_code=503, detail="Prediction service not initialized")
    return prediction_service


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
        logger.info(f"Prediction request for user: {request.userAcademic.userId}, candidates: {len(request.candidatePosts)}")
        
        # Validate request
        if not request.candidatePosts:
            raise HTTPException(status_code=400, detail="No candidate posts provided")
        
        if request.topK <= 0 or request.topK > 100:
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
