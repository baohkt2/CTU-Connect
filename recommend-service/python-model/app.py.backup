"""
CTU Connect Recommendation Service - Python ML Layer
FastAPI service for ML-based recommendation predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import logging
import os
from datetime import datetime

from api.routes import router
from services.prediction_service import PredictionService
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/python-service-{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CTU Connect Recommendation ML Service",
    description="Python ML layer for recommendation system with NLP and ranking",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")

# Global prediction service instance
prediction_service = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global prediction_service
    logger.info("Starting Python ML Recommendation Service...")
    logger.info(f"Model path: {settings.MODEL_PATH}")
    logger.info(f"Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    
    try:
        # Initialize prediction service (loads models)
        prediction_service = PredictionService()
        logger.info("✅ Prediction service initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize prediction service: {e}")
        logger.warning("Service will start in fallback mode")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Python ML Recommendation Service...")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "CTU Connect Recommendation ML Service",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global prediction_service
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "python-ml-service",
        "components": {
            "api": "up",
            "prediction_service": "up" if prediction_service else "down",
            "models_loaded": False
        }
    }
    
    if prediction_service:
        health_status["components"]["models_loaded"] = prediction_service.is_ready()
    
    return health_status


@app.get("/metrics")
async def metrics():
    """Metrics endpoint for monitoring"""
    global prediction_service
    
    metrics_data = {
        "timestamp": datetime.now().isoformat(),
        "predictions_count": 0,
        "avg_latency_ms": 0,
        "cache_hit_rate": 0
    }
    
    if prediction_service:
        metrics_data.update(prediction_service.get_metrics())
    
    return metrics_data


if __name__ == "__main__":
    port = int(os.getenv("PORT", settings.PORT))
    logger.info(f"🚀 Starting server on port {port}")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=settings.DEBUG,
        log_level="info"
    )
