"""
Configuration settings for Python ML Service
"""

import os
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings"""
    
    # Service configuration
    SERVICE_NAME: str = "recommendation-ml-service"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    PORT: int = int(os.getenv("PORT", "8097"))
    
    # Model paths
    MODEL_PATH: str = os.getenv("MODEL_PATH", "./academic_posts_model")
    VECTORIZER_PATH: str = f"{MODEL_PATH}/vectorizer.pkl"
    POST_ENCODER_PATH: str = f"{MODEL_PATH}/post_encoder.pkl"
    ACADEMIC_ENCODER_PATH: str = f"{MODEL_PATH}/academic_encoder.pkl"
    RANKING_MODEL_PATH: str = f"{MODEL_PATH}/ranking_model.pkl"
    
    # PhoBERT configuration
    PHOBERT_MODEL_NAME: str = "vinai/phobert-base"
    EMBEDDING_DIMENSION: int = 768
    MAX_SEQUENCE_LENGTH: int = 256
    
    # Redis configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    REDIS_DB: int = 0
    REDIS_EMBEDDING_TTL: int = 3600  # 1 hour
    
    # Kafka configuration
    KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    KAFKA_CONSUMER_GROUP: str = "recommendation-ml-consumer"
    KAFKA_TRAINING_TOPIC: str = "recommendation_training_data"
    
    # Ranking weights (matching Java configuration)
    WEIGHT_CONTENT_SIMILARITY: float = 0.35
    WEIGHT_IMPLICIT_FEEDBACK: float = 0.30
    WEIGHT_ACADEMIC_SCORE: float = 0.25
    WEIGHT_POPULARITY: float = 0.10
    
    # Academic categories
    ACADEMIC_CATEGORIES: List[str] = [
        "research",
        "scholarship",
        "qa",
        "announcement",
        "event",
        "academic_discussion"
    ]
    
    # Model thresholds
    ACADEMIC_CONFIDENCE_THRESHOLD: float = 0.6
    MIN_SIMILARITY_SCORE: float = 0.1
    
    # Performance settings
    BATCH_SIZE: int = 32
    MAX_CANDIDATES: int = 500
    DEFAULT_TOP_K: int = 20
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: str = "./logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
