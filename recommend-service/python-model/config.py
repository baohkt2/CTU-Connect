"""
Configuration settings for Python ML Service
"""

import os
from pydantic_settings import BaseSettings
from typing import List
from datetime import datetime

class Settings(BaseSettings):
    """Application settings"""

    # Service configuration
    SERVICE_NAME: str = "recommendation-ml-service"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    PORT: int = int(os.getenv("PORT", "8097"))
    PYTHONIOENCODING: str = os.getenv("PYTHONIOENCODING", "utf-8")
    
    # Model paths
    MODEL_PATH: str = os.getenv("MODEL_PATH", "vinai/phobert-base")
    VECTORIZER_PATH: str = f"./model/academic_posts_model/vectorizer.pkl"
    POST_ENCODER_PATH: str = f"./model/academic_posts_model/post_encoder.pkl"
    ACADEMIC_ENCODER_PATH: str = f"./model/academic_posts_model/academic_encoder.pkl"
    RANKING_MODEL_PATH: str = f"./model/academic_posts_model/ranking_model.pkl"
    
    # Academic Classifier Model (fine-tuned PhoBERT)
    ACADEMIC_CLASSIFIER_MODEL_PATH: str = os.getenv(
        "ACADEMIC_CLASSIFIER_MODEL_PATH", 
        "./model/academic_posts_model"
    )
    USE_ML_ACADEMIC_CLASSIFIER: bool = os.getenv("USE_ML_ACADEMIC_CLASSIFIER", "True").lower() == "true"
    
    # PhoBERT configuration
    PHOBERT_MODEL_NAME: str = "vinai/phobert-base"
    EMBEDDING_DIMENSION: int = 768
    MAX_SEQUENCE_LENGTH: int = 256
    
    # Redis configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6380"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "recommend_redis_pass")
    REDIS_URL: str = os.getenv("REDIS_URL", f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0")
    REDIS_DB: int = 0
    REDIS_EMBEDDING_TTL: int = 3600  # 1 hour

    # Kafka configuration
    KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    KAFKA_CONSUMER_GROUP: str = "recommendation-ml-consumer"
    KAFKA_TRAINING_TOPIC: str = "recommendation_training_data"
    
    # Postgres configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://recommend_user:recommend_pass@localhost:5435/recommend_db")

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
