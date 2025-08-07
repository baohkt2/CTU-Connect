import os
from typing import Dict, Any

class Config:
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # Database Configuration - Sử dụng database chung với hệ thống
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:password@auth_db:5432/auth_db"
    )

    # Redis Configuration - Riêng cho recommendation service
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://recommendation-redis:6379")
    REDIS_TTL: int = int(os.getenv("REDIS_TTL", "3600"))  # 1 hour

    # Kafka Configuration
    KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    KAFKA_TOPIC_USER_INTERACTIONS: str = "user_interactions"
    KAFKA_TOPIC_POST_UPDATES: str = "post_updates"
    KAFKA_GROUP_ID: str = "recommendation_service"

    # Eureka Service Discovery
    EUREKA_SERVER_URL: str = os.getenv("EUREKA_CLIENT_SERVICE_URL_DEFAULTZONE", "http://eureka-server:8761/eureka")
    SERVICE_NAME: str = "recommendation-service"
    SERVICE_ID: str = f"{SERVICE_NAME}-{PORT}"

    # API Gateway Integration
    API_GATEWAY_URL: str = os.getenv("API_GATEWAY_URL", "http://api-gateway:8090")

    # Microservices URLs
    USER_SERVICE_URL: str = "http://user-service:8081"
    POST_SERVICE_URL: str = "http://post-service:8085"
    AUTH_SERVICE_URL: str = "http://auth-service:8080"

    # Model Configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", "./models/recommendation_model.pt")
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "256"))
    NUM_HEADS: int = int(os.getenv("NUM_HEADS", "8"))

    # Recommendation Settings
    TOP_K_RECOMMENDATIONS: int = int(os.getenv("TOP_K_RECOMMENDATIONS", "10"))
    DIVERSITY_THRESHOLD: float = float(os.getenv("DIVERSITY_THRESHOLD", "0.7"))

    # A/B Testing
    AB_TEST_VARIANTS: Dict[str, float] = {
        "personalized_v1": 0.4,
        "personalized_v2": 0.4,
        "popularity_based": 0.2
    }

    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    MLFLOW_EXPERIMENT_NAME: str = "recommendation_experiments"

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "recommendation-secret-key")
    API_KEY_HEADER: str = "X-API-Key"

    # Performance
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))

    # Health Check
    HEALTH_CHECK_INTERVAL: int = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))

config = Config()
