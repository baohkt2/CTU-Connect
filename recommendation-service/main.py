from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import asyncio
import logging
from datetime import datetime
import uvicorn
from contextlib import asynccontextmanager

from core.recommendation_engine import RecommendationEngine
from db.models import get_db_session, create_tables, AsyncSession
from config.settings import config
from api.monitoring import setup_monitoring, metrics
from api.eureka_client import eureka_client
from api.microservice_client import microservice_client
import structlog

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger()

# Pydantic models for API requests/responses
class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User ID requesting recommendations")
    context: Dict[str, Any] = Field(default_factory=dict, description="Request context (device, location, etc.)")
    k: Optional[int] = Field(default=None, description="Number of recommendations (default: 10)")
    include_explanations: bool = Field(default=True, description="Include recommendation explanations")

class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    ab_variant: str
    timestamp: str
    user_id: str
    total_count: int
    processing_time_ms: float

class FeedbackRequest(BaseModel):
    user_id: str = Field(..., description="User ID providing feedback")
    post_id: str = Field(..., description="Post ID being interacted with")
    feedback_type: str = Field(..., description="Type of feedback: click, like, comment, share, skip, dislike")
    context: Dict[str, Any] = Field(default_factory=dict, description="Interaction context")
    session_id: Optional[str] = Field(default=None, description="Session identifier")

class FeedbackResponse(BaseModel):
    success: bool
    message: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    services: Dict[str, str]

# Global recommendation engine instance
recommendation_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with Eureka integration"""
    global recommendation_engine

    # Startup
    logger.info("Starting CTU Connect Recommendation Service...")

    try:
        # Initialize Eureka client
        await eureka_client.initialize()
        logger.info("Eureka client initialized")

        # Initialize microservice client
        await microservice_client.initialize()
        logger.info("Microservice client initialized")

        # Register with Eureka
        if await eureka_client.register():
            logger.info("Successfully registered with Eureka server")
        else:
            logger.warning("Failed to register with Eureka, continuing without service discovery")

        # Create database tables
        await create_tables()
        logger.info("Database tables created/verified")

        # Initialize recommendation engine
        recommendation_engine = RecommendationEngine()
        recommendation_engine.microservice_client = microservice_client
        await recommendation_engine.initialize()
        logger.info("Recommendation engine initialized")

        # Setup monitoring
        setup_monitoring(app)
        logger.info("Monitoring setup completed")

        yield

    except Exception as e:
        logger.error(f"Failed to start recommendation service: {e}")
        raise

    # Shutdown
    logger.info("Shutting down CTU Connect Recommendation Service...")

    # Cleanup
    if eureka_client:
        await eureka_client.close()
    if microservice_client:
        await microservice_client.close()
    if recommendation_engine and recommendation_engine.redis_client:
        await recommendation_engine.redis_client.close()

# Create FastAPI application
app = FastAPI(
    title="CTU Connect Recommendation Service",
    description="AI-powered personalized post recommendation system for CTU Connect",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for CTU Connect frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # CTU Connect frontends
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key or JWT token"""
    token = credentials.credentials

    # Try to validate with Auth Service first
    if microservice_client:
        user_info = await microservice_client.validate_user_token(token)
        if user_info:
            return user_info

    # Fallback to API key validation
    if config.SECRET_KEY and token != config.SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid authentication")

    return {"authenticated": True}

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests for monitoring"""
    start_time = datetime.utcnow()

    response = await call_next(request)

    processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

    logger.info(
        "Request processed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        processing_time_ms=processing_time
    )

    # Update metrics
    metrics.request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code
    ).inc()

    metrics.request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(processing_time / 1000)

    return response

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint compatible with Spring Boot actuator"""
    services_status = {}

    # Check Redis connection
    try:
        if recommendation_engine and recommendation_engine.redis_client:
            await recommendation_engine.redis_client.ping()
            services_status["redis"] = "UP"
        else:
            services_status["redis"] = "DOWN"
    except Exception:
        services_status["redis"] = "DOWN"

    # Check recommendation engine
    if recommendation_engine and recommendation_engine.is_initialized:
        services_status["recommendation_engine"] = "UP"
    else:
        services_status["recommendation_engine"] = "DOWN"

    # Check Eureka connection
    try:
        if eureka_client.registered:
            services_status["eureka"] = "UP"
        else:
            services_status["eureka"] = "DOWN"
    except Exception:
        services_status["eureka"] = "DOWN"

    # Check microservice connectivity
    try:
        if microservice_client.session:
            services_status["microservices"] = "UP"
        else:
            services_status["microservices"] = "DOWN"
    except Exception:
        services_status["microservices"] = "DOWN"

    overall_status = "UP" if all(
        status == "UP" for status in services_status.values()
    ) else "DOWN"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        services=services_status
    )

@app.post("/api/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    db_session: AsyncSession = Depends(get_db_session),
    auth_info: dict = Depends(verify_api_key)
):
    """Get personalized recommendations for a user (API Gateway compatible path)"""
    start_time = datetime.utcnow()

    try:
        if not recommendation_engine or not recommendation_engine.is_initialized:
            raise HTTPException(status_code=503, detail="Recommendation engine not available")

        # Get enhanced user context from User Service
        if microservice_client and 'user_id' in str(auth_info):
            user_profile = await microservice_client.get_user_profile(request.user_id)
            if user_profile:
                request.context.update({
                    "user_interests": user_profile.get("interests", {}),
                    "user_activity": user_profile.get("activityPattern", {}),
                    "user_demographics": user_profile.get("demographics", {})
                })

        # Get recommendations
        result = await recommendation_engine.get_recommendations(
            user_id=request.user_id,
            context=request.context,
            db_session=db_session,
            k=request.k
        )

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Update metrics
        metrics.recommendations_served.labels(
            variant=result.get("ab_variant", "unknown")
        ).inc()

        return RecommendationResponse(
            recommendations=result["recommendations"],
            ab_variant=result["ab_variant"],
            timestamp=result["timestamp"],
            user_id=result["user_id"],
            total_count=len(result["recommendations"]),
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Error getting recommendations for user {request.user_id}: {e}")
        metrics.error_count.labels(endpoint="/api/recommendations").inc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/feedback", response_model=FeedbackResponse)
async def record_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
    db_session: AsyncSession = Depends(get_db_session),
    auth_info: dict = Depends(verify_api_key)
):
    """Record user feedback for reinforcement learning (API Gateway compatible path)"""
    try:
        if not recommendation_engine or not recommendation_engine.is_initialized:
            raise HTTPException(status_code=503, detail="Recommendation engine not available")

        # Record feedback asynchronously
        background_tasks.add_task(
            recommendation_engine.record_feedback,
            request.user_id,
            request.post_id,
            request.feedback_type,
            request.context,
            db_session
        )

        # Send interaction event to other microservices
        if microservice_client:
            background_tasks.add_task(
                microservice_client.send_interaction_event,
                request.user_id,
                request.post_id,
                request.feedback_type,
                request.context
            )

        # Update metrics
        metrics.feedback_received.labels(
            feedback_type=request.feedback_type
        ).inc()

        return FeedbackResponse(
            success=True,
            message="Feedback recorded successfully",
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        metrics.error_count.labels(endpoint="/api/feedback").inc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Service discovery endpoints
@app.get("/api/services")
async def get_registered_services(auth_info: dict = Depends(verify_api_key)):
    """Get all registered services from Eureka"""
    try:
        services = await eureka_client.get_all_services()
        return {
            "services": services,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting services: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/services/{service_name}")
async def get_service_instances(service_name: str, auth_info: dict = Depends(verify_api_key)):
    """Get instances of a specific service"""
    try:
        instances = await eureka_client.discover_service(service_name)
        return {
            "service_name": service_name,
            "instances": instances or [],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting service instances: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Prometheus metrics endpoint
@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi import Response

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        workers=1  # Use 1 worker for development, scale in production
    )
