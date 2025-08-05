# CTU Connect Recommendation Service

A comprehensive AI-powered recommendation microservice for the CTU Connect social platform, featuring deep learning models, reinforcement learning, A/B testing, and real-time personalization.

## Features

### Core Capabilities
- **Deep Learning Personalization**: Uses PyTorch with PhoBERT for Vietnamese content understanding
- **Multi-head Attention**: Advanced attention mechanisms for user-post relevance scoring
- **Reinforcement Learning**: DQN-based continuous learning from user feedback
- **A/B Testing**: Built-in experimentation framework with multiple model variants
- **Real-time Processing**: Kafka-based streaming for instant recommendation updates
- **Caching**: Redis-powered caching for sub-100ms response times
- **Monitoring**: Prometheus metrics and comprehensive logging

### Technical Architecture
- **Framework**: FastAPI with async/await for high performance
- **Database**: PostgreSQL with async SQLAlchemy
- **Message Queue**: Apache Kafka for real-time data streaming
- **Caching**: Redis for recommendation and feature caching
- **ML Ops**: MLflow for experiment tracking and model versioning
- **Deployment**: Docker and Kubernetes with auto-scaling

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 12+
- Redis 6+
- Apache Kafka 2.8+

### Installation

1. **Clone and setup environment**:
```bash
cd recommendation-service
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure environment variables**:
```bash
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/recommendation_db"
export REDIS_URL="redis://localhost:6379"
export KAFKA_BOOTSTRAP_SERVERS="localhost:9092"
export SECRET_KEY="your-secret-key"
```

3. **Initialize database**:
```bash
python -c "
import asyncio
from db.models import create_tables
asyncio.run(create_tables())
"
```

4. **Start the service**:
```bash
python main.py
```

The service will be available at `http://localhost:8000`

## API Documentation

### Authentication
All endpoints require an API key in the Authorization header:
```
Authorization: Bearer your-api-key
```

### Endpoints

#### GET /health
Health check endpoint
```json
{
  "status": "healthy",
  "timestamp": "2025-08-05T10:30:00Z",
  "version": "1.0.0",
  "services": {
    "redis": "healthy",
    "recommendation_engine": "healthy"
  }
}
```

#### POST /recommendations
Get personalized recommendations for a user

**Request**:
```json
{
  "user_id": "user_123",
  "context": {
    "device_type": "mobile",
    "location": "Vietnam"
  },
  "k": 10,
  "include_explanations": true
}
```

**Response**:
```json
{
  "recommendations": [
    {
      "post_id": "post_456",
      "title": "Khoa học máy tính",
      "content": "Bài viết về AI...",
      "author_id": "author_789",
      "category": "Khoa Hoc",
      "tags": ["AI", "machine-learning"],
      "engagement_rate": 0.15,
      "relevance_score": 0.95,
      "rank": 1,
      "reason": "Recommended because you're interested in Khoa Hoc"
    }
  ],
  "ab_variant": "personalized_v1",
  "timestamp": "2025-08-05T10:30:00Z",
  "user_id": "user_123",
  "total_count": 10,
  "processing_time_ms": 45.2
}
```

#### POST /feedback
Record user interaction feedback

**Request**:
```json
{
  "user_id": "user_123",
  "post_id": "post_456",
  "feedback_type": "like",
  "context": {
    "device_type": "mobile",
    "session_id": "session_abc"
  }
}
```

**Response**:
```json
{
  "success": true,
  "message": "Feedback recorded successfully",
  "timestamp": "2025-08-05T10:30:00Z"
}
```

#### GET /recommendations/{user_id}/history
Get recommendation history for a user

**Response**:
```json
{
  "user_id": "user_123",
  "history": [
    {
      "timestamp": "2025-08-05T10:30:00Z",
      "post_ids": ["post_1", "post_2"],
      "model_version": "v1.0",
      "ab_test_variant": "personalized_v1",
      "served_count": 10,
      "clicked_count": 3,
      "ctr": 0.3
    }
  ],
  "total_count": 50
}
```

## Model Training

### Training Data Preparation
The system automatically collects training data from user interactions:
- **Positive samples**: Likes, comments, shares (weighted by engagement strength)
- **Negative samples**: Posts viewed but not interacted with
- **Features**: User profile, post content, temporal context

### Training Pipeline
```bash
# Start training
python core/training.py
```

The training pipeline includes:
1. **Data Collection**: Fetches recent interactions from database
2. **Feature Engineering**: Extracts user/post embeddings and context features
3. **Model Training**: Deep learning with attention mechanisms
4. **Evaluation**: Precision, recall, F1-score, diversity metrics
5. **Model Deployment**: Automatic model versioning and deployment

### Model Architecture
```
User Features (Profile + Context) 
    ↓
User Embedding (256d)
    ↓
Multi-Head Attention ← Post Embedding (256d)
    ↓                      ↑
Fusion Layer        Content Encoder (PhoBERT)
    ↓                      ↑
Prediction Head     Post Features + Metadata
    ↓
Relevance Score (0-1)
```

## A/B Testing

The service supports multiple recommendation variants:
- **personalized_v1**: Base deep learning model (40% traffic)
- **personalized_v2**: Enhanced with reinforcement learning (40% traffic)  
- **popularity_based**: Fallback algorithm (20% traffic)

Users are consistently assigned to variants based on user ID hash.

## Monitoring & Analytics

### Prometheus Metrics
- `recommendation_requests_total`: Total API requests
- `recommendations_served_total`: Recommendations served by variant
- `feedback_received_total`: User feedback by type
- `model_accuracy`: Current model performance
- `cache_hits_total` / `cache_misses_total`: Caching performance

### Dashboards
Access metrics at `/metrics` endpoint for Prometheus scraping.

### Logging
Structured JSON logging with:
- Request/response tracking
- Model prediction details
- Error tracking and debugging
- Performance monitoring

## Deployment

### Docker
```bash
# Build image
docker build -t ctu-connect/recommendation-service .

# Run container
docker run -p 8000:8000 \
  -e DATABASE_URL="postgresql+asyncpg://..." \
  -e REDIS_URL="redis://..." \
  ctu-connect/recommendation-service
```

### Kubernetes
```bash
# Deploy to Kubernetes
kubectl apply -f deployment/kubernetes.yaml

# Check status
kubectl get pods -l app=recommendation-service
kubectl logs -f deployment/recommendation-service
```

The Kubernetes deployment includes:
- **Auto-scaling**: 2-10 replicas based on CPU/memory usage
- **Health checks**: Liveness and readiness probes
- **Resource limits**: Memory and CPU constraints
- **Config management**: ConfigMaps and Secrets

### Production Configuration
For production deployment:

1. **Database**: Use managed PostgreSQL (AWS RDS, Google Cloud SQL)
2. **Redis**: Use managed Redis cluster
3. **Kafka**: Use managed Kafka service (Confluent, AWS MSK)
4. **Monitoring**: Set up Prometheus + Grafana
5. **Security**: Enable API authentication, rate limiting
6. **SSL/TLS**: Configure HTTPS termination

## Performance

### Benchmarks
- **Response Time**: < 100ms for cached recommendations
- **Throughput**: 1000+ RPS per instance
- **Accuracy**: 85%+ precision on validation set
- **Diversity**: 70%+ category diversity in recommendations

### Optimization Tips
1. **Caching Strategy**: Cache user features and popular recommendations
2. **Batch Processing**: Group similar requests for efficient inference
3. **Model Optimization**: Use TorchScript or ONNX for faster inference
4. **Database**: Optimize queries with proper indexing
5. **Scaling**: Use horizontal pod autoscaling in Kubernetes

## Testing

### Run Tests
```bash
# Unit tests
pytest tests/ -v

# Integration tests
pytest tests/ -v -m integration

# Load tests
pytest tests/test_recommendation.py::TestPerformance -v
```

### Test Coverage
- **API Endpoints**: Complete request/response testing
- **Model Components**: Unit tests for all neural network layers
- **Data Processing**: Feature extraction and transformation tests
- **Integration**: End-to-end recommendation flow tests

## Contributing

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Install dev dependencies: `pip install -r requirements.txt`
4. Run tests: `pytest`
5. Submit pull request

### Code Standards
- **Python**: Follow PEP 8, use Black formatter
- **Documentation**: Add docstrings for all functions
- **Testing**: Maintain >90% test coverage
- **Type Hints**: Use type annotations throughout

## Troubleshooting

### Common Issues

**Service won't start**:
- Check database connection string
- Verify Redis is running
- Ensure Kafka is accessible

**Poor recommendation quality**:
- Check training data quality and quantity
- Verify user interaction tracking
- Review model hyperparameters

**High response times**:
- Check Redis cache hit rates
- Monitor database query performance
- Review model inference time

**Memory issues**:
- Adjust batch sizes in training
- Optimize model architecture
- Configure appropriate resource limits

### Debug Mode
Enable debug mode for detailed logging:
```bash
export DEBUG=true
python main.py
```

### Logs
Check logs for debugging:
```bash
# Docker
docker logs <container-id>

# Kubernetes
kubectl logs -f deployment/recommendation-service
```

## License

MIT License - see LICENSE file for details.

## Support

For questions and support:
- GitHub Issues: Create issue in repository
- Documentation: Check API documentation at `/docs`
- Monitoring: Check service health at `/health`
