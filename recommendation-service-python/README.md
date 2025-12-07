# CTU Connect Recommendation Service - Python ML Layer

Python FastAPI service providing ML-based recommendation predictions for CTU Connect social network.

## 📋 Features

- **PhoBERT Embeddings**: Vietnamese text embedding using PhoBERT
- **Academic Classification**: Classify posts as academic content
- **ML Ranking**: Multi-factor ranking algorithm
- **Hot Reload**: Reload models without service restart
- **Redis Caching**: Cache embeddings for performance
- **Kafka Integration**: Event streaming for training pipeline

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Redis (for caching)
- Kafka (for event streaming)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env

# Edit .env with your configuration
```

### Run Service

```bash
# Development mode
python app.py

# Production mode with Gunicorn
gunicorn app:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8097
```

### Docker

```bash
# Build image
docker build -t recommendation-ml-service .

# Run container
docker run -p 8097:8097 \
  -e REDIS_HOST=host.docker.internal \
  -e KAFKA_BOOTSTRAP_SERVERS=host.docker.internal:9092 \
  -v $(pwd)/academic_posts_model:/app/models \
  recommendation-ml-service
```

## 📡 API Endpoints

### POST /api/model/predict

Main prediction endpoint for ranking recommendations.

**Request:**
```json
{
  "userAcademic": {
    "userId": "user123",
    "major": "Computer Science",
    "faculty": "Engineering",
    "degree": "Bachelor",
    "batch": "K48"
  },
  "userHistory": [
    {
      "postId": "post1",
      "liked": 1,
      "commented": 0,
      "shared": 0,
      "viewDuration": 4.3
    }
  ],
  "candidatePosts": [
    {
      "postId": "post2",
      "content": "Hội thảo Machine Learning...",
      "hashtags": ["#AI", "#Workshop"],
      "authorMajor": "Computer Science",
      "likesCount": 10,
      "commentsCount": 5
    }
  ],
  "topK": 20
}
```

**Response:**
```json
{
  "rankedPosts": [
    {
      "postId": "post2",
      "score": 0.92,
      "contentSimilarity": 0.85,
      "implicitFeedback": 0.78,
      "academicScore": 0.95,
      "popularityScore": 0.45,
      "rank": 1
    }
  ],
  "modelVersion": "1.0.0",
  "processingTimeMs": 150,
  "timestamp": "2024-01-01T12:00:00"
}
```

### POST /api/model/embed

Generate embedding for text.

**Request:**
```json
{
  "text": "Hội thảo Machine Learning"
}
```

**Response:**
```json
{
  "embedding": [0.123, -0.456, ...],
  "dimension": 768,
  "timestamp": "2024-01-01T12:00:00"
}
```

### POST /api/model/classify/academic

Classify if content is academic.

**Response:**
```json
{
  "isAcademic": true,
  "confidence": 0.87,
  "category": "academic",
  "timestamp": "2024-01-01T12:00:00"
}
```

### GET /health

Health check endpoint.

### GET /api/model/info

Get model information.

### POST /api/model/reload

Reload models from disk (hot reload).

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

## 📊 Model Files

Pre-trained models should be placed in `academic_posts_model/`:

```
academic_posts_model/
├── vectorizer.pkl          # Text vectorizer
├── post_encoder.pkl        # Post content encoder
├── academic_encoder.pkl    # Academic profile encoder
└── ranking_model.pkl       # ML ranking model
```

## 🔧 Configuration

See `.env.example` for all configuration options.

Key settings:
- `MODEL_PATH`: Path to pre-trained models
- `REDIS_HOST`: Redis host for caching
- `KAFKA_BOOTSTRAP_SERVERS`: Kafka servers
- `WEIGHT_*`: Ranking weights (must match Java service)

## 🔄 Training Pipeline

### Data Collection

User interactions are sent to Kafka topics:
- `user_interaction`
- `post_viewed`
- `post_liked`
- `recommendation_training_data`

### Retraining

```bash
# Run training script
python training/train_model.py --input datasets/academic_dataset.json --output academic_posts_model/
```

### Hot Reload

After retraining, reload models without service restart:

```bash
curl -X POST http://localhost:8097/api/model/reload
```

## 📈 Performance

| Metric | Target | Typical |
|--------|--------|---------|
| Embedding generation | < 50ms | 30-40ms |
| Prediction (20 posts) | < 200ms | 150-180ms |
| Cache hit response | < 10ms | 5-8ms |

## 🔗 Integration with Java Service

Java service (`recommendation-service-java`) calls this Python service:

```java
// Java side
PythonModelRequest request = PythonModelRequest.builder()
    .userAcademic(profile)
    .candidatePosts(posts)
    .topK(20)
    .build();

PythonModelResponse response = pythonModelServiceClient.predict(request);
```

## 🐛 Troubleshooting

### Models not loading

Check that model files exist in `MODEL_PATH`:
```bash
ls -la academic_posts_model/
```

### PhoBERT download fails

```bash
# Download manually
python -c "from transformers import AutoModel; AutoModel.from_pretrained('vinai/phobert-base')"
```

### Redis connection error

Check Redis is running:
```bash
redis-cli ping
```

## 📝 Development

### Adding new features

1. Update `services/prediction_service.py`
2. Update `models/schemas.py` for request/response
3. Update `api/routes.py` for new endpoints
4. Add tests in `tests/`

### Code style

```bash
# Format code
black .

# Lint
flake8 .

# Type check
mypy .
```

## 📚 Resources

- [PhoBERT](https://github.com/VinAIResearch/PhoBERT)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Transformers](https://huggingface.co/docs/transformers/)

## 📞 Support

For issues or questions, check:
- `logs/python-service-*.log` - Application logs
- `/health` endpoint - Service status
- `/api/model/info` - Model information
