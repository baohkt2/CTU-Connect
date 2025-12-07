# 📖 CTU Connect Recommendation Service - Complete Index

## 🎯 Start Here

**New to the project?** → Read in this order:
1. [PROJECT_SUMMARY.md](./PROJECT_SUMMARY.md) - Visual overview & statistics
2. [QUICKSTART.md](./QUICKSTART.md) - Get up and running in 15 minutes
3. [README.md](./README.md) - Complete user guide
4. [ARCHITECTURE.md](./ARCHITECTURE.md) - Deep dive into technical details

---

## 📚 Documentation Files

### Main Documentation (61KB total)

| File | Size | Purpose | Audience |
|------|------|---------|----------|
| [README.md](./README.md) | 18KB | Complete user guide, API docs, setup | All users |
| [ARCHITECTURE.md](./ARCHITECTURE.md) | 16KB | Technical architecture, design patterns | Developers |
| [QUICKSTART.md](./QUICKSTART.md) | 11KB | Quick setup guide, test scenarios | New users |
| [PROJECT_SUMMARY.md](./PROJECT_SUMMARY.md) | 16KB | Visual overview, statistics | Management |
| [INDEX.md](./INDEX.md) | (this file) | Navigation guide | All users |

### Additional Documentation

| File | Purpose |
|------|---------|
| [../RECOMMENDATION_SERVICE_IMPLEMENTATION.md](../RECOMMENDATION_SERVICE_IMPLEMENTATION.md) | Implementation summary at project root |
| [database/init.sql](./database/init.sql) | Database initialization script |
| [.env.example](./.env.example) | Environment configuration template |

---

## 🏗️ Source Code Structure

### Directory Tree

```
recommendation-service-java/
├── 📁 src/main/java/vn/ctu/edu/recommend/
│   ├── 📄 RecommendationServiceApplication.java     [Main Application]
│   │
│   ├── 📁 config/                                   [Configuration]
│   │   ├── RedisConfig.java                         → Redis & caching setup
│   │   ├── WebClientConfig.java                     → HTTP client config
│   │   └── KafkaConfig.java                         → Kafka topics config
│   │
│   ├── 📁 controller/                               [REST API Layer]
│   │   └── RecommendationController.java            → API endpoints
│   │
│   ├── 📁 service/                                  [Business Logic]
│   │   ├── RecommendationService.java               → Service interface
│   │   └── impl/
│   │       └── RecommendationServiceImpl.java       → Main orchestration
│   │
│   ├── 📁 repository/                               [Data Access]
│   │   ├── postgres/
│   │   │   ├── PostEmbeddingRepository.java         → Posts + embeddings
│   │   │   ├── UserFeedbackRepository.java          → User feedback
│   │   │   └── RecommendationCacheRepository.java   → Cache management
│   │   ├── neo4j/
│   │   │   ├── UserGraphRepository.java             → Graph queries
│   │   │   └── PostGraphRepository.java             → Post relationships
│   │   └── redis/
│   │       └── RedisCacheService.java               → Redis operations
│   │
│   ├── 📁 model/                                    [Data Models]
│   │   ├── entity/
│   │   │   ├── postgres/
│   │   │   │   ├── PostEmbedding.java               → Post + vector
│   │   │   │   ├── UserFeedback.java                → Feedback records
│   │   │   │   └── RecommendationCache.java         → Cache entity
│   │   │   └── neo4j/
│   │   │       ├── UserNode.java                    → User graph node
│   │   │       ├── PostNode.java                    → Post graph node
│   │   │       └── GraphRelationship.java           → Relationship result
│   │   ├── dto/
│   │   │   ├── RecommendationRequest.java           → API request
│   │   │   ├── RecommendationResponse.java          → API response
│   │   │   ├── FeedbackRequest.java                 → Feedback request
│   │   │   ├── EmbeddingRequest.java                → NLP request
│   │   │   ├── EmbeddingResponse.java               → NLP response
│   │   │   ├── ClassificationRequest.java           → Classifier request
│   │   │   └── ClassificationResponse.java          → Classifier response
│   │   └── enums/
│   │       ├── FeedbackType.java                    → Feedback types
│   │       ├── AcademicCategory.java                → Content categories
│   │       └── RelationshipType.java                → Graph relationships
│   │
│   ├── 📁 nlp/                                      [AI/NLP Components]
│   │   ├── EmbeddingService.java                    → PhoBERT embeddings
│   │   └── AcademicClassifier.java                  → Content classification
│   │
│   ├── 📁 ranking/                                  [Ranking Algorithm]
│   │   └── RankingEngine.java                       → Score calculation
│   │
│   ├── 📁 kafka/                                    [Event Processing]
│   │   ├── consumer/
│   │   │   ├── PostEventConsumer.java               → Post events
│   │   │   └── UserActionConsumer.java              → User actions
│   │   └── event/
│   │       ├── PostEvent.java                       → Post event model
│   │       └── UserActionEvent.java                 → Action event model
│   │
│   ├── 📁 scheduler/                                [Batch Jobs]
│   │   └── RecommendationScheduler.java             → Scheduled tasks
│   │
│   └── 📁 exception/                                [Error Handling]
│       └── GlobalExceptionHandler.java              → Exception handler
│
├── 📁 src/main/resources/                           [Configuration Files]
│   ├── application.yml                              → Main config
│   └── application-docker.yml                       → Docker config
│
├── 📁 database/                                     [Database Scripts]
│   └── init.sql                                     → PostgreSQL init
│
├── 📄 pom.xml                                       [Maven Build]
├── 📄 Dockerfile                                    [Docker Image]
├── 📄 .env.example                                  [Environment Template]
├── 📄 .gitignore                                    [Git Ignore]
└── 📄 setup.ps1                                     [Setup Script]
```

---

## 🔑 Key Components Quick Reference

### 1. API Endpoints

| Method | Endpoint | Purpose | File |
|--------|----------|---------|------|
| GET | `/api/recommend/posts` | Get recommendations | RecommendationController |
| POST | `/api/recommend/posts` | Advanced recommendations | RecommendationController |
| POST | `/api/recommend/feedback` | Record feedback | RecommendationController |
| POST | `/api/recommend/embedding/rebuild` | Rebuild embeddings | RecommendationController |
| POST | `/api/recommend/rank/rebuild` | Rebuild cache | RecommendationController |
| DELETE | `/api/recommend/cache/{userId}` | Invalidate cache | RecommendationController |

### 2. Core Services

| Service | File | Purpose |
|---------|------|---------|
| Recommendation | RecommendationServiceImpl | Main orchestration |
| Embedding | EmbeddingService | PhoBERT integration |
| Classification | AcademicClassifier | Content classification |
| Ranking | RankingEngine | Score calculation |
| Cache | RedisCacheService | Redis operations |

### 3. Database Entities

| Entity | Database | File | Purpose |
|--------|----------|------|---------|
| PostEmbedding | PostgreSQL | PostEmbedding.java | Posts + vectors |
| UserFeedback | PostgreSQL | UserFeedback.java | Feedback history |
| RecommendationCache | PostgreSQL | RecommendationCache.java | Cached results |
| UserNode | Neo4j | UserNode.java | User graph |
| PostNode | Neo4j | PostNode.java | Post graph |

### 4. Kafka Topics

| Topic | Consumer | Purpose |
|-------|----------|---------|
| post_created | PostEventConsumer | New post processing |
| post_updated | PostEventConsumer | Post update processing |
| post_deleted | PostEventConsumer | Post deletion cleanup |
| user_action | UserActionConsumer | User interaction tracking |

---

## 🎯 Common Tasks Guide

### Task 1: Add New API Endpoint
1. Add method to `RecommendationController.java`
2. Add service method to `RecommendationService.java`
3. Implement in `RecommendationServiceImpl.java`
4. Update documentation in README.md

### Task 2: Modify Ranking Algorithm
1. Edit `RankingEngine.java` → `computeFinalScore()`
2. Update weights in `application.yml`
3. Test with different weight configurations
4. Document changes in ARCHITECTURE.md

### Task 3: Add New Event Consumer
1. Create event model in `kafka/event/`
2. Create consumer in `kafka/consumer/`
3. Add `@KafkaListener` annotation
4. Update `KafkaConfig.java` with new topic

### Task 4: Add New Database Entity
1. Create entity class in `model/entity/`
2. Create repository interface in `repository/`
3. Update `database/init.sql` if needed
4. Add migration script

### Task 5: Modify Caching Strategy
1. Edit `RedisCacheService.java`
2. Update TTL in `application.yml`
3. Test cache invalidation
4. Monitor cache hit ratio

---

## 🧪 Testing Guide

### Unit Tests Location
```
src/test/java/vn/ctu/edu/recommend/
├── service/
│   └── RecommendationServiceTest.java
├── nlp/
│   ├── EmbeddingServiceTest.java
│   └── AcademicClassifierTest.java
├── ranking/
│   └── RankingEngineTest.java
└── repository/
    └── PostEmbeddingRepositoryTest.java
```

### Running Tests
```bash
# All tests
mvn test

# Specific test class
mvn test -Dtest=RecommendationServiceTest

# Integration tests
mvn verify -P integration-tests
```

### Test Data
- Sample posts: `database/init.sql`
- Mock users: See QUICKSTART.md
- Test scenarios: See README.md

---

## 📦 Deployment Guide

### Development
```bash
.\setup.ps1              # Interactive setup
mvn spring-boot:run      # Run service
```

### Docker
```bash
docker build -t ctu-recommend .
docker run -p 8095:8095 ctu-recommend
```

### Production
```bash
# Build JAR
mvn clean package -DskipTests

# Run with profile
java -jar -Dspring.profiles.active=prod target/recommendation-service-1.0.0.jar
```

### Docker Compose
```bash
docker-compose up -d recommendation-service
```

---

## 🔧 Configuration Reference

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `POSTGRES_HOST` | localhost | PostgreSQL host |
| `POSTGRES_PORT` | 5435 | PostgreSQL port |
| `NEO4J_HOST` | localhost | Neo4j host |
| `NEO4J_PORT` | 7687 | Neo4j bolt port |
| `REDIS_HOST` | localhost | Redis host |
| `REDIS_PORT` | 6379 | Redis port |
| `KAFKA_BOOTSTRAP_SERVERS` | localhost:9092 | Kafka servers |
| `EUREKA_SERVER_URL` | http://localhost:8761/eureka/ | Eureka URL |
| `PHOBERT_SERVICE_URL` | http://localhost:8096 | NLP service |
| `SERVER_PORT` | 8095 | Service port |

### Application Properties

Key configurations in `application.yml`:
- `recommendation.weights.*` - Ranking weights (α, β, γ, δ)
- `recommendation.graph-weights.*` - Relationship weights
- `recommendation.cache.*` - Cache TTLs
- `recommendation.batch.*` - Batch job schedules

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution | Reference |
|-------|----------|-----------|
| PostgreSQL connection failed | Check if container running | QUICKSTART.md |
| pgvector extension not found | Run `CREATE EXTENSION vector;` | database/init.sql |
| Service won't start | Check Eureka availability | README.md |
| No recommendations returned | Insert test data | QUICKSTART.md |
| Kafka consumer not working | Check topic exists | README.md |
| Redis connection timeout | Increase timeout in config | application.yml |

### Logs Location
- Console: Standard output
- File: `logs/recommendation-service.log`
- Docker: `docker logs recommendation-service`

---

## 📊 Monitoring

### Health Checks
- Service: `http://localhost:8095/api/recommend/health`
- Actuator: `http://localhost:8095/actuator/health`
- Eureka: `http://localhost:8761/eureka/apps/RECOMMENDATION-SERVICE`

### Metrics
- Prometheus: `http://localhost:8095/actuator/prometheus`
- Grafana: Import dashboard from `monitoring/` folder
- Custom metrics: See README.md

---

## 🔄 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-07 | Initial release - Full implementation |

---

## 👥 Team & Support

**Project**: CTU Connect  
**Component**: Recommendation Service  
**Technology**: Spring Boot 3, Java 17  
**Architecture**: Microservices  

**Support Channels**:
- Technical Issues → Check TROUBLESHOOTING section in README.md
- Feature Requests → See ARCHITECTURE.md for extension points
- Setup Problems → Follow QUICKSTART.md step by step

---

## 🎓 Learning Resources

### Understanding the Code
1. Start with `RecommendationServiceApplication.java`
2. Read `RecommendationServiceImpl.java` for main flow
3. Study `RankingEngine.java` for algorithm
4. Review `EmbeddingService.java` for NLP integration

### Understanding the Architecture
1. Read ARCHITECTURE.md
2. Study the data flow diagrams
3. Review database schemas
4. Examine API specifications

### Extending the System
1. Add new ranking factors → Edit `RankingEngine.java`
2. Add new content types → Edit `AcademicCategory.java`
3. Add new events → Create in `kafka/event/`
4. Add new metrics → Update Actuator config

---

## ✅ Implementation Checklist

Use this checklist when setting up:

- [ ] Prerequisites installed (Java 17, Maven, Docker)
- [ ] PostgreSQL with pgvector running
- [ ] Neo4j database running
- [ ] Redis cache running
- [ ] Kafka broker running
- [ ] Eureka server running
- [ ] Environment variables configured
- [ ] Database initialized with schema
- [ ] Service built successfully
- [ ] Service starts without errors
- [ ] Health check returns UP
- [ ] Test API endpoints working
- [ ] Kafka consumers connecting
- [ ] Metrics endpoint accessible
- [ ] Documentation reviewed

---

## 🚀 Quick Links

**Documentation**:
- [README](./README.md) - Main documentation
- [Architecture](./ARCHITECTURE.md) - Technical details
- [Quick Start](./QUICKSTART.md) - Setup guide
- [Summary](./PROJECT_SUMMARY.md) - Overview

**External Resources**:
- Spring Boot Docs: https://spring.io/projects/spring-boot
- Neo4j Cypher: https://neo4j.com/docs/cypher-manual/
- pgvector: https://github.com/pgvector/pgvector
- PhoBERT: https://github.com/VinAIResearch/PhoBERT

**Project Links**:
- Main Project: `d:\LVTN\CTU-Connect-demo\`
- Service: `d:\LVTN\CTU-Connect-demo\recommendation-service-java\`
- Documentation: All *.md files in service directory

---

**Last Updated**: 2025-12-07  
**Version**: 1.0.0  
**Status**: ✅ Complete & Production-Ready

**Need help?** Start with QUICKSTART.md or check the relevant section above.
