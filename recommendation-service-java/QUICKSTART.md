# ⚡ Quick Start Guide - CTU Connect Recommendation Service

## Prerequisites Check

```bash
# Check Java version (need 17+)
java -version

# Check Maven
mvn -version

# Check Docker
docker --version
docker-compose --version
```

## 🚀 Option 1: Quick Start with Docker

### Step 1: Start Infrastructure

```bash
# Navigate to project root
cd d:\LVTN\CTU-Connect-demo

# Start PostgreSQL with pgvector
docker run -d --name recommend_db \
  -p 5435:5432 \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=recommendation_db \
  ankane/pgvector:latest

# Start Neo4j
docker run -d --name neo4j-recommend \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5.13.0

# Start Redis
docker run -d --name redis-recommend \
  -p 6379:6379 \
  redis:7-alpine

# Verify all running
docker ps
```

### Step 2: Initialize Database

```bash
# Connect to PostgreSQL
docker exec -it recommend_db psql -U postgres -d recommendation_db

# Run initialization script
\i /path/to/database/init.sql

# Exit
\q
```

### Step 3: Build and Run Service

```bash
cd recommendation-service-java

# Copy environment config
cp .env.example .env

# Build with Maven
mvn clean package -DskipTests

# Run the service
java -jar target/recommendation-service-1.0.0-SNAPSHOT.jar

# Or use Maven Spring Boot plugin
mvn spring-boot:run
```

### Step 4: Verify Service

```bash
# Check health
curl http://localhost:8095/api/recommend/health

# Check actuator
curl http://localhost:8095/actuator/health

# Expected response:
{
  "status": "UP",
  "service": "recommendation-service",
  "timestamp": "2025-12-07T14:30:00"
}
```

## 🏃 Option 2: Full Docker Compose

### Step 1: Add to docker-compose.yml

```yaml
# Add these services to your main docker-compose.yml
services:
  # Add PostgreSQL with pgvector
  recommend_db:
    image: ankane/pgvector:latest
    container_name: recommend_db
    ports:
      - "5435:5432"
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=recommendation_db
    volumes:
      - recommend_db_data:/var/lib/postgresql/data
      - ./recommendation-service-java/database/init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - ctuconnect-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d recommendation_db"]
      interval: 10s
      timeout: 5s
      retries: 3

  # Add Recommendation Service
  recommendation-service:
    build: ./recommendation-service-java
    container_name: recommendation-service
    ports:
      - "8095:8095"
    environment:
      - SPRING_PROFILES_ACTIVE=docker
      - POSTGRES_HOST=recommend_db
      - POSTGRES_PORT=5432
      - NEO4J_HOST=neo4j
      - NEO4J_PORT=7687
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - KAFKA_BOOTSTRAP_SERVERS=kafka:29092
      - EUREKA_SERVER_URL=http://eureka-server:8761/eureka/
      - PHOBERT_SERVICE_URL=http://phobert-nlp:8096
    depends_on:
      eureka-server:
        condition: service_healthy
      recommend_db:
        condition: service_healthy
      neo4j:
        condition: service_healthy
      redis:
        condition: service_started
      kafka:
        condition: service_healthy
    networks:
      - ctuconnect-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8095/actuator/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  recommend_db_data:
```

### Step 2: Start Everything

```bash
# From project root
cd d:\LVTN\CTU-Connect-demo

# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f recommendation-service

# Wait for "Started RecommendationServiceApplication"
```

## 📝 Test the Service

### 1. Create Test Data

```bash
# Connect to PostgreSQL
docker exec -it recommend_db psql -U postgres -d recommendation_db

# Insert sample posts
INSERT INTO post_embeddings (post_id, author_id, content, academic_score, academic_category, popularity_score, like_count)
VALUES 
  ('post001', 'user001', 'Thông báo học bổng 2025', 0.9, 'SCHOLARSHIP', 15.5, 25),
  ('post002', 'user002', 'Nghiên cứu AI trong nông nghiệp', 0.95, 'RESEARCH', 20.3, 35),
  ('post003', 'user003', 'Hỏi về môn Cơ sở dữ liệu', 0.85, 'QA', 12.8, 18);

# Exit
\q
```

### 2. Get Recommendations

```bash
# Simple request
curl -X GET "http://localhost:8095/api/recommend/posts?userId=user123&size=10"

# Expected response:
{
  "userId": "user123",
  "recommendations": [
    {
      "postId": "post002",
      "authorId": "user002",
      "content": "Nghiên cứu AI...",
      "finalScore": 0.85,
      "contentSimilarity": 0.82,
      "graphRelationScore": 0.75,
      "academicScore": 0.95,
      "popularityScore": 0.65,
      "academicCategory": "RESEARCH",
      "rank": 1
    }
  ],
  "totalCount": 3,
  "page": 0,
  "size": 10,
  "processingTimeMs": 125
}
```

### 3. Record Feedback

```bash
curl -X POST http://localhost:8095/api/recommend/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "user123",
    "postId": "post002",
    "feedbackType": "LIKE",
    "feedbackValue": 1.0
  }'

# Expected response:
{
  "success": true,
  "message": "Feedback recorded successfully",
  "timestamp": "2025-12-07T14:30:00"
}
```

### 4. Advanced Request with Options

```bash
curl -X POST http://localhost:8095/api/recommend/posts \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "user123",
    "page": 0,
    "size": 20,
    "includeExplanations": true,
    "filterCategories": ["RESEARCH", "SCHOLARSHIP"],
    "excludePostIds": ["post001"]
  }'
```

## 🔍 Monitoring & Debugging

### Check Service Status

```bash
# Health check
curl http://localhost:8095/actuator/health

# Prometheus metrics
curl http://localhost:8095/actuator/prometheus

# Service info
curl http://localhost:8095/actuator/info
```

### View Logs

```bash
# Docker logs
docker logs -f recommendation-service

# Or if running locally
tail -f logs/recommendation-service.log
```

### Check Database

```bash
# PostgreSQL
docker exec -it recommend_db psql -U postgres -d recommendation_db

# Check post embeddings
SELECT post_id, author_id, academic_category, academic_score 
FROM post_embeddings LIMIT 10;

# Check user feedback
SELECT user_id, post_id, feedback_type, feedback_value 
FROM user_feedback ORDER BY timestamp DESC LIMIT 10;

# Neo4j
# Open browser: http://localhost:7474
# Username: neo4j, Password: password

# Redis
docker exec -it redis-recommend redis-cli

# Check keys
KEYS embedding:*
KEYS recommend:*
```

### Test Kafka Integration

```bash
# Publish test post event
docker exec -it kafka /opt/kafka/bin/kafka-console-producer.sh \
  --bootstrap-server localhost:9092 \
  --topic post_created

# Enter JSON:
{
  "eventId": "evt123",
  "eventType": "POST_CREATED",
  "postId": "post999",
  "authorId": "user999",
  "content": "Test post for recommendation",
  "category": "RESEARCH",
  "tags": ["AI", "ML"],
  "timestamp": "2025-12-07T14:30:00"
}

# Press Ctrl+C to exit

# Check logs to see if event was consumed
docker logs recommendation-service | grep "post_created"
```

## 🐛 Troubleshooting

### Issue 1: Cannot connect to PostgreSQL

```bash
# Check if container is running
docker ps | grep recommend_db

# Check logs
docker logs recommend_db

# Test connection
docker exec -it recommend_db psql -U postgres -d recommendation_db -c "SELECT version();"
```

### Issue 2: pgvector extension not found

```bash
# Connect to database
docker exec -it recommend_db psql -U postgres -d recommendation_db

# Enable extension
CREATE EXTENSION IF NOT EXISTS vector;

# Verify
\dx
```

### Issue 3: Service fails to start

```bash
# Check logs
docker logs recommendation-service

# Common issues:
# 1. Eureka not available → Check eureka-server is running
# 2. Database not ready → Wait for healthcheck to pass
# 3. Port already in use → Change port in .env file
```

### Issue 4: No recommendations returned

```bash
# Check if posts exist in database
docker exec -it recommend_db psql -U postgres -d recommendation_db \
  -c "SELECT COUNT(*) FROM post_embeddings;"

# Insert test data if empty
# Run the INSERT statements from Test Data section above
```

### Issue 5: Kafka consumer not working

```bash
# Check Kafka topics
docker exec -it kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --list

# Check consumer group
docker exec -it kafka /opt/kafka/bin/kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --describe --group recommendation-service-group
```

## 🧪 Testing End-to-End

### Complete Test Flow

```bash
# 1. Create a post via Kafka
echo '{
  "eventId": "test1",
  "eventType": "POST_CREATED",
  "postId": "testpost1",
  "authorId": "testuser1",
  "content": "Học bổng sinh viên xuất sắc 2025",
  "category": "SCHOLARSHIP",
  "tags": ["scholarship", "student"],
  "timestamp": "2025-12-07T14:30:00"
}' | docker exec -i kafka /opt/kafka/bin/kafka-console-producer.sh \
  --bootstrap-server localhost:9092 \
  --topic post_created

# 2. Wait 5 seconds for processing

# 3. Get recommendations
curl "http://localhost:8095/api/recommend/posts?userId=testuser2&size=5"

# 4. Like the post
curl -X POST http://localhost:8095/api/recommend/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "testuser2",
    "postId": "testpost1",
    "feedbackType": "LIKE"
  }'

# 5. Get recommendations again (should see updated scores)
curl "http://localhost:8095/api/recommend/posts?userId=testuser2&size=5"
```

## 📊 Performance Testing

### Load Test with curl

```bash
# Simple load test (100 requests)
for i in {1..100}; do
  curl -s "http://localhost:8095/api/recommend/posts?userId=user$i&size=10" > /dev/null &
done
wait

# Check metrics
curl http://localhost:8095/actuator/prometheus | grep recommendation_requests_total
```

### Benchmark with Apache Bench

```bash
# Install Apache Bench
# On Windows: choco install apache-httpd
# On Linux: apt-get install apache2-utils

# Run benchmark
ab -n 1000 -c 10 "http://localhost:8095/api/recommend/posts?userId=user123&size=10"
```

## 🎓 Next Steps

1. **Integrate with Frontend**: Update client-frontend to call recommendation API
2. **Add PhoBERT Service**: Deploy Vietnamese NLP service for embeddings
3. **Populate Neo4j**: Sync user relationships from user-service
4. **Monitor Performance**: Set up Grafana dashboard for metrics
5. **Scale**: Add more service instances behind load balancer

## 📚 Additional Resources

- [Full Documentation](./README.md)
- [Architecture Guide](./ARCHITECTURE.md)
- [API Examples](./docs/API_EXAMPLES.md)
- [Database Schema](./database/init.sql)

---

**Need Help?** Check the logs or file an issue in the repository.

**Success!** 🎉 Your recommendation service is now running!
