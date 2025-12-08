# 🐳 Docker Setup - Recommendation Service (Development Mode)

## 📋 Tổng quan

Docker setup này được tối ưu cho **development**, bao gồm:
- ✅ **PostgreSQL** - Database cho embeddings và cache
- ✅ **Redis** - Cache layer cho performance
- ✅ **Python Service** - Volume mount để live reload code

**Note:** Java service chạy trực tiếp trên IDE để phục vụ development, không qua Docker.

---

## 🚀 Quick Start

### 1. Tạo network (nếu chưa có)

```bash
docker network create ctu-connect-network
```

### 2. Khởi động services

```bash
cd recommend-service/docker
docker-compose -f docker-compose.dev.yml up -d
```

### 3. Kiểm tra

```bash
# Check containers
docker-compose -f docker-compose.dev.yml ps

# Check Python service
curl http://localhost:8000/health

# Check PostgreSQL
docker exec -it ctu-recommend-postgres psql -U recommend_user -d recommend_db -c "\dt recommend.*"

# Check Redis
docker exec -it ctu-recommend-redis redis-cli -a recommend_redis_pass ping
```

---

## 📦 Services

### 1. PostgreSQL (Port 5433)

**Container:** `ctu-recommend-postgres`

**Credentials:**
- Database: `recommend_db`
- User: `recommend_user`
- Password: `recommend_pass`
- Port: `5433` (host) → `5432` (container)

**Connection String:**
```
postgresql://recommend_user:recommend_pass@localhost:5433/recommend_db
```

**Schema:** `recommend`

**Tables:**
- `post_embeddings` - Post embeddings storage
- `user_embeddings` - User embeddings storage
- `recommendation_cache` - Recommendation cache
- `recommendation_logs` - Request logs
- `similarity_cache` - Similarity computation cache

**Management:**
```bash
# Connect to database
docker exec -it ctu-recommend-postgres psql -U recommend_user -d recommend_db

# View tables
\dt recommend.*

# View embedding stats
SELECT * FROM recommend.embedding_stats;

# Run maintenance
SELECT recommend.maintenance();
```

---

### 2. Redis (Port 6380)

**Container:** `ctu-recommend-redis`

**Credentials:**
- Password: `recommend_redis_pass`
- Port: `6380` (host) → `6379` (container)

**Connection String:**
```
redis://:recommend_redis_pass@localhost:6380/0
```

**Usage:**
- Cache user embeddings (TTL: 1 hour)
- Cache post embeddings (persistent)
- Cache recommendation results (TTL: 5 minutes)

**Management:**
```bash
# Connect to Redis
docker exec -it ctu-recommend-redis redis-cli -a recommend_redis_pass

# Check keys
KEYS *

# Get key info
TTL user:embedding:123

# Clear cache
FLUSHDB
```

---

### 3. Python Service (Port 8000)

**Container:** `ctu-recommend-python`

**Features:**
- ✅ **Live reload** - Code changes automatically reload
- ✅ **Volume mount** - `/app` mounted from `../python-model`
- ✅ **Hot reload** - Uvicorn `--reload` enabled
- ✅ **Debug mode** - `PYTHONUNBUFFERED=1`

**Endpoints:**
```
GET  /health                 - Health check
GET  /docs                   - Swagger UI
POST /embed/post             - Generate post embedding
POST /embed/post/batch       - Batch post embeddings
POST /embed/user             - Generate user embedding
POST /similarity             - Compute similarity
POST /similarity/batch       - Batch similarity
```

**Development Workflow:**
1. Edit code in `recommend-service/python-model/`
2. Save file
3. Service automatically reloads (watch logs)
4. Test immediately

**View Logs:**
```bash
# Follow logs
docker-compose -f docker-compose.dev.yml logs -f recommend-python

# Last 100 lines
docker-compose -f docker-compose.dev.yml logs --tail=100 recommend-python
```

---

## 🔧 Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Edit `.env`:
```env
# PostgreSQL
POSTGRES_DB=recommend_db
POSTGRES_USER=recommend_user
POSTGRES_PASSWORD=recommend_pass

# Redis
REDIS_PASSWORD=recommend_redis_pass

# Python Service
MODEL_PATH=/app/model/academic_posts_model
LOG_LEVEL=DEBUG  # Change to DEBUG for more logs
WORKERS=1
```

### Volume Mounts

Python service mounts:
- `../python-model:/app` - Full source code
- Excludes: `__pycache__`, `.pytest_cache`, `venv`

This allows instant code changes without rebuilding.

---

## 🧪 Testing

### Test Python Service

```bash
# Health check
curl http://localhost:8000/health

# Generate post embedding
curl -X POST http://localhost:8000/embed/post \
  -H "Content-Type: application/json" \
  -d '{
    "post_id": "test1",
    "content": "Mạng máy tính - giao thức TCP/IP",
    "title": "TCP/IP"
  }'

# Generate user embedding
curl -X POST http://localhost:8000/embed/user \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user1",
    "major": "Khoa học máy tính",
    "faculty": "CNTT"
  }'
```

### Test Database

```bash
# Check tables
docker exec -it ctu-recommend-postgres psql -U recommend_user -d recommend_db \
  -c "SELECT COUNT(*) FROM recommend.post_embeddings;"

# Insert test data
docker exec -it ctu-recommend-postgres psql -U recommend_user -d recommend_db \
  -c "INSERT INTO recommend.post_embeddings (post_id, embedding, dimension) 
      VALUES ('test1', ARRAY[0.1, 0.2, 0.3], 3);"

# Query test data
docker exec -it ctu-recommend-postgres psql -U recommend_user -d recommend_db \
  -c "SELECT * FROM recommend.post_embeddings WHERE post_id='test1';"
```

### Test Redis

```bash
# Set a test key
docker exec -it ctu-recommend-redis redis-cli -a recommend_redis_pass \
  SET test_key "test_value" EX 300

# Get the key
docker exec -it ctu-recommend-redis redis-cli -a recommend_redis_pass \
  GET test_key
```

---

## 🔄 Common Operations

### Start Services

```bash
docker-compose -f docker-compose.dev.yml up -d
```

### Stop Services

```bash
docker-compose -f docker-compose.dev.yml down
```

### Restart Python Service (after config change)

```bash
docker-compose -f docker-compose.dev.yml restart recommend-python
```

### View Logs

```bash
# All services
docker-compose -f docker-compose.dev.yml logs -f

# Specific service
docker-compose -f docker-compose.dev.yml logs -f recommend-python
docker-compose -f docker-compose.dev.yml logs -f recommend-postgres
docker-compose -f docker-compose.dev.yml logs -f recommend-redis
```

### Clean Up

```bash
# Stop and remove containers
docker-compose -f docker-compose.dev.yml down

# Also remove volumes (WARNING: deletes all data)
docker-compose -f docker-compose.dev.yml down -v
```

### Reset Database

```bash
# Remove only database volume
docker volume rm ctu-recommend-postgres-data

# Restart to reinitialize
docker-compose -f docker-compose.dev.yml up -d recommend-postgres
```

---

## 🐛 Troubleshooting

### Python service won't start

**Problem:** Container keeps restarting

**Check logs:**
```bash
docker-compose -f docker-compose.dev.yml logs recommend-python
```

**Common issues:**
1. Missing requirements: Check `requirements.txt`
2. Model not found: Check `MODEL_PATH` and model directory
3. Port conflict: Check if port 8000 is in use

**Solution:**
```bash
# Rebuild without cache
docker-compose -f docker-compose.dev.yml up --build --force-recreate recommend-python
```

### Database connection failed

**Check:**
```bash
# Is PostgreSQL running?
docker-compose -f docker-compose.dev.yml ps recommend-postgres

# Check health
docker inspect ctu-recommend-postgres | grep -A 10 Health
```

**Test connection:**
```bash
docker exec -it ctu-recommend-postgres psql -U recommend_user -d recommend_db -c "SELECT 1;"
```

### Redis connection failed

**Check:**
```bash
# Is Redis running?
docker-compose -f docker-compose.dev.yml ps recommend-redis

# Test connection
docker exec -it ctu-recommend-redis redis-cli -a recommend_redis_pass ping
```

### Code changes not reloading

**Check:**
1. Volume mount is correct: `docker-compose -f docker-compose.dev.yml config`
2. Uvicorn is in reload mode: Check logs for `--reload`
3. File permissions: Ensure files are writable

**Force reload:**
```bash
docker-compose -f docker-compose.dev.yml restart recommend-python
```

### Port already in use

**Find process:**
```bash
# Windows
netstat -ano | findstr :8000
netstat -ano | findstr :5433
netstat -ano | findstr :6380

# Linux/Mac
lsof -i :8000
lsof -i :5433
lsof -i :6380
```

**Change port in docker-compose.dev.yml:**
```yaml
ports:
  - "8001:8000"  # Change host port
```

---

## 📊 Monitoring

### Container Stats

```bash
docker stats ctu-recommend-python ctu-recommend-postgres ctu-recommend-redis
```

### Database Size

```bash
docker exec -it ctu-recommend-postgres psql -U recommend_user -d recommend_db \
  -c "SELECT pg_size_pretty(pg_database_size('recommend_db'));"
```

### Redis Memory

```bash
docker exec -it ctu-recommend-redis redis-cli -a recommend_redis_pass INFO memory
```

---

## 🔐 Security Notes

**For Development Only:**
- Default passwords are used
- Redis has simple password auth
- Database is exposed on host

**For Production:**
- Change all passwords
- Use secrets management
- Enable SSL/TLS
- Restrict network access
- Use strong authentication

---

## 📝 Development Workflow

### Typical Day

```bash
# Morning: Start services
cd recommend-service/docker
docker-compose -f docker-compose.dev.yml up -d

# Check everything is running
docker-compose -f docker-compose.dev.yml ps
curl http://localhost:8000/health

# Start Java service in IDE (IntelliJ/Eclipse)
# Port 8081

# Develop: Edit Python code
# Changes auto-reload, test immediately

# Debug: View logs
docker-compose -f docker-compose.dev.yml logs -f recommend-python

# Evening: Stop services
docker-compose -f docker-compose.dev.yml down
```

### Making Changes

1. **Python Code:**
   - Edit files in `recommend-service/python-model/`
   - Save → Auto reload
   - Test immediately

2. **Database Schema:**
   - Edit `init-db/01-init-recommend-db.sql`
   - Reset database: `docker volume rm ctu-recommend-postgres-data`
   - Restart: `docker-compose -f docker-compose.dev.yml up -d`

3. **Docker Configuration:**
   - Edit `docker-compose.dev.yml`
   - Restart: `docker-compose -f docker-compose.dev.yml up -d`

---

## 🎯 Next Steps

1. ✅ Start services with `docker-compose up`
2. ✅ Verify health endpoints
3. ✅ Connect IDE to databases for inspection
4. ✅ Start developing with live reload
5. ✅ Test endpoints with Postman/curl

---

## 📚 Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Redis Documentation](https://redis.io/documentation)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Happy Coding! 🚀**

*Last updated: December 2024*
