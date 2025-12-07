# 📋 Quick Reference - Development Mode

## 🚀 Start Development (1 command)

```bash
.\start-dev.ps1
```

Sau đó run service trong IDE hoặc:
```bash
mvn spring-boot:run -Dspring-boot.run.profiles=dev
```

---

## 🗂️ Configuration

### Database Ports
```
PostgreSQL:  localhost:5435
Neo4j HTTP:  localhost:7474
Neo4j Bolt:  localhost:7687  
Redis:       localhost:6379
```

### Service
```
Port:        8095
Profile:     dev
Eureka:      disabled
```

---

## 🔧 Common Commands

### Docker (Databases)

```bash
# Start all databases
docker-compose -f docker-compose.dev.yml up -d

# Stop all databases
docker-compose -f docker-compose.dev.yml down

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Restart single service
docker-compose -f docker-compose.dev.yml restart postgres-recommend
docker-compose -f docker-compose.dev.yml restart neo4j-recommend
docker-compose -f docker-compose.dev.yml restart redis-recommend

# Check status
docker-compose -f docker-compose.dev.yml ps
```

### Service (IDE)

```bash
# Run with Maven
mvn spring-boot:run -Dspring-boot.run.profiles=dev

# Build
mvn clean package -DskipTests

# Clean
mvn clean

# Update dependencies
mvn clean install -U
```

---

## 🗄️ Database Access

### PostgreSQL

```bash
# CLI
docker exec -it postgres-recommend-dev psql -U postgres -d recommendation_db

# Common queries
\dt                              # List tables
\d post_embeddings              # Describe table
SELECT * FROM post_embeddings LIMIT 10;
```

### Neo4j

```
URL: http://localhost:7474
User: neo4j
Pass: password
```

```cypher
// List all nodes
MATCH (n) RETURN n LIMIT 25;

// List users
MATCH (u:User) RETURN u;

// List relationships
MATCH ()-[r]->() RETURN type(r), count(*);
```

### Redis

```bash
# CLI
docker exec -it redis-recommend-dev redis-cli

# Commands
KEYS *
GET embedding:post123
TTL recommend:user456
FLUSHALL  # Clear all (careful!)
```

---

## 🧪 Test APIs

```bash
# Health check
curl http://localhost:8095/api/recommend/health

# Get recommendations
curl "http://localhost:8095/api/recommend/posts?userId=user123&size=10"

# Actuator endpoints
curl http://localhost:8095/actuator/health
curl http://localhost:8095/actuator/metrics
curl http://localhost:8095/actuator/prometheus
```

---

## 🐛 Troubleshooting

### Port conflicts
```bash
# Find process using port
netstat -ano | findstr :5435
netstat -ano | findstr :8095

# Kill process
taskkill /PID <pid> /F
```

### Database won't start
```bash
# Remove volumes and restart
docker-compose -f docker-compose.dev.yml down -v
docker-compose -f docker-compose.dev.yml up -d
```

### Service won't connect
```bash
# Check databases are running
docker-compose -f docker-compose.dev.yml ps

# Check logs
docker-compose -f docker-compose.dev.yml logs postgres-recommend

# Verify connectivity
telnet localhost 5435
telnet localhost 7687
telnet localhost 6379
```

### Build errors
```bash
# Clean rebuild
mvn clean install -DskipTests

# IDE: Invalidate caches
# IntelliJ: File → Invalidate Caches → Restart
# VS Code: Ctrl+Shift+P → Java: Clean Language Server
```

---

## 📊 Monitoring

### Logs

```bash
# Application logs
tail -f logs/recommendation-service-dev.log

# Docker logs
docker-compose -f docker-compose.dev.yml logs -f

# Specific service
docker logs -f postgres-recommend-dev
docker logs -f neo4j-recommend-dev
docker logs -f redis-recommend-dev
```

### Metrics

```bash
# Actuator health
curl http://localhost:8095/actuator/health

# All metrics
curl http://localhost:8095/actuator/metrics

# Specific metric
curl http://localhost:8095/actuator/metrics/jvm.memory.used
```

---

## 🔄 Development Workflow

```bash
# 1. Start databases
.\start-dev.ps1

# 2. Run service in IDE (F5 in VS Code)
# Or: mvn spring-boot:run -Dspring-boot.run.profiles=dev

# 3. Make code changes
# Service auto-reloads with DevTools

# 4. Test
curl http://localhost:8095/api/recommend/posts?userId=test&size=5

# 5. Check logs
tail -f logs/recommendation-service-dev.log

# 6. Stop (when done)
# Ctrl+C (service)
.\stop-dev.ps1  # (databases)
```

---

## 📝 Insert Test Data

### PostgreSQL
```sql
INSERT INTO post_embeddings (id, post_id, author_id, content, academic_score)
VALUES (gen_random_uuid(), 'post001', 'user123', 'Test post', 0.9);
```

### Neo4j
```cypher
CREATE (u:User {userId: 'user123', name: 'Test User', facultyId: 'CNTT'})
```

### Redis (auto-populated by service)
Keys created automatically when service runs.

---

## 🎯 Quick Tips

1. **Use `dev` profile** - Already configured in scripts
2. **DevTools enabled** - Code changes auto-reload
3. **Debug in IDE** - Set breakpoints, press F5
4. **Check health first** - `curl localhost:8095/api/recommend/health`
5. **View logs** - `tail -f logs/recommendation-service-dev.log`
6. **Stop databases** - `.\stop-dev.ps1` (keeps data)
7. **Fresh start** - `docker-compose -f docker-compose.dev.yml down -v`

---

## 📚 Full Documentation

- **DEV_SETUP_GUIDE.md** - Complete development setup
- **README.md** - Full user guide  
- **QUICKSTART.md** - Quick start instructions
- **ARCHITECTURE.md** - Technical architecture

---

**Last Updated:** 2025-12-07
