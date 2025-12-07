# 🛠️ Development Setup Guide - IDE + Docker

Hướng dẫn chạy Recommendation Service trên IDE (IntelliJ IDEA / VS Code) với databases chạy trên Docker.

---

## 📋 Prerequisites

### 1. Tools Required
- ✅ **Java 17** (JDK 17 or higher)
- ✅ **Maven 3.8+** 
- ✅ **Docker Desktop** (for databases)
- ✅ **IDE**: IntelliJ IDEA hoặc VS Code
- ✅ **Git** (for version control)

### 2. IDE Extensions

#### IntelliJ IDEA:
- ✅ Spring Boot plugin (built-in)
- ✅ Lombok plugin

#### VS Code:
- ✅ Extension Pack for Java (Microsoft)
- ✅ Spring Boot Extension Pack (VMware)
- ✅ Lombok Annotations Support

---

## 🚀 Quick Start (5 phút)

### Bước 1: Start Databases với Docker Compose

```bash
# Di chuyển vào thư mục project
cd d:\LVTN\CTU-Connect-demo\recommendation-service-java

# Start tất cả databases
docker-compose -f docker-compose.dev.yml up -d

# Kiểm tra trạng thái
docker-compose -f docker-compose.dev.yml ps
```

**Expected Output:**
```
NAME                      STATUS    PORTS
postgres-recommend-dev    Up        0.0.0.0:5435->5432/tcp
neo4j-recommend-dev       Up        0.0.0.0:7474->7474/tcp, 0.0.0.0:7687->7687/tcp
redis-recommend-dev       Up        0.0.0.0:6379->6379/tcp
```

### Bước 2: Verify Databases

```bash
# Check PostgreSQL
docker exec -it postgres-recommend-dev psql -U postgres -c "\l"

# Check Neo4j
# Open browser: http://localhost:7474
# Username: neo4j, Password: password

# Check Redis
docker exec -it redis-recommend-dev redis-cli ping
# Expected: PONG
```

### Bước 3: Run Service trong IDE

#### Option A: IntelliJ IDEA

1. **Open Project**:
   ```
   File → Open → Select: recommendation-service-java
   ```

2. **Wait for Maven import** (bottom-right corner)

3. **Configure Run Configuration**:
   - Click "Add Configuration" (top-right)
   - Click "+" → Spring Boot
   - Name: `RecommendationService-Dev`
   - Main class: `vn.ctu.edu.recommend.RecommendationServiceApplication`
   - VM options: `-Dspring.profiles.active=dev`
   - Working directory: `$MODULE_WORKING_DIR$`
   - Click "OK"

4. **Run Application**:
   - Click ▶️ (Run button) or press `Shift+F10`
   - Wait for startup (about 10-20 seconds)

#### Option B: VS Code

1. **Open Project**:
   ```
   File → Open Folder → Select: recommendation-service-java
   ```

2. **Wait for Java extension to load** (bottom status bar)

3. **Create Launch Configuration**:
   - Create `.vscode/launch.json`:
   ```json
   {
     "version": "0.2.0",
     "configurations": [
       {
         "type": "java",
         "name": "RecommendationService-Dev",
         "request": "launch",
         "mainClass": "vn.ctu.edu.recommend.RecommendationServiceApplication",
         "projectName": "recommendation-service",
         "vmArgs": "-Dspring.profiles.active=dev",
         "console": "internalConsole"
       }
     ]
   }
   ```

4. **Run Application**:
   - Press `F5` or click "Run and Debug" → "RecommendationService-Dev"
   - Or open terminal and run: `mvn spring-boot:run -Dspring-boot.run.profiles=dev`

#### Option C: Command Line (Terminal)

```bash
# Set profile and run
mvn spring-boot:run -Dspring-boot.run.profiles=dev

# Or run JAR directly
mvn clean package -DskipTests
java -jar -Dspring.profiles.active=dev target/recommendation-service-1.0.0-SNAPSHOT.jar
```

### Bước 4: Verify Service is Running

```bash
# Health check
curl http://localhost:8095/api/recommend/health

# Expected response:
# {"status":"UP","timestamp":"..."}

# Actuator health
curl http://localhost:8095/actuator/health
```

---

## 🎯 Configuration Details

### Database Ports (Docker)
```
PostgreSQL: localhost:5435
Neo4j HTTP: localhost:7474
Neo4j Bolt: localhost:7687
Redis:      localhost:6379
```

### Service Configuration
```
Service Port: 8095
Profile: dev
Eureka: disabled (for standalone dev)
```

---

## 🔧 Development Workflow

### 1. Start Development Session

```bash
# Terminal 1: Start databases
docker-compose -f docker-compose.dev.yml up

# Terminal 2: Run service in IDE or
mvn spring-boot:run -Dspring-boot.run.profiles=dev
```

### 2. Make Code Changes

- Edit Java files in IDE
- IDE will auto-compile
- Spring Boot DevTools will auto-reload (if enabled)

### 3. Test Changes

```bash
# Test API endpoint
curl http://localhost:8095/api/recommend/posts?userId=test&size=5

# Check logs
tail -f logs/recommendation-service-dev.log
```

### 4. Stop Development Session

```bash
# Stop service: Ctrl+C in terminal or Stop button in IDE

# Stop databases (but keep data)
docker-compose -f docker-compose.dev.yml stop

# Stop and remove containers
docker-compose -f docker-compose.dev.yml down

# Stop and remove ALL (including data)
docker-compose -f docker-compose.dev.yml down -v
```

---

## 📊 Database Access

### PostgreSQL

```bash
# Connect via Docker
docker exec -it postgres-recommend-dev psql -U postgres -d recommendation_db

# Or use GUI tool (e.g., pgAdmin, DBeaver)
Host: localhost
Port: 5435
Database: recommendation_db
Username: postgres
Password: postgres
```

**Common Queries:**
```sql
-- List tables
\dt

-- Check post_embeddings
SELECT id, post_id, academic_score FROM post_embeddings LIMIT 10;

-- Check user_feedback
SELECT * FROM user_feedback ORDER BY created_at DESC LIMIT 10;
```

### Neo4j

```
# Browser UI
URL: http://localhost:7474
Username: neo4j
Password: password
```

**Common Cypher Queries:**
```cypher
// List all nodes
MATCH (n) RETURN n LIMIT 25;

// Check users
MATCH (u:User) RETURN u LIMIT 10;

// Check relationships
MATCH (u1:User)-[r]->(u2:User) RETURN u1, r, u2 LIMIT 10;
```

### Redis

```bash
# Connect via CLI
docker exec -it redis-recommend-dev redis-cli

# Common commands
KEYS *
GET embedding:some-post-id
TTL recommend:some-user-id
```

---

## 🐛 Troubleshooting

### Issue 1: Port already in use

```bash
# Check what's using the port
netstat -ano | findstr :5435
netstat -ano | findstr :7687
netstat -ano | findstr :6379
netstat -ano | findstr :8095

# Kill process (Windows)
taskkill /PID <process_id> /F

# Or change ports in docker-compose.dev.yml
```

### Issue 2: Database connection refused

```bash
# Check if containers are running
docker-compose -f docker-compose.dev.yml ps

# Check logs
docker-compose -f docker-compose.dev.yml logs postgres-recommend
docker-compose -f docker-compose.dev.yml logs neo4j-recommend
docker-compose -f docker-compose.dev.yml logs redis-recommend

# Restart containers
docker-compose -f docker-compose.dev.yml restart
```

### Issue 3: Service won't start

**Check Java version:**
```bash
java -version
# Should be 17 or higher
```

**Check Maven:**
```bash
mvn -version
```

**Check profile is active:**
```
Look for log line: "The following profiles are active: dev"
```

**Check database connectivity:**
```bash
# Test PostgreSQL
telnet localhost 5435

# Test Neo4j
telnet localhost 7687

# Test Redis
telnet localhost 6379
```

### Issue 4: "Cannot find symbol" errors

```bash
# Clean and rebuild
mvn clean install -DskipTests

# In IDE: Invalidate caches and restart
# IntelliJ: File → Invalidate Caches → Invalidate and Restart
# VS Code: Ctrl+Shift+P → Java: Clean Java Language Server Workspace
```

---

## 🧪 Testing APIs

### Sample Requests

```bash
# 1. Health check
curl http://localhost:8095/api/recommend/health

# 2. Get recommendations
curl "http://localhost:8095/api/recommend/posts?userId=user123&size=10"

# 3. Advanced recommendations (POST)
curl -X POST http://localhost:8095/api/recommend/posts \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "user123",
    "page": 0,
    "size": 10,
    "includeExplanation": true
  }'

# 4. Record feedback
curl -X POST http://localhost:8095/api/recommend/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "user123",
    "postId": "post456",
    "feedbackType": "LIKE",
    "timestamp": "2025-12-07T10:00:00"
  }'

# 5. Rebuild embeddings (admin)
curl -X POST http://localhost:8095/api/recommend/embedding/rebuild

# 6. Clear cache
curl -X DELETE http://localhost:8095/api/recommend/cache/user123
```

---

## 📝 Insert Test Data

### PostgreSQL Test Data

```sql
-- Connect to database
docker exec -it postgres-recommend-dev psql -U postgres -d recommendation_db

-- Insert sample posts
INSERT INTO post_embeddings (id, post_id, author_id, content, academic_score) VALUES
(gen_random_uuid(), 'post001', 'user123', 'Nghiên cứu về AI trong giáo dục', 0.95),
(gen_random_uuid(), 'post002', 'user456', 'Học bổng toàn phần du học Mỹ', 0.90),
(gen_random_uuid(), 'post003', 'user789', 'Cuộc thi lập trình CTU Code War', 0.85);

-- Verify
SELECT * FROM post_embeddings;
```

### Neo4j Test Data

```cypher
// Create users
CREATE (u1:User {userId: 'user123', name: 'Nguyen Van A', facultyId: 'CNTT', majorId: 'KTPM', batchId: '2021'})
CREATE (u2:User {userId: 'user456', name: 'Tran Thi B', facultyId: 'CNTT', majorId: 'KTPM', batchId: '2021'})
CREATE (u3:User {userId: 'user789', name: 'Le Van C', facultyId: 'CNTT', majorId: 'HTTT', batchId: '2021'})

// Create relationships
CREATE (u1)-[:FRIEND]->(u2)
CREATE (u1)-[:SAME_MAJOR]->(u2)
CREATE (u1)-[:SAME_FACULTY]->(u3)

// Verify
MATCH (n) RETURN n;
```

---

## 🎓 Development Tips

### 1. Enable Hot Reload

Add to `pom.xml` (already included):
```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-devtools</artifactId>
    <scope>runtime</scope>
    <optional>true</optional>
</dependency>
```

### 2. Debug in IDE

**IntelliJ IDEA:**
- Set breakpoints in code
- Click 🐛 (Debug) instead of ▶️ (Run)
- Use Debug Console to evaluate expressions

**VS Code:**
- Set breakpoints in code
- Press `F5` to start debugging
- Use Debug Console for evaluation

### 3. Monitor Performance

```bash
# Actuator metrics
curl http://localhost:8095/actuator/metrics

# Prometheus metrics
curl http://localhost:8095/actuator/prometheus

# JVM info
curl http://localhost:8095/actuator/info
```

### 4. View Logs

**Real-time logs:**
```bash
tail -f logs/recommendation-service-dev.log
```

**Filter logs:**
```bash
# Only errors
grep "ERROR" logs/recommendation-service-dev.log

# Specific class
grep "RecommendationService" logs/recommendation-service-dev.log
```

---

## 🔄 Update Dependencies

```bash
# Update Maven dependencies
mvn clean install -U

# Check for updates
mvn versions:display-dependency-updates
```

---

## 📚 Related Files

```
docker-compose.dev.yml         # Docker setup for dev
application-dev.yml            # Dev profile configuration
DEV_SETUP_GUIDE.md            # This file
README.md                      # Main documentation
QUICKSTART.md                  # Quick start guide
```

---

## ✅ Development Checklist

Before starting development:
- [ ] Docker Desktop is running
- [ ] Databases are up: `docker-compose -f docker-compose.dev.yml ps`
- [ ] Java 17 is installed: `java -version`
- [ ] Maven is installed: `mvn -version`
- [ ] IDE has Java/Spring extensions
- [ ] Project builds successfully: `mvn clean install -DskipTests`

---

## 🎉 Happy Coding!

Bạn đã sẵn sàng để phát triển Recommendation Service!

**Support:**
- Check logs: `logs/recommendation-service-dev.log`
- Docker logs: `docker-compose -f docker-compose.dev.yml logs -f`
- Documentation: See README.md

**Quick Commands:**
```bash
# Start everything
docker-compose -f docker-compose.dev.yml up -d && mvn spring-boot:run -Dspring-boot.run.profiles=dev

# Stop everything
# Ctrl+C (service) && docker-compose -f docker-compose.dev.yml down

# Restart databases
docker-compose -f docker-compose.dev.yml restart
```

---

**Last Updated:** 2025-12-07  
**Status:** ✅ Ready for Development
