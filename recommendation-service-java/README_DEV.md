# 🚀 Development Mode - Quick Start

Chạy **Recommendation Service trong IDE** với **Databases trong Docker**.

---

## ⚡ Super Quick Start (Copy-Paste)

```bash
# 1. Start databases
cd d:\LVTN\CTU-Connect-demo\recommendation-service-java
.\start-dev.ps1

# 2. Run service
mvn spring-boot:run -Dspring-boot.run.profiles=dev

# 3. Test
curl http://localhost:8095/api/recommend/health
```

**Xong! Service đang chạy ở `http://localhost:8095`**

---

## 📋 What You Get

### ✅ Databases (Docker)
- **PostgreSQL** - Port 5435
- **Neo4j** - Ports 7474 (HTTP) & 7687 (Bolt)  
- **Redis** - Port 6379

### ✅ Service (IDE)
- Hot reload enabled
- Debug mode ready
- Live logs in console
- Profile: `dev`

---

## 🎯 IDE Setup

### IntelliJ IDEA

```
1. File → Open → recommendation-service-java
2. Wait for Maven sync
3. Run Config: RecommendationService-Dev
4. Click ▶️ (Shift+F10)
```

**Full guide:** [INTELLIJ_SETUP.md](./INTELLIJ_SETUP.md)

### VS Code

```
1. Open Folder → recommendation-service-java
2. Wait for Java extension
3. Press F5
4. Select: RecommendationService (Dev)
```

**Config:** [.vscode/launch.json](./.vscode/launch.json)

---

## 🗂️ Configuration Files

| File | Purpose |
|------|---------|
| `docker-compose.dev.yml` | Docker databases |
| `application-dev.yml` | Dev profile config |
| `start-dev.ps1` | Start databases |
| `stop-dev.ps1` | Stop databases |

---

## 🔧 Common Commands

### Databases

```bash
# Start
.\start-dev.ps1

# Stop (keep data)
.\stop-dev.ps1

# Stop (remove all)
docker-compose -f docker-compose.dev.yml down -v

# View logs
docker-compose -f docker-compose.dev.yml logs -f
```

### Service

```bash
# Run
mvn spring-boot:run -Dspring-boot.run.profiles=dev

# Build
mvn clean package -DskipTests

# Debug
mvn spring-boot:run -Dspring-boot.run.profiles=dev -Dagentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=5005
```

### Database Access

```bash
# PostgreSQL
docker exec -it postgres-recommend-dev psql -U postgres -d recommendation_db

# Neo4j Browser
# Open: http://localhost:7474
# Login: neo4j/password

# Redis CLI
docker exec -it redis-recommend-dev redis-cli
```

---

## 🧪 Test APIs

```bash
# Health check
curl http://localhost:8095/api/recommend/health

# Get recommendations
curl "http://localhost:8095/api/recommend/posts?userId=user123&size=10"

# Actuator
curl http://localhost:8095/actuator/health
curl http://localhost:8095/actuator/metrics
```

---

## 🐛 Troubleshooting

### Databases won't start

```bash
# Check Docker is running
docker version

# Remove old containers
docker-compose -f docker-compose.dev.yml down -v
.\start-dev.ps1
```

### Service won't connect

```bash
# Verify databases are up
docker-compose -f docker-compose.dev.yml ps

# Check ports
netstat -ano | findstr :5435
netstat -ano | findstr :7687
netstat -ano | findstr :6379
```

### Port 8095 in use

```bash
# Find and kill process
netstat -ano | findstr :8095
taskkill /PID <pid> /F
```

---

## 📚 Full Documentation

| Guide | Description |
|-------|-------------|
| **[QUICKREF_DEV.md](./QUICKREF_DEV.md)** | Quick reference card |
| **[DEV_SETUP_GUIDE.md](./DEV_SETUP_GUIDE.md)** | Complete dev guide |
| **[INTELLIJ_SETUP.md](./INTELLIJ_SETUP.md)** | IntelliJ IDEA setup |
| **[README.md](./README.md)** | Main documentation |

---

## 💡 Pro Tips

1. **Hot Reload**: Edit code → Save → Auto-reload (no restart!)
2. **Debug**: Set breakpoints → Press F5/Shift+F9
3. **Logs**: `tail -f logs/recommendation-service-dev.log`
4. **Neo4j UI**: Browse graph at `http://localhost:7474`
5. **Multiple instances**: Check "Allow multiple instances" in Run Config

---

## ✅ Checklist

Before coding:

- [ ] Docker Desktop running
- [ ] Databases started: `.\start-dev.ps1`
- [ ] Service running: `mvn spring-boot:run -Dspring-boot.run.profiles=dev`
- [ ] Health check passes: `curl localhost:8095/api/recommend/health`

---

## 🎉 Happy Coding!

You're ready to develop! 🚀

**Questions?** Check [DEV_SETUP_GUIDE.md](./DEV_SETUP_GUIDE.md)

---

**Last Updated:** 2025-12-07  
**Mode:** Development (IDE + Docker)
