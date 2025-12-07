# 🔧 Setup Fix Guide - Maven Structure

## Issue Fixed

The project was missing Maven wrapper files causing Docker build errors and VS Code recognition issues.

## ✅ What Was Fixed

1. **Created `.mvn/wrapper/` directory structure**
2. **Added `maven-wrapper.properties`**
3. **Created `mvnw` and `mvnw.cmd` scripts**
4. **Updated Dockerfile to use system Maven** (doesn't require wrapper)
5. **Added `.dockerignore`** for cleaner builds
6. **Created `Dockerfile.simple`** as alternative

## 🚀 Quick Test

### Test 1: Maven Validation
```bash
cd d:\LVTN\CTU-Connect-demo\recommendation-service-java
mvn validate
```
Expected: ✓ BUILD SUCCESS

### Test 2: Build Project
```bash
mvn clean package -DskipTests
```
Expected: Creates `target/recommendation-service-1.0.0-SNAPSHOT.jar`

### Test 3: VS Code Recognition
1. Open folder in VS Code
2. VS Code should detect it as a Maven project
3. Check bottom status bar for "Java Project" indicator
4. Extension Pack for Java should show project structure

### Test 4: Docker Build (Option 1 - Using main Dockerfile)
```bash
docker build -t ctu-recommend .
```

### Test 5: Docker Build (Option 2 - Using simple Dockerfile)
```bash
docker build -f Dockerfile.simple -t ctu-recommend .
```

## 📋 Maven Project Structure Checklist

```
✓ pom.xml at root
✓ src/main/java/ directory
✓ src/main/resources/ directory
✓ .mvn/wrapper/ directory (optional but good to have)
✓ mvnw and mvnw.cmd (optional)
✓ .gitignore
✓ .dockerignore
```

## 🔍 VS Code Setup

### Required Extensions
1. **Extension Pack for Java** (by Microsoft)
   - Language Support for Java
   - Debugger for Java
   - Test Runner for Java
   - Maven for Java
   - Project Manager for Java

2. **Spring Boot Extension Pack** (by VMware)
   - Spring Boot Tools
   - Spring Initializr
   - Spring Boot Dashboard

### Install Extensions
```bash
# Open VS Code Command Palette (Ctrl+Shift+P)
# Type: Extensions: Install Extensions
# Search for: "Extension Pack for Java"
# Search for: "Spring Boot Extension Pack"
```

### Verify Project Recognition

After opening the project in VS Code:

1. **Check Explorer Panel** (left sidebar):
   - Should show "JAVA PROJECTS" section
   - Should list "recommendation-service" project
   - Should show package structure

2. **Check Bottom Status Bar**:
   - Should show Java version (17)
   - Should show Spring Boot icon
   - Should show Maven icon

3. **Check Problems Panel** (Ctrl+Shift+M):
   - Should not show Maven errors
   - May show warnings (normal for new project)

## 🛠️ Troubleshooting

### Issue: "Cannot find symbol" errors in VS Code

**Solution**:
```bash
# Clean and rebuild
mvn clean install -DskipTests

# In VS Code: Reload window
# Press Ctrl+Shift+P
# Type: "Java: Clean Java Language Server Workspace"
# Type: "Developer: Reload Window"
```

### Issue: VS Code not recognizing as Maven project

**Solution**:
1. Close VS Code
2. Delete `.vscode/` folder if exists
3. Open VS Code
4. Open folder: `recommendation-service-java`
5. Wait for Java extension to initialize (see bottom status bar)
6. If prompted, click "Import Projects"

### Issue: Maven dependencies not downloading

**Solution**:
```bash
# Force update
mvn clean install -U -DskipTests

# Or delete local repo cache
rm -rf ~/.m2/repository
mvn clean install -DskipTests
```

### Issue: Docker build fails with "not found"

**Solution**: Use the simple Dockerfile
```bash
# Use Dockerfile.simple which doesn't need wrapper
docker build -f Dockerfile.simple -t ctu-recommend .

# Or rename it
mv Dockerfile Dockerfile.old
mv Dockerfile.simple Dockerfile
docker build -t ctu-recommend .
```

## ✨ Recommended Workflow

### For Development:
```bash
# 1. Open in VS Code
code d:\LVTN\CTU-Connect-demo\recommendation-service-java

# 2. Let extensions initialize (wait ~30 seconds)

# 3. Build with Maven
mvn clean package -DskipTests

# 4. Run from VS Code
# Press F5 or use "Run and Debug" panel
# Or use terminal:
mvn spring-boot:run
```

### For Docker Deployment:
```bash
# Build image
docker build -f Dockerfile.simple -t ctu-recommend .

# Run container
docker run -d -p 8095:8095 --name recommend-service ctu-recommend

# Check logs
docker logs -f recommend-service

# Stop container
docker stop recommend-service
docker rm recommend-service
```

## 📦 Alternative: Use Maven Wrapper (Optional)

If you want to use Maven wrapper, you can download it:

```bash
# Windows PowerShell
cd d:\LVTN\CTU-Connect-demo\recommendation-service-java

# Download wrapper
mvn -N wrapper:wrapper

# Now you can use ./mvnw
.\mvnw.cmd clean package -DskipTests
```

## 🎯 Next Steps

1. ✅ Verify Maven structure: `mvn validate`
2. ✅ Build project: `mvn clean package -DskipTests`
3. ✅ Open in VS Code and check recognition
4. ✅ Setup databases (PostgreSQL, Neo4j, Redis)
5. ✅ Run service: `mvn spring-boot:run`
6. ✅ Test API: `curl http://localhost:8095/api/recommend/health`

## 📚 Related Documentation

- [README.md](./README.md) - Complete setup guide
- [QUICKSTART.md](./QUICKSTART.md) - Quick start instructions
- [ARCHITECTURE.md](./ARCHITECTURE.md) - Technical details

---

**Issue Status**: ✅ FIXED  
**Date**: 2025-12-07  
**Changes**: Maven structure corrected, Dockerfile updated, VS Code compatibility ensured
