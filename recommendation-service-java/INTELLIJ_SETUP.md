# 🧠 IntelliJ IDEA Setup Guide

Hướng dẫn chi tiết setup Recommendation Service trong IntelliJ IDEA.

---

## 📋 Prerequisites

1. **IntelliJ IDEA 2023+** (Community hoặc Ultimate)
2. **Java 17 JDK**
3. **Maven** (built-in hoặc external)
4. **Docker Desktop** (đang chạy)

---

## 🚀 Setup Steps

### 1. Install Required Plugins

1. Open IntelliJ IDEA
2. `File → Settings → Plugins`
3. Install/Enable:
   - ✅ **Lombok** (if not installed)
   - ✅ **Spring Boot** (built-in in Ultimate)
   - ✅ **Docker** (optional, for managing containers)

### 2. Import Project

**Option A: Open existing project**
```
File → Open → Select: d:\LVTN\CTU-Connect-demo\recommendation-service-java
```

**Option B: Import from VCS**
```
File → New → Project from Version Control
URL: [your-git-repo]
```

### 3. Wait for Maven Sync

- IntelliJ will auto-detect `pom.xml`
- Watch bottom-right corner for Maven import
- Wait until indexing completes (~1-2 minutes)
- Resolve any dependency download issues

### 4. Configure JDK

1. `File → Project Structure → Project`
2. Set:
   - **SDK:** Java 17 (or download if not available)
   - **Language Level:** 17 - Sealed types, patterns...
3. Click `Apply`

### 5. Enable Lombok

1. `File → Settings → Build → Compiler → Annotation Processors`
2. Check: ✅ **Enable annotation processing**
3. Click `Apply`

### 6. Create Run Configuration

**Option A: Use pre-configured (recommended)**

The project includes `.idea/runConfigurations/RecommendationService_Dev.xml`

Just open and run!

**Option B: Manual setup**

1. Click `Add Configuration` (top-right)
2. Click `+` → `Spring Boot`
3. Configure:
   - **Name:** `RecommendationService-Dev`
   - **Main class:** `vn.ctu.edu.recommend.RecommendationServiceApplication`
   - **Active profiles:** `dev`
   - **VM options:** `-Dspring.profiles.active=dev`
   - **Working directory:** `$MODULE_WORKING_DIR$`
   - **Use classpath of module:** `recommendation-service`
4. Click `Apply` → `OK`

---

## 🗄️ Start Databases First

**Before running service:**

```bash
# In project root
.\start-dev.ps1
```

Or manually:
```bash
docker-compose -f docker-compose.dev.yml up -d
```

Verify:
```bash
docker-compose -f docker-compose.dev.yml ps
```

---

## ▶️ Run Application

### Method 1: Run Configuration (Recommended)

1. Select `RecommendationService-Dev` from dropdown (top-right)
2. Click ▶️ **Run** button (or press `Shift+F10`)
3. Wait for Spring Boot logo and "Started" message

### Method 2: Maven Tool Window

1. Open Maven tool window (right sidebar)
2. Expand `recommendation-service → Plugins → spring-boot`
3. Double-click `spring-boot:run`
4. Or right-click → Run Maven Goal

### Method 3: Terminal

```bash
mvn spring-boot:run -Dspring-boot.run.profiles=dev
```

---

## 🐛 Debug Application

### Enable Debug Mode

1. Select `RecommendationService-Dev` configuration
2. Click 🐛 **Debug** button (or press `Shift+F9`)
3. Set breakpoints by clicking left margin
4. Application will pause at breakpoints

### Debug Tips

**Set breakpoints:**
- Click left margin next to line number
- Red dot appears

**Conditional breakpoints:**
- Right-click breakpoint
- Enter condition (e.g., `userId.equals("test")`)

**Evaluate expressions:**
- While paused, press `Alt+F8`
- Enter expression to evaluate

**Step through code:**
- `F8` - Step over
- `F7` - Step into
- `Shift+F8` - Step out
- `F9` - Resume

---

## 🔍 View Logs

### Run Console

- Appears automatically when running
- View in bottom panel
- Scroll to see Spring Boot banner and logs

### Log File

- Location: `logs/recommendation-service-dev.log`
- Open in IntelliJ: `View → Tool Windows → File`
- Auto-refreshes when file changes

### Filter Logs

- Use search box in console
- Right-click → `Fold Lines Like This` to hide noise

---

## 🧪 Test APIs

### Built-in HTTP Client

1. Create file: `api-test.http`
2. Add requests:

```http
### Health Check
GET http://localhost:8095/api/recommend/health

### Get Recommendations
GET http://localhost:8095/api/recommend/posts?userId=user123&size=10

### Advanced Recommendations
POST http://localhost:8095/api/recommend/posts
Content-Type: application/json

{
  "userId": "user123",
  "page": 0,
  "size": 10,
  "includeExplanation": true
}

### Record Feedback
POST http://localhost:8095/api/recommend/feedback
Content-Type: application/json

{
  "userId": "user123",
  "postId": "post456",
  "feedbackType": "LIKE"
}
```

3. Click ▶️ next to each request

### Or use Terminal + curl

```bash
curl http://localhost:8095/api/recommend/health
```

---

## 🗄️ Database Tools (Ultimate Edition)

### Configure Database Connection

1. `View → Tool Windows → Database`
2. Click `+` → `Data Source` → `PostgreSQL`
3. Configure:
   - **Host:** localhost
   - **Port:** 5435
   - **Database:** recommendation_db
   - **User:** postgres
   - **Password:** postgres
4. Test Connection → OK

### Query Database

- Right-click database → `New → Query Console`
- Write SQL and press `Ctrl+Enter`

```sql
SELECT * FROM post_embeddings LIMIT 10;
```

---

## 🔧 Useful IntelliJ Features

### 1. Hot Reload (Spring DevTools)

Code changes auto-reload without restart:
- Edit Java file
- Press `Ctrl+F9` (Build Project)
- Changes apply automatically

### 2. Live Templates

Speed up coding with templates:
- `psvm` → public static void main
- `sout` → System.out.println
- `fori` → for loop

### 3. Code Completion

- `Ctrl+Space` - Basic completion
- `Ctrl+Shift+Space` - Smart completion
- `Alt+Enter` - Quick fixes

### 4. Refactoring

- `Shift+F6` - Rename
- `Ctrl+Alt+M` - Extract method
- `Ctrl+Alt+V` - Extract variable
- `Ctrl+Alt+C` - Extract constant

### 5. Navigation

- `Ctrl+N` - Find class
- `Ctrl+Shift+N` - Find file
- `Ctrl+B` - Go to declaration
- `Alt+F7` - Find usages

### 6. Maven Tool Window

- Right sidebar → `Maven`
- Quick access to:
  - Clean, compile, package
  - Reload dependencies
  - View dependency tree

---

## 🎯 Productivity Tips

### 1. Multi-instance Run

Run multiple services for testing:
1. `Run → Edit Configurations`
2. Check: ✅ **Allow multiple instances**
3. Can run multiple ports simultaneously

### 2. Spring Boot Dashboard

**Ultimate Edition only:**
- `View → Tool Windows → Spring Boot`
- See all running Spring Boot apps
- Start/stop/restart with one click

### 3. Bookmarks

Mark important code locations:
- `F11` - Toggle bookmark
- `Shift+F11` - Show bookmarks

### 4. TODO Comments

```java
// TODO: Implement recommendation algorithm
// FIXME: Handle null values
```

View all: `View → Tool Windows → TODO`

### 5. Code Formatting

- `Ctrl+Alt+L` - Format code
- `Ctrl+Alt+O` - Optimize imports

---

## 🐛 Troubleshooting

### Issue: "Cannot resolve symbol"

**Solution:**
```
File → Invalidate Caches → Invalidate and Restart
```

### Issue: Lombok not working

**Solution:**
1. Enable annotation processing (see step 5)
2. Restart IntelliJ
3. Rebuild project: `Build → Rebuild Project`

### Issue: Maven dependencies not downloading

**Solution:**
1. Right-click `pom.xml`
2. `Maven → Reload Project`
3. Or: `mvn clean install -U` in terminal

### Issue: Wrong Java version

**Solution:**
1. `File → Project Structure → Project → SDK`
2. Select Java 17
3. `File → Project Structure → Modules → Language level`
4. Set to 17

### Issue: Port 8095 already in use

**Solution:**
```bash
# Find process
netstat -ano | findstr :8095

# Kill it
taskkill /PID <pid> /F

# Or change port in application-dev.yml
```

---

## ⚙️ Recommended Settings

### 1. Auto-Import

```
File → Settings → Editor → General → Auto Import
✅ Add unambiguous imports on the fly
✅ Optimize imports on the fly
```

### 2. Save Actions

```
File → Settings → Tools → Actions on Save
✅ Reformat code
✅ Optimize imports
✅ Rearrange code (optional)
```

### 3. Memory Settings

For large projects:
```
Help → Change Memory Settings
Set: 4096 MB (or higher)
Restart IntelliJ
```

---

## 📚 Keyboard Shortcuts

### Essential Shortcuts

| Action | Shortcut |
|--------|----------|
| Run | `Shift+F10` |
| Debug | `Shift+F9` |
| Stop | `Ctrl+F2` |
| Build Project | `Ctrl+F9` |
| Find | `Ctrl+F` |
| Find in Files | `Ctrl+Shift+F` |
| Go to Class | `Ctrl+N` |
| Go to File | `Ctrl+Shift+N` |
| Recent Files | `Ctrl+E` |
| Code Completion | `Ctrl+Space` |
| Quick Fix | `Alt+Enter` |

---

## 📊 Performance Monitoring

### Enable JMX Monitoring

Add to VM options:
```
-Dcom.sun.management.jmxremote
-Dcom.sun.management.jmxremote.port=9010
-Dcom.sun.management.jmxremote.authenticate=false
-Dcom.sun.management.jmxremote.ssl=false
```

Connect with JConsole:
```bash
jconsole localhost:9010
```

---

## 🎓 Learning Resources

**IntelliJ IDEA:**
- [Official Guide](https://www.jetbrains.com/idea/guide/)
- [Spring Boot in IntelliJ](https://www.jetbrains.com/help/idea/spring-boot.html)
- [Debugging Tips](https://www.jetbrains.com/help/idea/debugging-code.html)

**Shortcuts PDF:**
```
Help → Keyboard Shortcuts PDF
```

---

## ✅ Quick Checklist

Before starting development:

- [ ] IntelliJ IDEA installed
- [ ] Lombok plugin enabled
- [ ] Annotation processing enabled
- [ ] Java 17 configured
- [ ] Maven synced successfully
- [ ] Docker containers running (`.\start-dev.ps1`)
- [ ] Run configuration created
- [ ] Application starts successfully
- [ ] Health check passes: `curl localhost:8095/api/recommend/health`

---

## 🎉 Ready to Code!

You're all set! Start coding in IntelliJ IDEA with full debugging, hot reload, and database integration.

**Quick Start:**
1. `.\start-dev.ps1` - Start databases
2. Press `Shift+F10` - Run service
3. Code → Save → Auto-reload ✨

**Need help?** Check:
- `DEV_SETUP_GUIDE.md` - Full development guide
- `QUICKREF_DEV.md` - Quick reference
- `README.md` - Complete documentation

---

**Last Updated:** 2025-12-07  
**IntelliJ Version:** 2023.3+
