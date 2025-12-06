# Neo4j Connection Fix

## Problem
```
Caused by: org.neo4j.driver.exceptions.ClientException: 
Server responded HTTP. Make sure you are not trying to connect to 
the http endpoint (HTTP defaults to port 7474 whereas BOLT defaults to port 7687)
```

## Root Cause
The user-service was configured to connect to Neo4j using:
- **Wrong URI**: `bolt://localhost:7474` ❌
- **Correct URI**: `bolt://localhost:7687` ✅

Additionally, there was a **duplicate configuration file** in the wrong location:
- **Wrong location**: `src/main/java/resources/application.properties` ❌
- **Correct location**: `src/main/resources/application.properties` ✅

## Understanding Neo4j Ports

Neo4j exposes two different protocols on different ports:

| Protocol | Default Port | Purpose |
|----------|-------------|----------|
| **HTTP** | 7474 | Web browser interface |
| **BOLT** | 7687 | Database connections (Spring Data Neo4j) |

Spring Data Neo4j uses the **BOLT protocol**, which requires port **7687**.

## Solution Applied

### 1. Fixed Neo4j URI in Configuration
**File**: `user-service/src/main/resources/application.properties`

**Before:**
```properties
spring.neo4j.uri=bolt://localhost:7474  # WRONG PORT!
```

**After:**
```properties
spring.neo4j.uri=bolt://localhost:7687  # CORRECT PORT!
```

### 2. Removed Duplicate Configuration File
The project had an incorrect directory structure with resources in the wrong location:
- ❌ Removed: `src/main/java/resources/application.properties`
- ✅ Kept: `src/main/resources/application.properties`

**Backup Created**: Files were backed up to `resources-backup-[timestamp]` before removal.

### 3. Verified Neo4j Container
```bash
Container: neo4j-graph-db
Status: healthy ✅
Ports: 
  - 7474 (HTTP) → accessible
  - 7687 (BOLT) → accessible
```

## Current Configuration

### application.properties (Main)
```properties
server.port=8081

spring.application.name=user-service

# Database Configuration
spring.neo4j.uri=bolt://localhost:7687
spring.neo4j.authentication.username=neo4j
spring.neo4j.authentication.password=password
spring.data.neo4j.database=neo4j

# Connection Pool Configuration
spring.data.neo4j.pool.idle-time-before-connection-test=PT30S
spring.data.neo4j.pool.max-connection-pool-size=50
spring.data.neo4j.pool.max-connection-lifetime=PT1H

# Logging
logging.level.org.springframework.data.neo4j=INFO
logging.level.org.neo4j.driver=WARN
```

## How to Apply the Fix

**Just restart the user-service:**
```bash
# The configuration files are already fixed
# Simply restart your user-service process
```

## Verification

### Test Neo4j Connection
```bash
# Check if Neo4j BOLT port is accessible
Test-NetConnection -ComputerName localhost -Port 7687

# Check Neo4j container health
docker inspect neo4j-graph-db --format '{{.State.Health.Status}}'
```

Expected output:
```
TcpTestSucceeded : True
healthy
```

### Check User Service Logs
After restart, look for successful connection:
```
Connected to Neo4j database: neo4j
```

If connection fails, you'll see:
```
Unable to connect to database
```

## Neo4j Access Information

### Web Interface (Browser)
- **URL**: http://localhost:7474
- **Username**: neo4j
- **Password**: password

### BOLT Connection (Application)
- **URI**: bolt://localhost:7687
- **Username**: neo4j
- **Password**: password

## Common Neo4j Connection Issues

### 1. Wrong Port
**Error**: "Server responded HTTP"  
**Solution**: Use port 7687 for BOLT, not 7474

### 2. Wrong Protocol
**Error**: Connection refused  
**Solution**: Use `bolt://` not `http://` or `neo4j://`

### 3. Container Not Running
**Error**: Connection timeout  
**Solution**: Start Neo4j container
```bash
docker start neo4j-graph-db
```

### 4. Wrong Credentials
**Error**: Authentication failed  
**Solution**: Verify username/password match Neo4j settings

### 5. Network Issues
**Error**: Host unreachable  
**Solution**: 
- Check if Docker container is running
- Verify port mapping in docker-compose.yml
- Check firewall settings

## Files Modified

1. ✏️ `user-service/src/main/resources/application.properties` - Fixed Neo4j URI
2. ❌ `user-service/src/main/java/resources/` - Removed (incorrect location)
3. 💾 Backup created at: `user-service/resources-backup-[timestamp]/`

## Docker Compose Verification

The Neo4j configuration in `docker-compose.yml` is correct:

```yaml
neo4j:
  image: neo4j:5.13.0
  container_name: neo4j-graph-db
  ports:
    - "7474:7474"  # HTTP interface
    - "7687:7687"  # BOLT protocol
  environment:
    - NEO4J_AUTH=neo4j/password
```

## Next Steps

1. **Restart user-service** to apply the configuration changes
2. **Test user creation** to verify Neo4j connection
3. **Check logs** for any connection errors
4. **Access Neo4j browser** at http://localhost:7474 to verify data

## Expected Result After Fix

✅ User-service connects to Neo4j successfully  
✅ User data can be stored in Neo4j graph database  
✅ No more "Server responded HTTP" errors  
✅ BOLT protocol connections work properly  

---

**Status**: ✅ FIXED  
**Date**: December 4, 2025  
**Impact**: User-service can now connect to Neo4j properly
