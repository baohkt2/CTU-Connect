# Test Integration Script for Recommendation Service
# Tests connectivity with other services and basic functionality

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Recommendation Service Integration Test" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

$EUREKA_URL = "http://localhost:8761"
$RECOMMEND_URL = "http://localhost:8095"
$API_GATEWAY_URL = "http://localhost:8090"
$NEO4J_URL = "http://localhost:7474"
$POSTGRES_HOST = "localhost"
$POSTGRES_PORT = "5435"
$REDIS_HOST = "localhost"
$REDIS_PORT = "6380"
$KAFKA_HOST = "localhost"
$KAFKA_PORT = "9092"
$PYTHON_ML_URL = "http://localhost:8000"

# Test 1: Eureka Server
Write-Host "[1/10] Testing Eureka Server..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "$EUREKA_URL/actuator/health" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "  ✅ Eureka Server is UP" -ForegroundColor Green
    }
} catch {
    Write-Host "  ❌ Eureka Server is DOWN" -ForegroundColor Red
    Write-Host "     Please start: docker-compose up -d eureka-server" -ForegroundColor Gray
}

# Test 2: Check if Recommendation Service is registered with Eureka
Write-Host "[2/10] Checking Recommendation Service registration..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$EUREKA_URL/eureka/apps/RECOMMENDATION-SERVICE" -UseBasicParsing -TimeoutSec 5
    if ($response) {
        Write-Host "  ✅ Recommendation Service registered with Eureka" -ForegroundColor Green
    }
} catch {
    Write-Host "  ❌ Recommendation Service NOT registered" -ForegroundColor Red
    Write-Host "     Service may not be running or not yet registered" -ForegroundColor Gray
}

# Test 3: Recommendation Service Health
Write-Host "[3/10] Testing Recommendation Service health..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$RECOMMEND_URL/actuator/health" -UseBasicParsing -TimeoutSec 5
    if ($response.status -eq "UP") {
        Write-Host "  ✅ Recommendation Service is UP" -ForegroundColor Green
        Write-Host "     Components:" -ForegroundColor Gray
        $response.components.PSObject.Properties | ForEach-Object {
            $status = $_.Value.status
            $color = if ($status -eq "UP") { "Green" } else { "Red" }
            Write-Host "       - $($_.Name): $status" -ForegroundColor $color
        }
    }
} catch {
    Write-Host "  ❌ Recommendation Service is DOWN" -ForegroundColor Red
    Write-Host "     Please start: docker-compose up -d recommendation-service" -ForegroundColor Gray
}

# Test 4: API Gateway
Write-Host "[4/10] Testing API Gateway..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "$API_GATEWAY_URL/actuator/health" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "  ✅ API Gateway is UP" -ForegroundColor Green
    }
} catch {
    Write-Host "  ❌ API Gateway is DOWN" -ForegroundColor Red
    Write-Host "     Please start: docker-compose up -d api-gateway" -ForegroundColor Gray
}

# Test 5: Neo4j
Write-Host "[5/10] Testing Neo4j..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri $NEO4J_URL -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "  ✅ Neo4j is UP" -ForegroundColor Green
    }
} catch {
    Write-Host "  ❌ Neo4j is DOWN" -ForegroundColor Red
    Write-Host "     Please start: docker-compose up -d neo4j" -ForegroundColor Gray
}

# Test 6: PostgreSQL
Write-Host "[6/10] Testing PostgreSQL (recommend_db)..." -ForegroundColor Yellow
try {
    $result = Test-NetConnection -ComputerName $POSTGRES_HOST -Port $POSTGRES_PORT -WarningAction SilentlyContinue
    if ($result.TcpTestSucceeded) {
        Write-Host "  ✅ PostgreSQL is UP (port $POSTGRES_PORT)" -ForegroundColor Green
    } else {
        Write-Host "  ❌ PostgreSQL is DOWN" -ForegroundColor Red
    }
} catch {
    Write-Host "  ❌ PostgreSQL is DOWN" -ForegroundColor Red
    Write-Host "     Please start: docker-compose up -d recommend-postgres" -ForegroundColor Gray
}

# Test 7: Redis
Write-Host "[7/10] Testing Redis..." -ForegroundColor Yellow
try {
    $result = Test-NetConnection -ComputerName $REDIS_HOST -Port $REDIS_PORT -WarningAction SilentlyContinue
    if ($result.TcpTestSucceeded) {
        Write-Host "  ✅ Redis is UP (port $REDIS_PORT)" -ForegroundColor Green
    } else {
        Write-Host "  ❌ Redis is DOWN" -ForegroundColor Red
    }
} catch {
    Write-Host "  ❌ Redis is DOWN" -ForegroundColor Red
    Write-Host "     Please start: docker-compose up -d recommend-redis" -ForegroundColor Gray
}

# Test 8: Kafka
Write-Host "[8/10] Testing Kafka..." -ForegroundColor Yellow
try {
    $result = Test-NetConnection -ComputerName $KAFKA_HOST -Port $KAFKA_PORT -WarningAction SilentlyContinue
    if ($result.TcpTestSucceeded) {
        Write-Host "  ✅ Kafka is UP (port $KAFKA_PORT)" -ForegroundColor Green
    } else {
        Write-Host "  ❌ Kafka is DOWN" -ForegroundColor Red
    }
} catch {
    Write-Host "  ❌ Kafka is DOWN" -ForegroundColor Red
    Write-Host "     Please start: docker-compose up -d kafka" -ForegroundColor Gray
}

# Test 9: Python ML Service
Write-Host "[9/10] Testing Python ML Service..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$PYTHON_ML_URL/health" -UseBasicParsing -TimeoutSec 5
    if ($response.status -eq "healthy") {
        Write-Host "  ✅ Python ML Service is UP" -ForegroundColor Green
        Write-Host "     Model loaded: $($response.model_loaded)" -ForegroundColor Gray
    }
} catch {
    Write-Host "  ❌ Python ML Service is DOWN" -ForegroundColor Red
    Write-Host "     Please start: docker-compose up -d recommend-python" -ForegroundColor Gray
}

# Test 10: Check Kafka Topics
Write-Host "[10/10] Checking Kafka Topics..." -ForegroundColor Yellow
try {
    $topics = docker exec kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list 2>$null
    $requiredTopics = @("post_created", "post_updated", "post_deleted", "user_action")
    
    Write-Host "  Topics found:" -ForegroundColor Gray
    foreach ($topic in $requiredTopics) {
        if ($topics -contains $topic) {
            Write-Host "    ✅ $topic" -ForegroundColor Green
        } else {
            Write-Host "    ⚠️  $topic (will be auto-created)" -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host "  ⚠️  Could not list Kafka topics" -ForegroundColor Yellow
    Write-Host "     Topics will be auto-created on first use" -ForegroundColor Gray
}

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Integration Test Complete!" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Check any failed services above" -ForegroundColor Gray
Write-Host "  2. Review logs: docker-compose logs -f recommendation-service" -ForegroundColor Gray
Write-Host "  3. Test API: curl http://localhost:8095/actuator/health" -ForegroundColor Gray
Write-Host "  4. View docs: recommend-service/INTEGRATION_GUIDE.md" -ForegroundColor Gray
Write-Host ""
