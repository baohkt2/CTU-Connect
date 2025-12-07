# CTU Connect Recommendation Service - Setup Script
# PowerShell script for Windows

Write-Host "================================" -ForegroundColor Cyan
Write-Host "CTU Connect Recommendation Service" -ForegroundColor Cyan
Write-Host "Setup Script v1.0" -ForegroundColor Cyan
Write-Host "================================`n" -ForegroundColor Cyan

# Check prerequisites
Write-Host "Checking prerequisites..." -ForegroundColor Yellow

# Check Java
try {
    $javaVersion = java -version 2>&1 | Select-String "version"
    Write-Host "✓ Java found: $javaVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Java not found. Please install Java 17+" -ForegroundColor Red
    exit 1
}

# Check Maven
try {
    $mavenVersion = mvn -version 2>&1 | Select-String "Apache Maven"
    Write-Host "✓ Maven found: $mavenVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Maven not found. Please install Maven 3.8+" -ForegroundColor Red
    exit 1
}

# Check Docker
try {
    $dockerVersion = docker --version
    Write-Host "✓ Docker found: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker not found. Please install Docker Desktop" -ForegroundColor Red
    exit 1
}

Write-Host "`n================================`n" -ForegroundColor Cyan

# Menu
Write-Host "Select setup option:" -ForegroundColor Yellow
Write-Host "1. Quick Setup (Start all infrastructure + build service)"
Write-Host "2. Start Infrastructure Only (PostgreSQL + Neo4j + Redis)"
Write-Host "3. Build Service Only"
Write-Host "4. Run Service"
Write-Host "5. Full Setup + Run"
Write-Host "6. Stop All"
Write-Host "0. Exit"

$choice = Read-Host "`nEnter your choice"

switch ($choice) {
    "1" {
        Write-Host "`nStarting Quick Setup..." -ForegroundColor Yellow
        
        # Start infrastructure
        Write-Host "`n[1/4] Starting PostgreSQL with pgvector..." -ForegroundColor Cyan
        docker run -d --name recommend_db `
            -p 5435:5432 `
            -e POSTGRES_PASSWORD=postgres `
            -e POSTGRES_DB=recommendation_db `
            ankane/pgvector:latest
        
        Write-Host "[2/4] Starting Neo4j..." -ForegroundColor Cyan
        docker run -d --name neo4j-recommend `
            -p 7474:7474 -p 7687:7687 `
            -e NEO4J_AUTH=neo4j/password `
            neo4j:5.13.0
        
        Write-Host "[3/4] Starting Redis..." -ForegroundColor Cyan
        docker run -d --name redis-recommend `
            -p 6379:6379 `
            redis:7-alpine
        
        Write-Host "[4/4] Building service..." -ForegroundColor Cyan
        mvn clean package -DskipTests
        
        Write-Host "`n✓ Quick setup completed!" -ForegroundColor Green
        Write-Host "Next steps:" -ForegroundColor Yellow
        Write-Host "1. Wait 30 seconds for databases to initialize"
        Write-Host "2. Run: mvn spring-boot:run"
        Write-Host "3. Check health: curl http://localhost:8095/api/recommend/health"
    }
    
    "2" {
        Write-Host "`nStarting infrastructure..." -ForegroundColor Yellow
        
        Write-Host "Starting PostgreSQL..." -ForegroundColor Cyan
        docker run -d --name recommend_db `
            -p 5435:5432 `
            -e POSTGRES_PASSWORD=postgres `
            -e POSTGRES_DB=recommendation_db `
            ankane/pgvector:latest
        
        Write-Host "Starting Neo4j..." -ForegroundColor Cyan
        docker run -d --name neo4j-recommend `
            -p 7474:7474 -p 7687:7687 `
            -e NEO4J_AUTH=neo4j/password `
            neo4j:5.13.0
        
        Write-Host "Starting Redis..." -ForegroundColor Cyan
        docker run -d --name redis-recommend `
            -p 6379:6379 `
            redis:7-alpine
        
        Write-Host "`n✓ Infrastructure started!" -ForegroundColor Green
    }
    
    "3" {
        Write-Host "`nBuilding service..." -ForegroundColor Yellow
        mvn clean package -DskipTests
        Write-Host "`n✓ Build completed!" -ForegroundColor Green
    }
    
    "4" {
        Write-Host "`nStarting service..." -ForegroundColor Yellow
        Write-Host "Service will start on http://localhost:8095" -ForegroundColor Cyan
        mvn spring-boot:run
    }
    
    "5" {
        Write-Host "`nFull setup and run..." -ForegroundColor Yellow
        
        # Infrastructure
        Write-Host "`n[1/6] Starting PostgreSQL..." -ForegroundColor Cyan
        docker run -d --name recommend_db `
            -p 5435:5432 `
            -e POSTGRES_PASSWORD=postgres `
            -e POSTGRES_DB=recommendation_db `
            ankane/pgvector:latest
        
        Write-Host "[2/6] Starting Neo4j..." -ForegroundColor Cyan
        docker run -d --name neo4j-recommend `
            -p 7474:7474 -p 7687:7687 `
            -e NEO4J_AUTH=neo4j/password `
            neo4j:5.13.0
        
        Write-Host "[3/6] Starting Redis..." -ForegroundColor Cyan
        docker run -d --name redis-recommend `
            -p 6379:6379 `
            redis:7-alpine
        
        Write-Host "[4/6] Waiting for databases to initialize..." -ForegroundColor Cyan
        Start-Sleep -Seconds 30
        
        Write-Host "[5/6] Building service..." -ForegroundColor Cyan
        mvn clean package -DskipTests
        
        Write-Host "[6/6] Starting service..." -ForegroundColor Cyan
        Write-Host "`nService starting on http://localhost:8095" -ForegroundColor Green
        mvn spring-boot:run
    }
    
    "6" {
        Write-Host "`nStopping all containers..." -ForegroundColor Yellow
        docker stop recommend_db neo4j-recommend redis-recommend 2>$null
        docker rm recommend_db neo4j-recommend redis-recommend 2>$null
        Write-Host "✓ All containers stopped and removed!" -ForegroundColor Green
    }
    
    "0" {
        Write-Host "Exiting..." -ForegroundColor Yellow
        exit 0
    }
    
    default {
        Write-Host "Invalid choice!" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n================================`n" -ForegroundColor Cyan
