# Load Test Data into Databases

Write-Host "╔════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Loading Test Data for Recommendation     ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check if containers are running
Write-Host "Checking databases..." -ForegroundColor Yellow
$postgres = docker ps --filter "name=postgres-recommend-dev" --format "{{.Names}}"
$neo4j = docker ps --filter "name=neo4j-recommend-dev" --format "{{.Names}}"

if (-not $postgres) {
    Write-Host "✗ PostgreSQL container not running!" -ForegroundColor Red
    Write-Host "  Run: .\start-dev.ps1" -ForegroundColor Yellow
    exit 1
}

if (-not $neo4j) {
    Write-Host "✗ Neo4j container not running!" -ForegroundColor Red
    Write-Host "  Run: .\start-dev.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Databases are running" -ForegroundColor Green
Write-Host ""

# Load PostgreSQL data
Write-Host "Loading PostgreSQL test data..." -ForegroundColor Yellow
Get-Content test-data.sql | docker exec -i postgres-recommend-dev psql -U postgres -d recommendation_db
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ PostgreSQL data loaded" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to load PostgreSQL data" -ForegroundColor Red
}
Write-Host ""

# Load Neo4j data
Write-Host "Loading Neo4j test data..." -ForegroundColor Yellow
Write-Host "  Open Neo4j Browser: http://localhost:7474" -ForegroundColor Cyan
Write-Host "  Login: neo4j / password" -ForegroundColor Cyan
Write-Host "  Copy content from: test-data.cypher" -ForegroundColor Cyan
Write-Host "  Or use cypher-shell (if installed)" -ForegroundColor Cyan
Write-Host ""

# Try to load via cypher-shell
Write-Host "Attempting to load via cypher-shell..." -ForegroundColor Yellow
Get-Content test-data.cypher | docker exec -i neo4j-recommend-dev cypher-shell -u neo4j -p password
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Neo4j data loaded" -ForegroundColor Green
} else {
    Write-Host "! Please load Neo4j data manually" -ForegroundColor Yellow
    Write-Host "  1. Open: http://localhost:7474" -ForegroundColor White
    Write-Host "  2. Copy-paste: test-data.cypher" -ForegroundColor White
}
Write-Host ""

Write-Host "════════════════════════════════════════════" -ForegroundColor Green
Write-Host "✅ Test data loading complete!" -ForegroundColor Green
Write-Host "════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Verify data: .\verify-test-data.ps1" -ForegroundColor White
Write-Host "  2. Test APIs: .\test-api.ps1" -ForegroundColor White
Write-Host ""
