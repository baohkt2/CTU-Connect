# Stop Development Environment

Write-Host "Stopping databases..." -ForegroundColor Yellow
docker-compose -f docker-compose.dev.yml down

Write-Host "✓ Databases stopped" -ForegroundColor Green
