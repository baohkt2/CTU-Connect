# Start Chat Database
Write-Host "Starting Chat Database (MongoDB)..." -ForegroundColor Cyan

# Start chat_db container
docker-compose up -d chat_db

# Wait for database to be healthy
Write-Host "Waiting for MongoDB to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check status
docker-compose ps chat_db

Write-Host ""
Write-Host "Chat Database Status:" -ForegroundColor Green
Write-Host "  MongoDB: localhost:27019" -ForegroundColor White
Write-Host "  Database: chat_db" -ForegroundColor White
Write-Host ""
Write-Host "To start chat-service in IDE, run:" -ForegroundColor Cyan
Write-Host "  cd chat-service" -ForegroundColor White
Write-Host "  ./mvnw spring-boot:run" -ForegroundColor White
Write-Host ""
Write-Host "Or use Maven in your IDE to run ChatServiceApplication" -ForegroundColor Cyan
