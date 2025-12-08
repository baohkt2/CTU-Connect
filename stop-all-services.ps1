# Stop All CTU-Connect Services

Write-Host "`n╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║         CTU-Connect - Stop All Services                     ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

Write-Host "🛑 Stopping all services..." -ForegroundColor Yellow

docker-compose down

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ All services stopped successfully!" -ForegroundColor Green
    
    Write-Host "`n📊 Cleanup Options:" -ForegroundColor Cyan
    Write-Host "   Keep data:     Done! (Data preserved in volumes)" -ForegroundColor White
    Write-Host "   Remove data:   docker-compose down -v" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   View volumes:  docker volume ls | findstr ctu" -ForegroundColor White
    
    Write-Host ""
} else {
    Write-Host "`n❌ Error stopping services!" -ForegroundColor Red
    exit 1
}
