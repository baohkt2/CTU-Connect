# Stop Recommendation Service (Development Mode)

Write-Host "`n╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Recommendation Service - Stopping Development Environment  ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

Write-Host "🛑 Stopping services..." -ForegroundColor Yellow

docker-compose -f docker-compose.dev.yml down

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Services stopped successfully!" -ForegroundColor Green
    
    Write-Host "`n📊 Cleanup Options:" -ForegroundColor Cyan
    Write-Host "   Keep data:     Done! (Data preserved in volumes)" -ForegroundColor White
    Write-Host "   Remove data:   docker-compose -f docker-compose.dev.yml down -v" -ForegroundColor Yellow
    
    Write-Host ""
} else {
    Write-Host "`n❌ Error stopping services!" -ForegroundColor Red
    exit 1
}
