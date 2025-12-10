# Restart Frontend Dev Server
Write-Host "Restarting Frontend Dev Server..." -ForegroundColor Cyan

# Find and kill existing Next.js dev server on port 3000
Write-Host "Stopping existing dev server..." -ForegroundColor Yellow
$processes = Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
if ($processes) {
    foreach ($pid in $processes) {
        Write-Host "  Killing process $pid" -ForegroundColor Gray
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
}

# Clear Next.js cache
Write-Host "Clearing Next.js cache..." -ForegroundColor Yellow
if (Test-Path "client-frontend\.next") {
    Remove-Item -Path "client-frontend\.next" -Recurse -Force
    Write-Host "  Cache cleared" -ForegroundColor Green
}

# Start dev server
Write-Host ""
Write-Host "Starting dev server..." -ForegroundColor Green
Write-Host "Navigate to: http://localhost:3000/messages" -ForegroundColor Cyan
Write-Host ""

Set-Location client-frontend
npm run dev
