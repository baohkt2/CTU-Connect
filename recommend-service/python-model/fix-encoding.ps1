# Script to fix Unicode encoding issues on Windows
# Run this before starting the Python service

Write-Host "=== Fixing Windows Console Encoding ===" -ForegroundColor Green
Write-Host ""

# Set console code page to UTF-8 (65001)
Write-Host "Setting console encoding to UTF-8..." -ForegroundColor Cyan
chcp 65001 | Out-Null

# Set PowerShell output encoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Set environment variable for Python
$env:PYTHONIOENCODING = "utf-8"

Write-Host "✓ Console encoding set to UTF-8" -ForegroundColor Green
Write-Host "✓ PYTHONIOENCODING=utf-8" -ForegroundColor Green
Write-Host ""

Write-Host "You can now run:" -ForegroundColor Cyan
Write-Host "  .\run-dev.ps1" -ForegroundColor White
Write-Host ""

# Verify encoding
Write-Host "Current encoding: $([Console]::OutputEncoding.EncodingName)" -ForegroundColor Gray
Write-Host ""
