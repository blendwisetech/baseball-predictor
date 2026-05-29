# Standalone MLB predictor (not the multi-sport hub).
# Usage (from anywhere):
#   powershell -ExecutionPolicy Bypass -File scripts\run_baseball_app.ps1
#
# Opens http://localhost:8510 (see .streamlit/config.toml).

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

# Ensure we are NOT embedded in the sports hub (hub sets SPORTS_HUB=1).
Remove-Item Env:SPORTS_HUB -ErrorAction SilentlyContinue

Write-Host "Starting Baseball Predictor only on port 8510..." -ForegroundColor Green
Write-Host "URL: http://localhost:8510" -ForegroundColor Cyan
python -m streamlit run app/main.py
