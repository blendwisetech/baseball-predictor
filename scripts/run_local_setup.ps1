# Run from PowerShell in repo root (baseball_predictor):
#   powershell -ExecutionPolicy Bypass -File scripts\run_local_setup.ps1
# Or: cd baseball_predictor; .\scripts\run_local_setup.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "==> pip: upgrade streamlit" -ForegroundColor Cyan
python -m pip install -U "streamlit>=1.36"

Write-Host "==> merge training data" -ForegroundColor Cyan
python -m ml.merge_training_data

Write-Host "==> ML bootstrap (train win + runs + eval)" -ForegroundColor Cyan
python -m ml.bootstrap_models

Write-Host "Done. Start UI with: streamlit run app/main.py" -ForegroundColor Green
