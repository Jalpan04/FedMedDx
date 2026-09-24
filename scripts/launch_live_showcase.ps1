# FedMedDx Live Presentation One-Click Launcher
# Launches the Central Server and 2 Fast Hospital Nodes (COVID + Pneumonia) in separate PowerShell processes.
# Each round executes in ~3 seconds.

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "        FedMedDx Live Federated Showcase Launcher         " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$VENV_PYTHON = ".\.venv\Scripts\python.exe"

# 1. Start Server
Write-Host "[1/3] Launching Flower Coordinator Server on port 8080..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; Write-Host '--- CENTRAL SERVER NODE ---' -ForegroundColor Cyan; $VENV_PYTHON -m federated.server --port 8080 --rounds 5 --min_clients 2"

Start-Sleep -Seconds 2

# 2. Start Hospital 0 (COVID-19 Node)
Write-Host "[2/3] Launching Hospital 0 (COVID-19 Radiography Node)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; Write-Host '--- HOSPITAL 0: COVID-19 NODE ---' -ForegroundColor Yellow; $VENV_PYTHON -m federated.client --server 127.0.0.1:8080 --modality covid --hospital_id 0 --fast"

Start-Sleep -Seconds 1

# 3. Start Hospital 1 (Pneumonia Node)
Write-Host "[3/3] Launching Hospital 1 (Pneumonia Detection Node)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; Write-Host '--- HOSPITAL 1: PNEUMONIA NODE ---' -ForegroundColor Magenta; $VENV_PYTHON -m federated.client --server 127.0.0.1:8080 --modality pneumonia --hospital_id 1 --fast"

Write-Host "`nAll 3 federated nodes launched successfully in Fast Demo Mode!" -ForegroundColor Green
Write-Host "Communication rounds will complete in ~3 seconds each." -ForegroundColor White
