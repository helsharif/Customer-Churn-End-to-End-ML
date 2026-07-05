param(
    [string]$ApiHost = "127.0.0.1",
    [int]$ApiPort = 8000,
    [int]$UiPort = 8502,
    [string]$CondaEnv = "churn_ml_env001"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ApiUrl = "http://${ApiHost}:${ApiPort}"
$UiUrl = "http://localhost:$UiPort"

Write-Host ""
Write-Host "Telco Churn MLOps App Launcher" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot" -ForegroundColor DarkGray
Write-Host "To close the app: press Ctrl+C in this Streamlit terminal, then close the FastAPI window." -ForegroundColor Yellow
Write-Host ""

function Test-ApiHealth {
    try {
        $response = Invoke-RestMethod -Uri "$ApiUrl/health" -TimeoutSec 2
        return $response.status -eq "ok"
    }
    catch {
        return $false
    }
}

if (Test-ApiHealth) {
    Write-Host "FastAPI is already running at $ApiUrl." -ForegroundColor Green
}
else {
    Write-Host "Starting FastAPI at $ApiUrl ..." -ForegroundColor Cyan
    $CondaCommand = Get-Command conda -ErrorAction Stop
    Start-Process -FilePath $CondaCommand.Source -WorkingDirectory $ProjectRoot -ArgumentList @(
        "run",
        "-n",
        $CondaEnv,
        "uvicorn",
        "churn_ml.api.main:app",
        "--reload",
        "--host",
        $ApiHost,
        "--port",
        $ApiPort
    )
}

Write-Host "Waiting for FastAPI health check ..." -ForegroundColor Yellow
$deadline = (Get-Date).AddSeconds(30)
$healthy = $false
while ((Get-Date) -lt $deadline) {
    if (Test-ApiHealth) {
        $healthy = $true
        break
    }
    Start-Sleep -Seconds 1
}

if (-not $healthy) {
    Write-Warning "FastAPI did not respond within 30 seconds. Streamlit will still start, but predictions may fail until the API is ready."
}
else {
    Write-Host "FastAPI health check passed: $ApiUrl/health" -ForegroundColor Green
}

Write-Host ""
Write-Host "Opening instructions" -ForegroundColor Cyan
Write-Host "Streamlit UI: $UiUrl" -ForegroundColor Green
Write-Host "FastAPI docs: $ApiUrl/docs" -ForegroundColor Green
Write-Host "MLflow UI: run the MLflow command separately, then open http://localhost:5000" -ForegroundColor DarkGray
Write-Host ""
Write-Host "Closing instructions" -ForegroundColor Cyan
Write-Host "1. Press Ctrl+C here to stop Streamlit." -ForegroundColor Yellow
Write-Host "2. Close the FastAPI PowerShell window, or press Ctrl+C in that window." -ForegroundColor Yellow
Write-Host ""
Write-Host "Starting Streamlit at $UiUrl ..." -ForegroundColor Cyan
Set-Location -LiteralPath $ProjectRoot
conda run -n $CondaEnv python -m streamlit run "$ProjectRoot/src/churn_ml/ui/app.py" --server.port $UiPort
