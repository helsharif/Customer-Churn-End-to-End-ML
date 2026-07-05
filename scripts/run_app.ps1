param(
    [string]$ApiHost = "127.0.0.1",
    [int]$ApiPort = 8000,
    [int]$UiPort = 8502,
    [string]$CondaEnv = "churn_ml_env001"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ApiUrl = "http://${ApiHost}:${ApiPort}"

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
    Write-Host "FastAPI is already running at $ApiUrl."
}
else {
    Write-Host "Starting FastAPI at $ApiUrl ..."
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

Write-Host "Waiting for FastAPI health check ..."
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

Write-Host "Starting Streamlit at http://localhost:$UiPort ..."
Set-Location -LiteralPath $ProjectRoot
conda run -n $CondaEnv python -m streamlit run "$ProjectRoot/src/churn_ml/ui/app.py" --server.port $UiPort
