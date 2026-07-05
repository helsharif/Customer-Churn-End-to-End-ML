param(
    [string]$ApiHost = "127.0.0.1",
    [int]$ApiPort = 8000,
    [string]$CondaEnv = "churn_ml_env001"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location -LiteralPath $ProjectRoot

conda run -n $CondaEnv uvicorn churn_ml.api.main:app --reload --host $ApiHost --port $ApiPort
