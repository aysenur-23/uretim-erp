$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendPython = Join-Path $Root ".venv\Scripts\python.exe"

Start-Process `
  -FilePath $BackendPython `
  -ArgumentList @("-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload") `
  -WorkingDirectory $Root `
  -WindowStyle Hidden

Start-Process `
  -FilePath "npm.cmd" `
  -ArgumentList @("run", "dev") `
  -WorkingDirectory (Join-Path $Root "frontend") `
  -WindowStyle Hidden

Write-Host "MEGA QC backend:  http://127.0.0.1:8000"
Write-Host "MEGA QC frontend: http://localhost:3000"
