$ErrorActionPreference = 'Stop'

$environmentRoot = Join-Path $env:LOCALAPPDATA 'CatVTON\catvton_env'
$python = Join-Path $environmentRoot 'Scripts\python.exe'
if (-not (Test-Path $python)) {
    throw 'CatVTON environment not found. Run setup_catvton.ps1 first.'
}

$env:CATVTON_BACKEND = 'auto'
$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
$listener.Start()
$port = $listener.LocalEndpoint.Port
$listener.Stop()
$env:GRADIO_SERVER_PORT = "$port"
Write-Host "Starting CatVTON at http://127.0.0.1:$port/"
& $python (Join-Path $PSScriptRoot 'catvton_app.py')
exit $LASTEXITCODE
