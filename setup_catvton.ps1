$ErrorActionPreference = 'Stop'

$projectRoot = $PSScriptRoot
$environmentRoot = Join-Path $env:LOCALAPPDATA 'CatVTON\catvton_env'
$python = Join-Path $environmentRoot 'Scripts\python.exe'

if (-not (Test-Path $python)) {
    py -3.10 -m venv $environmentRoot
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not create the environment. Install Python 3.10 and retry.'
    }
}

& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Could not upgrade pip in the CatVTON environment.' }

& $python -m pip install -r (Join-Path $projectRoot 'requirements-directml.txt')
if ($LASTEXITCODE -ne 0) { throw 'Could not install CatVTON inference dependencies.' }

& $python -c "import torch, torch_directml; device = torch_directml.device(); print('PyTorch:', torch.__version__); print('DirectML device:', device); print('DirectML test:', (torch.tensor([1]).to(device) + torch.tensor([2]).to(device)).item())"
if ($LASTEXITCODE -ne 0) {
    throw 'PyTorch/DirectML smoke test failed. Check the missing native DLL or GPU driver before launching CatVTON.'
}

Write-Host "Environment ready at $environmentRoot"
Write-Host 'Model weights are not downloaded by this setup script.'
