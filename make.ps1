<#
.SYNOPSIS
    Windows task runner — the PowerShell twin of the Makefile.

.DESCRIPTION
    GNU make is not installed by default on Windows, and the Phase 0 exit criteria
    require `setup`, `test` and `lint` to be runnable from a clean clone. Both files
    must be kept in sync; the Makefile is the reference.

.EXAMPLE
    .\make.ps1 setup
    .\make.ps1 lint
    .\make.ps1 train-debug
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Target = 'help',

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

$Py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $Py)) {
    Write-Host "No virtualenv at .venv - creating one." -ForegroundColor Yellow
    python -m venv .venv
}
$Paths = @('src', 'tests', 'scripts')

function Invoke-Step {
    param([string]$Label, [scriptblock]$Body)
    Write-Host "==> $Label" -ForegroundColor Cyan
    & $Body
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Label (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

switch ($Target) {
    'help' {
        Write-Host 'VulnLoc tasks:' -ForegroundColor Cyan
        @(
            '  setup         Editable install with dev extras + pre-commit hooks'
            '  setup-cuda    Replace CPU torch with a CUDA build (-Rest = channel, default cu130)'
            '  gpu-check     Report the accelerator that device.kind=auto would select'
            '  test          Run the full test suite with coverage'
            '  test-fast     Run tests, skipping anything marked slow'
            '  lint          ruff + black --check + import-linter + mypy'
            '  format        Auto-format and auto-fix'
            '  format-check  Verify formatting without writing'
            '  typecheck     Static type check'
            '  imports       Enforce the L0->L6 layer rule'
            '  train-debug   Phase 0 smoke run: no-op loop, logs to MLflow'
            '  train-gpu     Same smoke run, but fail loudly if no GPU is usable'
            '  mlflow-ui     Browse tracked runs at http://127.0.0.1:5000'
            '  freeze        Pin the environment into requirements.lock.txt'
            '  docker-build  Build the CPU image'
            '  docker-train  Run the smoke run inside the image'
            '  clean         Remove caches and build leftovers'
        ) | ForEach-Object { Write-Host $_ }
    }
    'setup' {
        Invoke-Step 'pip bootstrap' { & $Py -m pip install --upgrade pip setuptools wheel }
        Invoke-Step 'install -e .[dev]' { & $Py -m pip install -e '.[dev]' }
        Invoke-Step 'pre-commit install' { & $Py -m pre_commit install }
    }
    'setup-cuda' {
        # The channel must match the host driver; `nvidia-smi` reports the maximum
        # CUDA version it supports.
        $channel = if ($Rest) { $Rest[0] } else { 'cu130' }
        Invoke-Step "install torch ($channel)" {
            & $Py -m pip install --index-url "https://download.pytorch.org/whl/$channel" `
                --extra-index-url https://pypi.org/simple --force-reinstall torch
        }
    }
    'gpu-check' {
        Invoke-Step 'accelerator' {
            & $Py -c "from vulnloc.training.device import log_accelerator, resolve_device; from vulnloc.utils.logging import setup_logging; setup_logging('INFO', json_to_file=False); log_accelerator(resolve_device('auto'))"
        }
    }
    'test'         { Invoke-Step 'pytest' { & $Py -m pytest @Rest } }
    'test-fast'    { Invoke-Step 'pytest (fast)' { & $Py -m pytest -m 'not slow' --no-cov -q @Rest } }
    'lint' {
        Invoke-Step 'black --check' { & $Py -m black --check @Paths }
        Invoke-Step 'ruff' { & $Py -m ruff check @Paths }
        Invoke-Step 'import-linter' { & $Py -m importlinter.cli lint }
        Invoke-Step 'mypy' { & $Py -m mypy src/vulnloc }
    }
    'format' {
        Invoke-Step 'black' { & $Py -m black @Paths }
        Invoke-Step 'ruff --fix' { & $Py -m ruff check --fix @Paths }
    }
    'format-check' { Invoke-Step 'black --check' { & $Py -m black --check @Paths } }
    'typecheck'    { Invoke-Step 'mypy' { & $Py -m mypy src/vulnloc } }
    'imports'      { Invoke-Step 'import-linter' { & $Py -m importlinter.cli lint } }
    'train-debug'  { Invoke-Step 'train (debug)' { & $Py scripts/train.py train=debug @Rest } }
    'train-gpu' {
        Invoke-Step 'train (gpu)' { & $Py scripts/train.py train=debug device.kind=cuda @Rest }
    }
    'mlflow-ui' {
        & $Py -m mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db `
            --default-artifact-root ./artifacts/mlartifacts
    }
    'freeze' {
        Invoke-Step 'pip freeze' { & $Py -m pip freeze --exclude-editable | Out-File -Encoding utf8 requirements.lock.txt }
    }
    'docker-build' { Invoke-Step 'docker build' { docker compose build } }
    'docker-train' { Invoke-Step 'docker run' { docker compose run --rm vulnloc python scripts/train.py train=debug } }
    'clean' {
        foreach ($p in '.pytest_cache', '.mypy_cache', '.ruff_cache', 'htmlcov', 'coverage.xml', '.coverage', 'build', 'dist') {
            if (Test-Path $p) { Remove-Item -Recurse -Force $p }
        }
        Get-ChildItem -Recurse -Directory -Filter '__pycache__' |
            Where-Object { $_.FullName -notlike '*\.venv\*' } |
            Remove-Item -Recurse -Force
        Write-Host 'Cleaned.' -ForegroundColor Green
    }
    default {
        Write-Host "Unknown target '$Target'. Run '.\make.ps1 help'." -ForegroundColor Red
        exit 2
    }
}
