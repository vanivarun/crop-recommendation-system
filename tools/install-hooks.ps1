#!/usr/bin/env pwsh
try {
    $precommit = Get-Command pre-commit -ErrorAction SilentlyContinue
} catch {
    $precommit = $null
}
if (-not $precommit) {
    Write-Host "pre-commit not found — installing via pip..."
    python -m pip install --upgrade pip
    python -m pip install pre-commit
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install pre-commit. Please install it manually and re-run this script."
        exit 1
    }
}

Write-Host "Installing git hooks with pre-commit..."
pre-commit install
if ($LASTEXITCODE -ne 0) {
    Write-Error "pre-commit install failed. Check output above for details."
    exit 1
}
Write-Host "pre-commit hooks installed successfully."
