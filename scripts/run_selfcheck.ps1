# scripts/run_selfcheck.ps1
#
# Offline self-check for P5 dataset.py: vocab round-trip + HTML→text.
# No downloads, no PDFs, no network. Fast; safe to run any time.
#
# Run from the project root:
#     powershell -ExecutionPolicy Bypass -File scripts/run_selfcheck.ps1

$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

$venvPy = Join-Path $PSScriptRoot "..\generative_env\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    throw "generative_env not found at $venvPy. Create the venv first."
}

& $venvPy -X utf8 src/dataset.py
