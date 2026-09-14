param(
    [string]$Python,
    [string]$RuntimeRoot,
    [switch]$Vision
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$runtime = if ($RuntimeRoot) { $RuntimeRoot } elseif ($env:HF2026_RUNTIME_ROOT) { $env:HF2026_RUNTIME_ROOT } else { Split-Path -Parent $projectRoot }
if (-not $Python) {
    $Python = Join-Path $runtime 'python\python.exe'
    if (-not (Test-Path -LiteralPath $Python)) {
        throw 'Official Python not found. Specify -RuntimeRoot <official release> or -Python <python.exe>.'
    }
}
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    & $Python -m venv (Join-Path $projectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Creating .venv failed; use a full Python installation with venv support.' }
}
$package = if ($Vision) { "$projectRoot[vision]" } else { $projectRoot }
& $venvPython -m pip install -e $package
if ($LASTEXITCODE -ne 0) { throw 'Installing dependencies failed.' }
