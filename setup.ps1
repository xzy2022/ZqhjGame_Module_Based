param(
    [string]$Python = 'python',
    [switch]$Vision
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    & $Python -m venv (Join-Path $projectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Creating .venv failed; use a full Python installation with venv support.' }
}
$package = if ($Vision) { "$projectRoot[vision]" } else { $projectRoot }
& $venvPython -m pip install -e $package
if ($LASTEXITCODE -ne 0) { throw 'Installing dependencies failed.' }
