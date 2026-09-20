$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$sdkRoot = if ($env:HF2026_SIM_ROOT) { $env:HF2026_SIM_ROOT } else { Split-Path -Parent $projectRoot }
$runtimeRoot = $env:HF2026_RUNTIME_ROOT
for ($i = 0; $i -lt $args.Count - 1; $i++) {
    if ($args[$i] -eq '--sim-root') { $sdkRoot = $args[$i + 1] }
    if ($args[$i] -eq '--runtime-root') { $runtimeRoot = $args[$i + 1] }
}
$candidates = @()
if ($env:HF2026_PYTHON) { $candidates += $env:HF2026_PYTHON }
$candidates += Join-Path $projectRoot '.venv\Scripts\python.exe'
if ($runtimeRoot) { $candidates += Join-Path $runtimeRoot 'python\python.exe' }
$candidates += Join-Path $sdkRoot 'python\python.exe'
$pythonExe = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $pythonExe) { throw 'Python not found. Run setup.ps1 or specify an official runtime root.' }
& $pythonExe -B -X utf8 (Join-Path $PSScriptRoot 'run_official.py') @args
exit $LASTEXITCODE
