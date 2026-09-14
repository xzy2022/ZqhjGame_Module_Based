@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\launch.ps1" %*
exit /b %errorlevel%
