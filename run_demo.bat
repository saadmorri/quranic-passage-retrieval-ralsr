@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_demo.ps1"
if errorlevel 1 (
  echo.
  echo The demo did not start. See runtime\backend_stderr.log for details.
  pause
)
endlocal
