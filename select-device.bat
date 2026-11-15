@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=py -3"
where py >nul 2>&1 || set "PY=python"
%PY% "%~dp0select_device.py"
echo.
pause
