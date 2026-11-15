@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=py -3"
where py >nul 2>&1 || set "PY=python"
%PY% "%~dp0选输出设备.py"
echo.
pause
