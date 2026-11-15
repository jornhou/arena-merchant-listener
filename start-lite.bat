@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=py -3"
where py >nul 2>&1 || set "PY=python"
echo 正在启动「战地行商 · 听声辨物」（省电模式）...
%PY% listen.py --light
if errorlevel 1 (
  echo.
  echo 启动失败：请先安装 Python 3.10+ 并执行 pip install -r requirements.txt
  pause
)
