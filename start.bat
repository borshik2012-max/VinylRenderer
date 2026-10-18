@echo off
chcp 65001 >nul
title Vinyl Visualizer
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python не найден. Установите Python 3.10+ с python.org ^(галочка "Add python.exe to PATH"^).
  pause
  exit /b 1
)
python -c "import numpy, PIL" >nul 2>nul
if errorlevel 1 (
  echo Устанавливаю numpy и Pillow...
  python -m pip install --user numpy pillow
)
python app\server.py %*
pause
