@echo off
cd /d "%~dp0"
title YOAC Player Tracker
echo Checking what the app needs (the first time takes a few minutes)...
py -m pip install --quiet --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
  echo.
  echo Could not install the requirements. Is Python installed? See README.md.
  pause
  exit /b 1
)
py app.py
pause
