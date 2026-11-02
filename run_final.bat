@echo off
title Wi-Fi CSI Master Runner (Live Hardware ^& Dashboard)
color 0B
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

echo ================================================================================
echo             WI-FI CSI HUMAN SENSING ^& SPATIAL TRACKING SYSTEM
echo                          Starting Master Runner...
echo ================================================================================
echo.

python run_final.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ================================================================================
    echo [ERROR] An unexpected error occurred while executing run_final.py.
    echo ================================================================================
    pause
)
