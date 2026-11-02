@echo off
title Wi-Fi CSI Live Sensing Dashboard
color 0A
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python -m streamlit run app.py
pause
