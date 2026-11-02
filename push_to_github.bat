@echo off
title Push Wi-Fi CSI Project to GitHub
color 0A
cd /d "%~dp0"

echo ================================================================================
echo             PUSH WI-FI CSI PROJECT TO GITHUB (saifadel20071234-sudo)
echo ================================================================================
echo.
echo 1) Open GitHub and create a new repository: https://github.com/new
echo    - Repository name: e.g. wifi-csi-human-sensing
echo    - Do NOT initialize with README, .gitignore or license (keep it empty).
echo.
echo 2) Paste your repository link below.
echo.
set /p REPO_URL="Enter GitHub Repository URL: "

if "%REPO_URL%"=="" (
    echo [ERROR] No URL entered. Aborted.
    pause
    exit /b
)

echo.
echo [*] Setting up remote origin...
git remote remove origin 2>nul
git remote add origin %REPO_URL%
git branch -M main

echo [*] Pushing all files to GitHub...
git push -u origin main

echo.
if %ERRORLEVEL% EQU 0 (
    echo ================================================================================
    echo [SUCCESS] Project successfully uploaded to GitHub!
    echo ================================================================================
) else (
    echo ================================================================================
    echo [ERROR] Push failed. Ensure you have permissions and are signed in to GitHub.
    echo ================================================================================
)
echo.
pause
