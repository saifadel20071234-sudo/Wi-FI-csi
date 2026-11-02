@echo off
title ESP32 CP2102 Driver Auto-Fix
color 0A
echo ====================================================
echo      Installing Official CP2102 Driver for ESP32
echo ====================================================
echo.
cd /d "C:\Users\Adel\Desktop\Wi-Fi csi\CP210x_Driver_Setup"
pnputil /add-driver slabvcp.inf /install
echo.
echo ====================================================
echo      Scanning for connected ESP32...
echo ====================================================
pnputil /scan-devices
echo.
echo [DONE] The driver has been installed successfully!
echo Please open Arduino IDE now and check your Port.
echo.
pause
