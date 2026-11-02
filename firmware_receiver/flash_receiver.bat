@echo off
title Flash ESP32 CSI Receiver (Member 2 Firmware)
color 0B
cd /d "%~dp0"

echo ================================================================
echo      ESP32 CSI Receiver 1-Click Flasher (Member 2)
echo ================================================================
echo.
echo Make sure ESP32 #2 is connected via USB.
echo.
set /p PORT=Enter COM Port (e.g. COM8, COM5, COM6) or press Enter to AUTO-DETECT: 

if "%PORT%"=="" (
    echo [INFO] Auto-detecting COM port and flashing...
    python -m esptool --chip esp32 --baud 460800 write_flash -z 0x1000 "bin\bootloader.bin" 0x8000 "bin\partition-table.bin" 0x10000 "bin\csi_receiver.bin"
) else (
    echo [INFO] Flashing on %PORT%...
    python -m esptool --chip esp32 --port %PORT% --baud 460800 write_flash -z 0x1000 "bin\bootloader.bin" 0x8000 "bin\partition-table.bin" 0x10000 "bin\csi_receiver.bin"
)

echo.
if %ERRORLEVEL% EQU 0 (
    echo ================================================================
    echo [SUCCESS] ESP32 Receiver Flashed Successfully!
    echo ================================================================
) else (
    echo [ERROR] Flashing failed. If needed, hold BOOT button on the ESP32 while flashing.
)
echo.
pause
