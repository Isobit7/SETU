@echo off
setlocal
cd /d "%~dp0"
title SETU - RealSign 26-Class Alphabet Recognizer
echo ===========================================================================
echo SETU - RealSign 26-Class ISL Alphabet Real-Time Webcam Demo
echo ===========================================================================
echo.
echo Launching camera feed and MediaPipe hand landmarker...
echo Press 'q' or 'ESC' on the video window to exit.
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    pause
    exit /b 1
)

".venv\Scripts\python.exe" scripts\demo_realsign_26class_webcam.py

echo.
echo Demo session ended.
pause
