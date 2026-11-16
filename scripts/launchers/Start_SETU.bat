@echo off
setlocal enabledelayedexpansion

REM ===============================================================================
REM   SETU — Master One-Click Workstation Launcher
REM   Launches both Backend (FastAPI, Port 8000) and Frontend (Vite, Port 5173)
REM ===============================================================================

title SETU Workstation Launcher
color 0F

set "ROOT=%~dp0..\.."
for %%i in ("%ROOT%") do set "ROOT=%%~fi"

cls
echo ===============================================================================
echo                SETU ISL WORKSTATION -- ONE-CLICK LAUNCHER
echo          Bridging Silence and Sound Through Authentic Multimodal AI
echo ===============================================================================
echo.
echo  Root Directory: %ROOT%
echo.

REM --- 1. Validate Python Virtual Environment ---
if not exist "%ROOT%\.venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment was not found:
    echo         %ROOT%\.venv\Scripts\python.exe
    echo.
    echo Please verify that Python virtual environment is set up.
    pause
    exit /b 1
)

REM --- 2. Validate Node.js and npm ---
where npm >nul 2>nul
if errorlevel 1 (
    echo [ERROR] npm was not found. Please install Node.js from https://nodejs.org
    pause
    exit /b 1
)

REM --- 3. Clean Port Conflicts (8000 & 5173) ---
echo [1/3] Checking for previous running instances on ports 8000 and 5173...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr :8000 ^| findstr LISTENING') do (
    echo       Releasing occupied port 8000 (PID: %%a)...
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr :5173 ^| findstr LISTENING') do (
    echo       Releasing occupied port 5173 (PID: %%a)...
    taskkill /F /PID %%a >nul 2>&1
)

REM --- 4. Launch Backend and Frontend ---
echo.
echo [2/3] Launching SETU Backend (FastAPI + PyTorch CUDA)...
start "SETU Backend - FastAPI Engine" "%ROOT%\scripts\launchers\run_backend.bat"

echo.
echo [3/3] Launching SETU Frontend (React 18 + Vite Workstation)...
start "SETU Frontend - Vite Workstation" "%ROOT%\scripts\launchers\run_frontend.bat"

echo.
echo ===============================================================================
echo                         SETU ENGINES INITIALIZING
echo ===============================================================================
echo.
echo  Please keep both server terminal windows open while using SETU.
echo.
echo  -----------------------------------------------------------------------------
echo  * Frontend Workstation : http://localhost:5173
echo  * Interactive 3D Studio: http://localhost:5173 (Click "Practice Studio")
echo  * Backend REST API Docs: http://localhost:8000/docs
echo  * System Health Check  : http://localhost:8000/health
echo  * Live WebSocket Stream: ws://localhost:8000/api/v1/ws/recognize
echo  * GitHub Repository    : https://github.com/Isobit7/SETU
echo  * Online README Guide  : https://github.com/Isobit7/SETU#readme
echo  -----------------------------------------------------------------------------
echo.

echo Waiting 5 seconds for services to initialize...
timeout /t 5 /nobreak >nul

echo Opening SETU in your default browser...
start http://localhost:5173

echo.
echo ===============================================================================
echo   SETU is now active! To stop all services anytime, run: Stop_SETU.bat
echo ===============================================================================
echo.
endlocal
