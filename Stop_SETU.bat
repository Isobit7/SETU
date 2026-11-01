@echo off
setlocal

REM SETU Clean Process Stopper
REM Safely terminates SETU FastAPI backend (port 8000) and Vite frontend (port 5173).

echo ===================================================
echo   Stopping SETU Translation Workstation Processes
echo ===================================================

REM 1. Free Port 8000 (FastAPI / Uvicorn Backend)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    echo Terminating backend process PID %%a on port 8000...
    taskkill /F /PID %%a >nul 2>&1
)

REM 2. Free Port 5173 (Vite Frontend)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5173 ^| findstr LISTENING') do (
    echo Terminating frontend process PID %%a on port 5173...
    taskkill /F /PID %%a >nul 2>&1
)

echo.
echo All SETU server ports (8000 and 5173) have been cleanly closed.
echo Workstation stopped successfully.
timeout /t 3 >nul
exit /b 0
