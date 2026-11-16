@echo off
setlocal

REM ===============================================================================
REM   SETU — Master Clean Process Stopper
REM   Safely terminates SETU FastAPI backend (port 8000) and Vite frontend (port 5173)
REM ===============================================================================

title SETU Shutdown
color 0C

echo ===============================================================================
echo                SETU ISL WORKSTATION -- CLEAN SHUTDOWN
echo ===============================================================================
echo.

set "KILLED_ANY=0"

REM 1. Free Port 8000 (FastAPI / Uvicorn Backend)
echo [1/2] Checking Port 8000 (FastAPI Backend)...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr :8000 ^| findstr LISTENING') do (
    echo       Terminating backend process PID %%a on port 8000...
    taskkill /F /PID %%a >nul 2>&1
    set "KILLED_ANY=1"
)
if "!KILLED_ANY!"=="0" (
    echo       Port 8000 is clean (no active backend processes).
)

REM 2. Free Port 5173 (Vite Frontend)
set "KILLED_FE=0"
echo.
echo [2/2] Checking Port 5173 (Vite Frontend)...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr :5173 ^| findstr LISTENING') do (
    echo       Terminating frontend process PID %%a on port 5173...
    taskkill /F /PID %%a >nul 2>&1
    set "KILLED_FE=1"
)
if "!KILLED_FE!"=="0" (
    echo       Port 5173 is clean (no active frontend processes).
)

echo.
echo ===============================================================================
echo   All SETU background processes have been safely terminated.
echo   Ports 8000 and 5173 are now completely free.
echo.
echo   * To restart SETU: Run Start_SETU.bat
echo   * GitHub Repo    : https://github.com/Isobit7/SETU
echo   * Online README  : https://github.com/Isobit7/SETU#readme
echo ===============================================================================
echo.
timeout /t 3 >nul
exit /b 0
