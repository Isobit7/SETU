@echo off
setlocal EnableDelayedExpansion
title SETU - Automated Environment Setup

echo ======================================================================
echo                SETU AUTOMATED ENVIRONMENT SETUP
echo ======================================================================
echo Checking environment prerequisites, virtual environments, and dependencies...
echo.

:: 1. Check Python
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found in your system PATH!
    echo Please install Python 3.10, 3.11, or 3.12 from python.org and ensure
    echo "Add python.exe to PATH" is checked during installation.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version 2^>^&1') do set "PY_VER=%%i"
echo [OK] %PY_VER% found.

:: 2. Check Node.js and npm
where node >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Node.js was not found in your system PATH!
    echo Please install Node.js 18+ LTS from nodejs.org to run the frontend.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('node -v 2^>^&1') do set "NODE_VER=%%i"
echo [OK] Node.js %NODE_VER% found.

:: 3. Setup Python Virtual Environment (.venv)
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [*] Creating isolated Python virtual environment in .venv ...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created successfully.
) else (
    echo [OK] Virtual environment (.venv) already exists.
)

:: 4. Install / Update Python Requirements
echo.
echo [*] Installing / validating Python dependencies from requirements.txt ...
.\.venv\Scripts\python.exe -m pip install --upgrade pip >nul 2>&1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Encountered an error installing Python requirements.
    pause
    exit /b 1
)
echo [OK] Python dependencies verified.

:: 5. Install / Update Frontend Dependencies
echo.
echo [*] Installing / validating React frontend dependencies ...
cd /d "%~dp0frontend"
call npm install
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install frontend packages via npm.
    cd /d "%~dp0"
    pause
    exit /b 1
)
echo [*] Verifying frontend production build ...
call npm run build
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Frontend build failed.
    cd /d "%~dp0"
    pause
    exit /b 1
)
cd /d "%~dp0"
echo [OK] Frontend packages and Vite build verified.

:: 6. Fast Verification Smoke Test
echo.
echo [*] Running fast system verification smoke test ...
.\.venv\Scripts\python.exe -m pytest tests/test_api.py -k "test_health_endpoint or test_subtitle_collision" --no-header -q
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Smoke test encountered warnings or failures. Review above output.
) else (
    echo [OK] Smoke test passed cleanly!
)

echo.
echo ======================================================================
echo          SETU ENVIRONMENT SETUP COMPLETED SUCCESSFULLY!
echo ======================================================================
echo You are ready to launch! Double-click "SETU.bat" to start the workstation.
echo.
pause
