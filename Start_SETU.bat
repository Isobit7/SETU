@echo off
setlocal

REM One-click launcher for the SETU backend and frontend.
REM Update ROOT if your project folder is somewhere else.
set "ROOT=C:\MINI PROJ-SETU"

if not exist "%ROOT%\.venv\Scripts\python.exe" (
    echo ERROR: Python virtual environment not found:
    echo   %ROOT%\.venv\Scripts\python.exe
    echo Edit ROOT in this file to match your SETU project folder.
    pause
    exit /b 1
)

if not exist "%ROOT%\backend\app\main.py" (
    echo ERROR: SETU backend file was not found:
    echo   %ROOT%\backend\app\main.py
    echo Check that ROOT points to your SETU project folder.
    pause
    exit /b 1
)

if not exist "%ROOT%\frontend\package.json" (
    echo ERROR: Frontend package.json was not found:
    echo   %ROOT%\frontend\package.json
    echo Check that ROOT points to your SETU project folder.
    pause
    exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
    echo ERROR: npm was not found. Install Node.js, then reopen this file.
    pause
    exit /b 1
)

if not exist "%ROOT%\frontend\node_modules" (
    echo Frontend dependencies are missing. Installing them first...
    pushd "%ROOT%\frontend"
    call npm install
    if errorlevel 1 (
        popd
        echo ERROR: npm install failed. Read the output above and fix the dependency issue.
        pause
        exit /b 1
    )
    popd
)

echo Starting SETU backend and frontend in separate terminal windows...
start "SETU Backend - FastAPI" /D "%ROOT%" cmd /k ""%ROOT%\.venv\Scripts\python.exe" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload"
start "SETU Frontend - Vite" /D "%ROOT%\frontend" cmd /k "npm run dev -- --host 127.0.0.1"

echo.
echo Both startup windows have been opened.
echo Wait for the backend to report that Uvicorn is running and the frontend to show its Local URL.
echo Then open: http://localhost:5173
 echo.
echo Keep both terminal windows open while using SETU. Close them to stop the services.
endlocal
