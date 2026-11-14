@echo off
setlocal
set "ROOT=%~dp0.."
pushd "%ROOT%"

title SETU Backend - FastAPI Engine (Port 8000)
color 0B

echo ===============================================================================
echo                SETU BACKEND ENGINE (FastAPI + PyTorch CUDA)
echo ===============================================================================
echo  Project Root : %CD%
echo  Python Venv  : %ROOT%\.venv\Scripts\python.exe
echo.
echo  [LINKS]
echo  - API Docs (Swagger UI) : http://localhost:8000/docs
echo  - Backend Health Check   : http://localhost:8000/health
echo  - Live WebSocket Stream  : ws://localhost:8000/api/v1/ws/recognize
echo  - GitHub Repository      : https://github.com/Isobit7/SETU
echo  - Online README          : https://github.com/Isobit7/SETU#readme
echo ===============================================================================
echo.

if not exist "%ROOT%\.venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at: %ROOT%\.venv\Scripts\activate.bat
    popd
    pause
    exit /b 1
)

echo Activating Python virtual environment...
call "%ROOT%\.venv\Scripts\activate.bat"
set "PYTHONPATH=%ROOT%"

echo Starting Uvicorn ASGI server on http://127.0.0.1:8000 ...
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

if errorlevel 1 (
    echo.
    echo [ERROR] Backend crashed or exited with an error code.
)

popd
echo.
echo Press any key to close this terminal...
pause >nul

