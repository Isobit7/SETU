@echo off
setlocal
set "ROOT=%~dp0.."
pushd "%ROOT%\frontend"

title SETU Frontend - Vite Workstation (Port 5173)
color 0A

echo ===============================================================================
echo                SETU FRONTEND WORKSTATION (React 18 + Vite)
echo ===============================================================================
echo  Project Root : %CD%
echo.
echo  [LINKS]
echo  - Live Workstation UI    : http://localhost:5173
echo  - Practice Studio 3D     : http://localhost:5173 (Select Practice Studio tab)
echo  - GitHub Repository      : https://github.com/Isobit7/SETU
echo  - Online README          : https://github.com/Isobit7/SETU#readme
echo ===============================================================================
echo.

where npm >nul 2>nul
if errorlevel 1 (
    echo [ERROR] npm command not found! Please install Node.js from https://nodejs.org
    popd
    pause
    exit /b 1
)

if not exist "%ROOT%\frontend\node_modules" (
    echo [INFO] node_modules missing. Running npm install...
    call npm install
    if errorlevel 1 (
        echo [ERROR] npm install failed.
        popd
        pause
        exit /b 1
    )
)

echo Starting Vite development server on http://127.0.0.1:5173 ...
npm run dev -- --host 127.0.0.1 --port 5173

if errorlevel 1 (
    echo.
    echo [ERROR] Frontend server crashed or stopped.
)

popd
echo.
echo Press any key to close this terminal...
pause >nul

