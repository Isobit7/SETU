@echo off
setlocal
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

:menu
cls
title SETU — Indian Sign Language AI Translation Workstation
color 0B

echo ===============================================================================
echo       ____  _____ _____ _   _     ___ ____  _     __  __ ____  ____  
echo      / ___^|| ____|_   _^| ^| ^| ^|   ^|_ _/ ___^|^| ^|   ^|  \/  ^|  _ \^|  _ \ 
echo      \___ \^|  _^|   ^| ^| ^| ^| ^| ^|    ^| ^|\___ \^| ^|   ^| ^|\/^| ^| ^|_) ^| ^|_) ^|
echo       ___) ^| ^|___  ^| ^| ^| ^|_^| ^|    ^| ^| ___) ^| ^|___^| ^|  ^| ^|  __/^|  __/ 
echo      ^|____/^|_____^| ^|_^|  \___/    ^|___^|____/^|_____^|_^|  ^|_^|_^|   ^|_^|    
echo.
echo     SETU: Authentic Indian Sign Language (ISL) Multimodal AI Workstation
echo ===============================================================================
echo.
echo  [1] START SETU      - Launch Backend (8000) ^& Frontend (5173) Together
echo  [2] STOP SETU       - Safely Terminate All Background Services
echo  [3] VERIFY SETU     - Run Hardware Diagnostics, Models ^& Regression Tests
echo  [4] GITHUB REPO     - Open GitHub Repository ^& Online README
echo  [5] BROWSER OPEN    - Open Live Frontend (http://localhost:5173)
echo  [6] API DOCS        - Open Backend Swagger Docs (http://localhost:8000/docs)
echo  [7] EXIT
echo.
echo ===============================================================================
set /p "CHOICE=Enter your choice [1-7]: "

if "%CHOICE%"=="1" (
    call "%ROOT%\Start_SETU.bat"
    goto menu
)
if "%CHOICE%"=="2" (
    call "%ROOT%\Stop_SETU.bat"
    goto menu
)
if "%CHOICE%"=="3" (
    call "%ROOT%\Verify_SETU.bat"
    goto menu
)
if "%CHOICE%"=="4" (
    start https://github.com/Isobit7/SETU#readme
    goto menu
)
if "%CHOICE%"=="5" (
    start http://localhost:5173
    goto menu
)
if "%CHOICE%"=="6" (
    start http://localhost:8000/docs
    goto menu
)
if "%CHOICE%"=="7" (
    exit /b 0
)

echo Invalid choice. Please enter a number from 1 to 7.
timeout /t 2 >nul
goto menu

