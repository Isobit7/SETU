@echo off
setlocal enabledelayedexpansion
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
echo  [4] EVAL DASHBOARD  - Open Model Evaluation ^& Accuracy Benchmarks
echo  [5] FACTSHEET       - Open Executive Technical Project Factsheet
echo  [6] SHOWCASE PORTAL - Open Interactive Presentation Portal (HTML)
echo  [7] VIVA ^& EXAM     - Open Comprehensive Viva ^& Architecture Guide
echo  [8] SIGN LEXICON    - Open Visual ISL Vocabulary Guide (Alphabet, Phrases)
echo  [9] API DOCS (REST) - Open Swagger UI Documentation (http://localhost:8000/docs)
echo  [10] SETUP / RESET  - Run Automated Machine Setup (setup.bat)
echo  [0] EXIT
echo.
echo ===============================================================================
set /p "CHOICE=Enter your choice [0-10]: "

if "%CHOICE%"=="1" (
    call "%ROOT%\scripts\launchers\Start_SETU.bat"
    goto menu
)
if "%CHOICE%"=="2" (
    call "%ROOT%\scripts\launchers\Stop_SETU.bat"
    pause
    goto menu
)
if "%CHOICE%"=="3" (
    call "%ROOT%\scripts\launchers\Verify_SETU.bat"
    pause
    goto menu
)
if "%CHOICE%"=="4" (
    start http://localhost:5173/?tab=evaluation
    goto menu
)
if "%CHOICE%"=="5" (
    if exist "%ROOT%\docs\SETU_PROJECT_FACTSHEET.md" (
        start "" "%ROOT%\docs\SETU_PROJECT_FACTSHEET.md"
    )
    goto menu
)
if "%CHOICE%"=="6" (
    if exist "%ROOT%\docs\showcase\index.html" (
        start "" "%ROOT%\docs\showcase\index.html"
    ) else (
        echo [INFO] Showcase portal path: docs/showcase/index.html
    )
    goto menu
)
if "%CHOICE%"=="7" (
    if exist "%ROOT%\docs\viva_prep\SETU_EXAM_AND_VIVA_PREPARATION_GUIDE.md" (
        start "" "%ROOT%\docs\viva_prep\SETU_EXAM_AND_VIVA_PREPARATION_GUIDE.md"
    )
    goto menu
)
if "%CHOICE%"=="8" (
    if exist "%ROOT%\docs\vocabulary\README.md" (
        start "" "%ROOT%\docs\vocabulary\README.md"
    )
    goto menu
)
if "%CHOICE%"=="9" (
    start http://localhost:8000/docs
    goto menu
)
if "%CHOICE%"=="10" (
    call "%ROOT%\setup.bat"
    goto menu
)
if "%CHOICE%"=="0" (
    exit /b 0
)

echo Invalid choice. Please select 1 through 9.
timeout /t 2 >nul
goto menu
