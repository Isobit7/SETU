@echo off
setlocal enabledelayedexpansion

REM ===============================================================================
REM   SETU — Master Pre-Flight Diagnostics and System Verification
REM ===============================================================================

title SETU Pre-Flight Diagnostics and Verification
color 0E

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

cls
echo ===============================================================================
echo                SETU ISL WORKSTATION -- PRE-FLIGHT VERIFICATION
echo ===============================================================================
echo  Project Root : %ROOT%
echo.

set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
set "PYTHONPATH=%ROOT%"

if not exist "%PYTHON%" (
    echo [FAIL] Python virtual environment not detected at:
    echo        %PYTHON%
    goto :error
)

echo [STEP 1/5] Checking PyTorch and Hardware Acceleration...
"%PYTHON%" -c "import torch; print('  [OK] PyTorch Version:', torch.__version__); print('  [OK] CUDA Available:', torch.cuda.is_available()); print('  [OK] GPU Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Fallback')"
if errorlevel 1 (
    echo [FAIL] PyTorch runtime check failed.
    goto :error
)
echo.

echo [STEP 2/5] Checking Registered ML Model Zoo...
"%PYTHON%" -c "from models.registry import ModelRegistry; models = ModelRegistry.list_models(); print(f'  [OK] {len(models)} Production Models Registered: ' + ', '.join(m['id'] for m in models))"
if errorlevel 1 (
    echo [FAIL] ModelRegistry discovery check failed.
    goto :error
)
echo.

echo [STEP 3/5] Running Core Vision, NLP, and Multilingual Test Suite...
"%PYTHON%" -m pytest tests/test_one_euro_filter.py tests/test_kinetic_segmenter.py tests/test_syntax_transformer.py tests/test_neural_translator.py tests/test_indic_languages.py -q
if errorlevel 1 (
    echo [FAIL] Core backend unit tests failed.
    goto :error
)
echo   [OK] Core test suites passed cleanly with 100%% green status.
echo.

echo [STEP 4/5] Verifying Frontend Production Build (TypeScript + Vite)...
call npm --prefix "%ROOT%\frontend" run build
if errorlevel 1 (
    echo [FAIL] Frontend TypeScript/Vite compilation failed.
    goto :error
)
echo   [OK] Frontend build compiled cleanly with zero errors.
echo.

echo [STEP 5/5] Checking System Ports (8000 and 5173)...
set "PORT8000=FREE"
set "PORT5173=FREE"
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr :8000 ^| findstr LISTENING') do set "PORT8000=IN USE (PID: %%a)"
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr :5173 ^| findstr LISTENING') do set "PORT5173=IN USE (PID: %%a)"

echo   Port 8000 (Backend API):  !PORT8000!
echo   Port 5173 (Frontend UI):  !PORT5173!
if not "!PORT8000!"=="FREE" (
    echo   [NOTE] Port 8000 is active. Run Stop_SETU.bat if you need a clean restart.
)
if not "!PORT5173!"=="FREE" (
    echo   [NOTE] Port 5173 is active. Run Stop_SETU.bat if you need a clean restart.
)
echo.

echo ===============================================================================
echo                [SUCCESS] ALL SETU ENGINES FULLY OPERATIONAL!
echo ===============================================================================
echo  The workstation is verified and ready for live demonstration.
echo.
echo  * To Launch Workstation : Run Start_SETU.bat
echo  * To Stop Workstation   : Run Stop_SETU.bat
echo  * Frontend Workstation  : http://localhost:5173
echo  * Backend API Docs      : http://localhost:8000/docs
echo  * GitHub Repository     : https://github.com/Isobit7/SETU
echo  * Online README Guide   : https://github.com/Isobit7/SETU#readme
echo ===============================================================================
echo.
pause
exit /b 0

:error
echo.
echo ===============================================================================
echo                [ERROR] SYSTEM VERIFICATION DETECTED AN ISSUE
echo ===============================================================================
echo.
pause
exit /b 1
