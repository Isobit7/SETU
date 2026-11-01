@echo off
setlocal enabledelayedexpansion
title SETU Pre-Flight Diagnostics and System Verification

echo ===============================================================================
echo                SETU ISL WORKSTATION -- PRE-FLIGHT VERIFICATION
echo ===============================================================================
echo.

set "PYTHON=.venv\Scripts\python.exe"

if not exist "!PYTHON!" (
    echo [FAIL] Python virtual environment not detected at .venv\Scripts\python.exe
    echo        Please run setup_env.bat before running verification.
    goto :error
)

echo [STEP 1/5] Checking PyTorch and Hardware Acceleration...
!PYTHON! -c "import torch; print('  [OK] PyTorch Version:', torch.__version__); print('  [OK] CUDA Available:', torch.cuda.is_available()); print('  [OK] GPU Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Fallback')"
if errorlevel 1 (
    echo [FAIL] PyTorch runtime check failed.
    goto :error
)
echo.

echo [STEP 2/5] Checking Registered ML Model Zoo...
!PYTHON! -c "from models.registry import ModelRegistry; models = ModelRegistry.list_models(); print(f'  [OK] {len(models)} Models Registered: ' + ', '.join(m['id'] for m in models))"
if errorlevel 1 (
    echo [FAIL] ModelRegistry discovery check failed.
    goto :error
)
echo.

echo [STEP 3/5] Running Core Vision, NLP, and Model Test Suite...
!PYTHON! -m pytest tests/test_one_euro_filter.py tests/test_kinetic_segmenter.py tests/test_syntax_transformer.py tests/test_neural_translator.py -q
if errorlevel 1 (
    echo [FAIL] Core backend unit tests failed.
    goto :error
)
echo   [OK] Core test suites passed cleanly.
echo.

echo [STEP 4/5] Verifying Frontend Production Build (TypeScript + Vite)...
call npm --prefix frontend run build
if errorlevel 1 (
    echo [FAIL] Frontend TypeScript/Vite compilation failed.
    goto :error
)
echo   [OK] Frontend build compiled cleanly with zero errors.
echo.

echo [STEP 5/5] Checking System Ports (8000 and 5173)...
set "PORT8000=FREE"
set "PORT5173=FREE"
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING 2^>nul') do set "PORT8000=IN USE (PID: %%a)"
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5173 ^| findstr LISTENING 2^>nul') do set "PORT5173=IN USE (PID: %%a)"

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
echo         The workstation is completely verified and ready for live demo.
echo         To start: Run Start_SETU.bat
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
