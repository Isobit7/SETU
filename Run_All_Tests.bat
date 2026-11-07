@echo off
setlocal
cd /d "%~dp0"
title SETU Automated Test Execution & Report Generator

echo =====================================================================
echo                SETU ISL TRANSLATOR — AUTOMATED TEST RUNNER           
echo =====================================================================
echo.
echo Running all verification test suites and generating human-readable
echo dated execution report in tests\TEST_EXECUTION_LOG.md...
echo.

".venv\Scripts\python.exe" -m tests.runner

if %ERRORLEVEL% EQU 0 (
    echo.
    echo =====================================================================
    echo [SUCCESS] All test suites passed! Report written to:
    echo           tests\TEST_EXECUTION_LOG.md
    echo =====================================================================
) else (
    echo.
    echo =====================================================================
    echo [WARNING] Some tests encountered failures. Check report for details:
    echo           tests\TEST_EXECUTION_LOG.md
    echo =====================================================================
)

pause
