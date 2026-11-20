@echo off
REM ============================================================================
REM AI-Based Anti-Money Laundering (AML) Detection & Core Banking System
REM One-Click Virtual Environment (.venv) Setup & Dashboard Launcher
REM ============================================================================
title AML Detection & Core Banking Dashboard Launcher

cd /d "%~dp0"

echo ======================================================================
echo  AI-Based Anti-Money Laundering (AML) Detection ^& Banking System
echo ======================================================================

REM 1. Create a local virtual environment (.venv) if it does not exist yet
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating local Python virtual environment ^(.venv^)...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create .venv. Please ensure Python 3.10+ is installed and added to PATH.
        pause
        exit /b 1
    )
    echo [2/3] Installing required packages from requirements.txt...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
) else (
    echo [1/3] Existing virtual environment ^(.venv^) found.
    echo [2/3] Skipping installation ^(.venv is already set up^).
)

REM 2. Launch the Streamlit AML & Core Banking Dashboard
echo [3/3] Launching Streamlit Dashboard at http://localhost:8501 ...
".venv\Scripts\python.exe" -m streamlit run "dashboard\app.py"
pause
