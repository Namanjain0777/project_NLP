@echo off
REM ── Multilingual SMS Spam Classifier — Local startup script ─────────────────
REM Run this from the project root:
REM   cd "Spam SMS Classification"
REM   run.bat

SET PROJECT_DIR=%~dp0

echo.
echo  =========================================================
echo   Multilingual SMS Spam Classifier
echo   Starting FastAPI backend + Streamlit frontend
echo  =========================================================
echo.

REM Check Python is available
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python not found. Please install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

REM Check uvicorn is installed
python -c "import uvicorn" >nul 2>&1
IF ERRORLEVEL 1 (
    echo [WARN] uvicorn not found. Installing requirements...
    pip install -r requirements.txt
)

echo [1/2] Starting FastAPI backend on http://localhost:8000 ...
echo       API docs: http://localhost:8000/docs
start "SMS Spam API" cmd /k "cd /d "%PROJECT_DIR%" && python -m uvicorn src.api:app --reload --port 8000"

REM Small delay so API has time to start before browser/streamlit
timeout /t 3 /nobreak >nul

echo [2/2] Starting Streamlit frontend on http://localhost:8501 ...
start "SMS Spam UI" cmd /k "cd /d "%PROJECT_DIR%" && python -m streamlit run src/app.py"

echo.
echo  Both services are starting in separate windows.
echo  Press any key to exit this launcher (services keep running).
echo.
pause
