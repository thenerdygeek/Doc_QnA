@echo off
REM ── Doc QA Tool — Windows Launcher ─────────────────────────────
REM Double-click this file or run from Command Prompt.
REM First run: creates venv, installs dependencies, copies config.
REM Subsequent runs: just starts the server.
REM ────────────────────────────────────────────────────────────────

cd /d "%~dp0"

echo.
echo  ====================================
echo   Doc QA Tool
echo  ====================================
echo.

REM ── Check Python ──────────────────────────────────────────────
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo  [ERROR] Python not found on PATH.
    echo  Please install Python 3.11+ from https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Verify Python version >= 3.11
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PYMAJOR=%%a
    set PYMINOR=%%b
)
if %PYMAJOR% lss 3 goto :pyver_fail
if %PYMAJOR% equ 3 if %PYMINOR% lss 11 goto :pyver_fail
echo  Python %PYVER% OK
goto :pyver_ok

:pyver_fail
echo  [ERROR] Python 3.11+ required, found %PYVER%.
pause
exit /b 1

:pyver_ok

REM ── Create virtual environment (first run) ────────────────────
if exist ".venv\Scripts\activate.bat" goto :venv_ok
echo.
echo  Creating virtual environment...
python -m venv .venv
if %errorlevel% neq 0 (
    echo  [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)
echo  Virtual environment created.
:venv_ok

REM ── Activate venv ─────────────────────────────────────────────
call .venv\Scripts\activate.bat

REM ── Install / upgrade dependencies (first run or update) ──────
if exist ".venv\.installed" goto :deps_ok
echo.
echo  Installing dependencies (this may take a few minutes)...
pip install --quiet --upgrade pip
pip install -e .
if %errorlevel% neq 0 (
    echo  [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo installed> .venv\.installed
echo  Dependencies installed.
goto :deps_done
:deps_ok
echo  Dependencies already installed. Delete .venv\.installed to force reinstall.
:deps_done

REM ── Copy example config (first run) ───────────────────────────
if not exist "config.yaml" (
    if exist "config.example.yaml" (
        echo.
        echo  Creating config.yaml from example...
        copy config.example.yaml config.yaml >nul
        echo  Config created.
    )
)

REM ── Check Cody access token ───────────────────────────────────
if defined SRC_ACCESS_TOKEN (
    echo  Cody token: found via SRC_ACCESS_TOKEN env var
) else (
    echo.
    echo  [WARNING] SRC_ACCESS_TOKEN environment variable is not set.
    echo  Set it to your Sourcegraph access token:
    echo    set SRC_ACCESS_TOKEN=sgp_xxxxxxxxxxxx
    echo  Or set it permanently in System Environment Variables.
    echo.
)
if defined SRC_ENDPOINT (
    echo  Cody endpoint: %SRC_ENDPOINT%
) else (
    echo  Cody endpoint: https://sourcegraph.com
)

REM ── Start server ──────────────────────────────────────────────
echo.
echo  Starting Doc QA server...
echo  Open http://127.0.0.1:8000 in your browser.
echo  Press Ctrl+C to stop.
echo.

python -m doc_qa --log-level INFO serve

pause
