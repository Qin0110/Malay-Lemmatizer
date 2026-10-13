@echo off
REM Starts the app at http://localhost:8501. Runs SETUP.bat first if the
REM environment is missing. Close this window to stop.

setlocal EnableExtensions
cd /d "%~dp0"

set "ENVROOT=%LOCALAPPDATA%\RojakLemma"
set "VENV=%ENVROOT%\venv"
set "PY=%VENV%\Scripts\python.exe"
set "HF_HOME=%ENVROOT%\hf"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "PYTHONIOENCODING=utf-8"

if not exist "%PY%" (
    echo.
    echo  The app is not set up on this computer yet.
    echo  Running SETUP.bat first - this needs an internet connection.
    echo.
    pause
    call "%~dp0SETUP.bat"
    if not exist "%PY%" (
        echo.
        echo  Setup did not complete, so the app cannot start.
        echo.
        pause
        exit /b 1
    )
)

if not exist "%~dp0app.py" (
    echo.
    echo  [X] app.py is missing. Keep RUN.bat in the project folder,
    echo      next to app.py.
    echo.
    pause
    exit /b 1
)

echo.
echo ===================================================================
echo   Malay Rojak Lemmatiser
echo ===================================================================
echo   Starting... your browser will open at http://localhost:8501
echo   Leave this window open while you use the app.
echo   Press Ctrl+C or close this window to stop.
echo ===================================================================
echo.

"%PY%" -m streamlit run app.py
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo ===================================================================
    echo   The app stopped with an error ^(code %RC%^).
    echo.
    echo   Run SETUP.bat again - it re-checks every package and prints
    echo   what is wrong.
    echo ===================================================================
    echo.
    pause
)

exit /b %RC%
