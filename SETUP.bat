@echo off
REM One-time setup: creates a virtual environment under %LOCALAPPDATA%\RojakLemma
REM and installs everything into it. Delete that folder to undo.
REM
REM   SETUP.bat         asks before installing the fine-tuned model support
REM   SETUP.bat gpu     installs it without asking
REM   SETUP.bat nogpu   skips it

setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

REM Kept outside the project on a short path: the torch wheels nest files
REM deep enough to hit the 260-character path limit otherwise.
set "ENVROOT=%LOCALAPPDATA%\RojakLemma"
set "RT=%ENVROOT%\rt"
set "VENV=%ENVROOT%\venv"
set "PY=%VENV%\Scripts\python.exe"
set "HF_HOME=%ENVROOT%\hf"
set "PROBE=%TEMP%\rojaklemma_probe.txt"
set "BASE_MODEL=unsloth/Llama-3.2-3B-Instruct-bnb-4bit"

set "PYURL=https://github.com/astral-sh/python-build-standalone/releases/download/20260901/cpython-3.12.14%%2B20260901-x86_64-pc-windows-msvc-install_only_stripped.tar.gz"

set "GPUMODE=ask"
if /i "%~1"=="gpu" set "GPUMODE=yes"
if /i "%~1"=="nogpu" set "GPUMODE=no"

echo.
echo ===================================================================
echo   Malay Rojak Lemmatiser - Setup
echo ===================================================================
echo   Project     : %CD%
echo   Environment : %VENV%
echo.

echo [1/6] Looking for Python 3.10-3.12...

REM A python.org install is preferred: its binaries are signed, so Windows
REM Smart App Control trusts them. The downloaded fallback is not signed.
set "BASEPY="
call :FINDPY py -3.12
if not defined BASEPY call :FINDPY py -3.11
if not defined BASEPY call :FINDPY py -3.10
if not defined BASEPY call :FINDPY python
if not defined BASEPY call :FINDPY "%RT%\python.exe"

if defined BASEPY (
    echo       Using: !BASEPY!
    goto HAVEPYTHON
)

echo       No suitable Python found - downloading a private one ^(~21 MB^).
echo       [!] It is unsigned, so Windows Smart App Control may block it.
echo           Installing Python 3.12 from python.org and re-running
echo           SETUP.bat avoids that.

where curl.exe >nul 2>&1
if errorlevel 1 goto NOTOOLS
where tar.exe >nul 2>&1
if errorlevel 1 goto NOTOOLS

if not exist "%ENVROOT%" mkdir "%ENVROOT%"
curl.exe -L --fail --retry 3 --retry-delay 2 -o "%ENVROOT%\python.tar.gz" "%PYURL%"
if errorlevel 1 (
    echo [X] Download failed. Check your internet connection and retry.
    goto FAILED
)

echo       Extracting...
tar.exe -xzf "%ENVROOT%\python.tar.gz" -C "%ENVROOT%"
if errorlevel 1 (
    echo [X] Could not extract the runtime.
    goto FAILED
)
if exist "%ENVROOT%\python\python.exe" (
    if exist "%RT%" rmdir /s /q "%RT%"
    move /y "%ENVROOT%\python" "%RT%" >nul
)
del /q "%ENVROOT%\python.tar.gz" >nul 2>&1

call :FINDPY "%RT%\python.exe"
if not defined BASEPY (
    echo [X] The downloaded runtime did not unpack as expected.
    goto FAILED
)
echo       Using: !BASEPY!

:HAVEPYTHON
echo.

REM Reuse an existing environment only if it was built on the same Python.
if exist "%PY%" (
    set "VBASE="
    set "NBASE="
    "%PY%" -c "import sys;sys.stdout.write(sys.base_prefix)" > "%PROBE%" 2>nul
    set /p VBASE=<"%PROBE%"
    "%BASEPY%" -c "import sys;sys.stdout.write(sys.prefix)" > "%PROBE%" 2>nul
    set /p NBASE=<"%PROBE%"
    if /i "!VBASE!"=="!NBASE!" (
        echo [2/6] Virtual environment already exists - reusing it.
        goto HAVEVENV
    )
    echo [2/6] Existing environment was built on a different Python - rebuilding.
    rmdir /s /q "%VENV%" 2>nul
    if exist "%VENV%" (
        echo [X] The old environment could not be removed. Close RUN.bat and
        echo     any window running the app, then run SETUP.bat again.
        goto FAILED
    )
)

echo [2/6] Creating the virtual environment...
"%BASEPY%" -m venv "%VENV%"
if errorlevel 1 (
    echo [X] Could not create the virtual environment at:
    echo     %VENV%
    goto FAILED
)

:HAVEVENV
if not exist "%PY%" (
    echo [X] The virtual environment has no python.exe at:
    echo     %PY%
    goto FAILED
)
"%PY%" -c "import _multiprocessing, ssl, ctypes" 2>nul
if errorlevel 1 (
    echo [X] The environment cannot load core Python modules.
    echo     This is usually Windows Smart App Control blocking an unsigned
    echo     Python. Install Python 3.12 from https://www.python.org/downloads/
    echo     then delete %ENVROOT% and run SETUP.bat again.
    goto FAILED
)
"%PY%" --version

echo.
echo [3/6] Preparing pip...
"%PY%" -m ensurepip --upgrade >nul 2>&1
"%PY%" -m pip install --quiet --disable-pip-version-check --no-warn-script-location --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [X] Could not prepare pip inside the virtual environment.
    goto FAILED
)

echo.
echo [4/6] Installing the app packages ^(Streamlit, pandas, ...^)
"%PY%" -m pip install --disable-pip-version-check --no-warn-script-location -r requirements-app.txt
if errorlevel 1 (
    echo.
    echo [X] Package installation failed, usually because the connection
    echo     dropped. Run SETUP.bat again - it picks up where it stopped.
    goto FAILED
)

echo.
echo [5/6] Fine-tuned model support ^(optional, NVIDIA GPU only^)
set "VERIFYARGS="

if /i "%GPUMODE%"=="no" (
    echo       Skipped.
    goto VERIFY
)
"%PY%" -c "import importlib.util as u,sys;s=u.find_spec('unsloth');sys.exit(0 if s and s.origin else 1)" >nul 2>&1
if not errorlevel 1 (
    echo       Already installed - making sure it is complete.
    goto GPUINSTALL
)

nvidia-smi -L >nul 2>&1
if errorlevel 1 (
    if /i "%GPUMODE%"=="yes" goto GPUINSTALL
    echo       No NVIDIA GPU found - skipping. The rule-based lemmatiser
    echo       works without it.
    goto VERIFY
)

set "GPUNAME="
set "DRIVER="
for /f "delims=" %%G in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do if not defined GPUNAME set "GPUNAME=%%G"
for /f "tokens=1 delims=." %%D in ('nvidia-smi --query-gpu^=driver_version --format^=csv^,noheader 2^>nul') do if not defined DRIVER set "DRIVER=%%D"
echo       Found: !GPUNAME! ^(driver !DRIVER!^)
if defined DRIVER if !DRIVER! LSS 528 (
    echo       [!] This driver is too old for CUDA 12.6. Update it from
    echo           https://www.nvidia.com/Download/index.aspx first, or the
    echo           model will not be able to use the GPU.
)

if /i "%GPUMODE%"=="yes" goto GPUINSTALL
echo.
echo       This adds PyTorch, Unsloth, the Llama 3.2 3B base model and
echo       the fine-tuned adapter. About 7 GB - allow 15-30 minutes.
choice /c YN /n /m "      Install it now? [Y/N] "
if errorlevel 2 (
    echo       Skipped. Run "SETUP.bat gpu" any time to add it.
    goto VERIFY
)

:GPUINSTALL
echo.
echo       Installing PyTorch ^(CUDA 12.6^), Unsloth and friends...
"%PY%" -m pip install --disable-pip-version-check --no-warn-script-location -r requirements-gpu.txt
if errorlevel 1 (
    echo.
    echo       [!] The GPU packages did not install. The app still works
    echo           with the rules only. Re-run "SETUP.bat gpu" to retry.
    goto VERIFY
)
echo.
echo       Downloading the base model ^(about 2.2 GB^)...
"%PY%" -c "from huggingface_hub import snapshot_download; snapshot_download('%BASE_MODEL%')"
if errorlevel 1 (
    echo       [!] Could not download it now - it will be fetched on first use.
)
echo.
echo       Downloading the fine-tuned model ^(about 100 MB^)...
"%PY%" download_model.py
if errorlevel 1 (
    echo       [!] Could not download it now. Run download_model.py later,
    echo           or train your own model - see README.md.
)
set "VERIFYARGS=--gpu"

:VERIFY
echo.
echo [6/6] Checking the installation...
"%PY%" verify_setup.py %VERIFYARGS%
if errorlevel 1 goto VERIFYFAILED

echo.
echo ===================================================================
echo   SETUP COMPLETE
echo.
echo   Now double-click RUN.bat to start the app.
echo ===================================================================
echo.
pause
exit /b 0


REM --- accept a Python only if it is 3.10-3.12 and actually runs -----------
:FINDPY
if defined BASEPY goto :eof
set "CAND="
%* -c "import sys;sys.stdout.write(sys.executable if (3,10)<=sys.version_info[:2]<(3,13) else '')" > "%PROBE%" 2>nul
if errorlevel 1 goto :eof
for %%S in ("%PROBE%") do if %%~zS EQU 0 goto :eof
set /p CAND=<"%PROBE%"
if not defined CAND goto :eof
set "BASEPY=%CAND%"
goto :eof


:NOTOOLS
echo [X] No Python was found, and curl.exe / tar.exe are not available to
echo     download one. Install Python 3.12 from
echo     https://www.python.org/downloads/ and run SETUP.bat again.
goto FAILED

:VERIFYFAILED
echo.
echo ===================================================================
echo   SETUP FINISHED, BUT THE CHECK ABOVE FOUND PROBLEMS
echo.
echo   The [FAIL] lines say what is wrong.
echo.
echo   "CUDA is not available" usually means the NVIDIA driver is too
echo   old - update it and run SETUP.bat again. The rule-based
echo   lemmatiser still works.
echo.
echo   "Application Control policy" means Windows Smart App Control
echo   blocked an unsigned file. Install Python 3.12 from python.org,
echo   delete %ENVROOT%
echo   and run SETUP.bat again.
echo ===================================================================
echo.
pause
exit /b 1

:FAILED
echo.
echo ===================================================================
echo   SETUP FAILED - see the message above.
echo ===================================================================
echo.
pause
exit /b 1
