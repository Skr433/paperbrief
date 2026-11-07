@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "APP_VER=0.2.3"

rem ---- 1) locate offline deps (_pylibs with PyMuPDF) ----
set "PYTHONPATH="
if exist "%~dp0..\_pylibs\pymupdf" set "PYTHONPATH=%~dp0..\_pylibs"
if not defined PYTHONPATH if exist "%~dp0..\..\05_deps\_pylibs\pymupdf" set "PYTHONPATH=%~dp0..\..\05_deps\_pylibs"
if not defined PYTHONPATH if exist "%~dp0_pylibs\pymupdf" set "PYTHONPATH=%~dp0_pylibs"

rem ---- 2) locate official pdf test set ----
if not defined PAPERBRIEF_PDF_DIR if exist "%~dp0..\ref_kxyzs\pdf" set "PAPERBRIEF_PDF_DIR=%~dp0..\ref_kxyzs\pdf"
if not defined PAPERBRIEF_PDF_DIR if exist "%~dp0..\..\03_pdf" set "PAPERBRIEF_PDF_DIR=%~dp0..\..\03_pdf"
if not defined PAPERBRIEF_PDF_DIR if exist "%~dp0..\pdf" set "PAPERBRIEF_PDF_DIR=%~dp0..\pdf"

rem ---- 3) find python: bundled runtime -> python -> py ----
set "PY="
if exist "C:\Users\ASUS\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe" set "PY=C:\Users\ASUS\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
if not defined PY (
    where python >nul 2>nul
    if not errorlevel 1 set "PY=python"
)
if not defined PY (
    where py >nul 2>nul
    if not errorlevel 1 set "PY=py"
)

echo.
echo ============================================
echo   PaperBrief  v%APP_VER%   (2026-10-29)
echo ============================================
echo   PDF dir : %PAPERBRIEF_PDF_DIR%
echo   Deps    : %PYTHONPATH%
echo.

if not defined PY (
    echo [FAIL] Python not found. Install Python 3.9+ or edit PY in this file.
    echo.
    pause
    exit /b 1
)

set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
"%PY%" "%~dp0scripts\check_env.py"
if errorlevel 1 (
    echo.
    echo [FAIL] Environment check failed. See FAIL lines above.
    echo        Fix: pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

rem ---- 5) port: 8000, fallback 8010 / 8011 ----
if not defined PORT set "PORT=8000"
set "PORT_OK="
for %%P in (%PORT% 8010 8011) do (
    if not defined PORT_OK (
        netstat -ano ^| findstr /R /C:":%%P .*LISTENING" >nul 2>nul
        if errorlevel 1 set "PORT_OK=%%P"
    )
)
if not defined PORT_OK (
    echo [FAIL] Ports 8000/8010/8011 are busy. Close the program using them.
    pause
    exit /b 1
)
if not "%PORT_OK%"=="%PORT%" (
    echo [WARN] Port %PORT% busy, switching to %PORT_OK%
    set "PORT=%PORT_OK%"
)

echo.
echo   URL   : http://127.0.0.1:%PORT%
echo   Demo  : http://127.0.0.1:%PORT%/?auto=1706.03762v7.pdf^&brief=1
echo   Log   : data\server.log
echo   Close this window to stop the server.
echo ============================================
echo.

if not exist "%~dp0data" mkdir "%~dp0data"
start "" "http://127.0.0.1:%PORT%/"
"%PY%" "%~dp0backend\server.py" 2^>^>"%~dp0data\server.log"

echo.
echo Server stopped. See data\server.log
pause
