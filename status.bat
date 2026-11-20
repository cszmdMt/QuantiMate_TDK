@echo off
rem QuantiMate - allapot: hany kivalasztas es ismetles keszult el savonkent. Nem tanit, futas kozben is indithato.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Nincs .venv. Elobb futtasd: 1_setup.bat
    pause
    exit /b 1
)
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" experiments\runners\run_parallel.py status %*
echo.
pause
