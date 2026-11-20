@echo off
rem QuantiMate - osszefesules: a hat savmappa eredmenyet egy mappaba masolja (data\test2\repeated_experiment),
rem es ott felepiti az osszesito tablakat (results.csv, tracks.csv). Csak akkor indithato, ha egy futtato sem fut.
rem A V8 utan es a V8.5 utan is lefuttathato; a savmappakat nem modositja.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Nincs .venv. Elobb futtasd: 1_setup.bat
    pause
    exit /b 1
)
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" experiments\runners\run_parallel.py merge %*
echo.
pause
