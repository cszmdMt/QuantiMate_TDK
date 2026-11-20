@echo off
rem QuantiMate - 2. nap: a V8.5 hat savja, ugyanabba a hat mappaba, parhuzamos futtatokkal (max. 6).
rem A V8 es a V8.5 futasa kozott a kodhoz, a notebookokhoz es a pillanatkephez nem szabad nyulni.
rem Megszakitas utan ugyanezt a fajlt kell ujra inditani: a kesz feladatok megmaradnak.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Nincs .venv. Elobb futtasd: 1_setup.bat
    pause
    exit /b 1
)
set PYTHONIOENCODING=utf-8
rem Parhuzamos futtatok szama: a gep logikai processzorainak szama, legfeljebb 6 (minden futtato egy szalon szamol).
rem Kezzel felulirhato: 3_run_V8_5.bat --max-parallel 3
set PAR=6
if defined NUMBER_OF_PROCESSORS set /a PAR=%NUMBER_OF_PROCESSORS%
if %PAR% GTR 6 set PAR=6
if %PAR% LSS 1 set PAR=1
echo Logikai processzorok: %NUMBER_OF_PROCESSORS%  -  egyszerre futo notebookok: %PAR%
".venv\Scripts\python.exe" experiments\runners\run_parallel.py V8.5 --max-parallel %PAR% %*
echo.
pause
