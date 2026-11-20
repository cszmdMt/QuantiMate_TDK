@echo off
rem QuantiMate - 0. lepes: Python-kornyezet, a bemeneti adatok letoltese es a tesztkeszlet.
rem Gepenkent egyszer kell. Python 3.12 szukseges (python.org, "py launcher" opcioval).
rem Az adatok jogi okbol nincsenek a repoban: ez a fajl tolti le oket az eredeti forrasokbol
rem (Yahoo, SEC EDGAR, FINRA, FRED, GPR), majd elkesziti a kiserlet pillanatkepet.
rem Ujrainditva a meglevo pillanatkepet nem tolti le ujra es nem irja felul.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
if not exist ".venv\Scripts\python.exe" (
    echo HIBA: a .venv nem jott letre. Telepitsd a Python 3.12-t, majd futtasd ujra ezt a fajlt.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo HIBA: a csomagok telepitese nem sikerult.
    pause
    exit /b 1
)
set PYTHONIOENCODING=utf-8

if exist "data\test2\experiment_snapshot\snapshot.json" goto tests
rem A SEC EDGAR azonosithato User-Agentet ker: kontakt e-mail cim kell (kulcs nem).
if not defined SEC_CONTACT_EMAIL if exist ".env" for /f "usebackq tokens=1,* delims==" %%a in (".env") do if /i "%%a"=="SEC_CONTACT_EMAIL" set "SEC_CONTACT_EMAIL=%%b"
if not defined SEC_CONTACT_EMAIL (
    echo.
    echo A SEC EDGAR letoltesehez egy kontakt e-mail cim kell. A SEC kereri a lekerdezesekben, mas celra nem kerul felhasznalasra.
    set /p "SEC_CONTACT_EMAIL=E-mail cim: "
)
if not defined SEC_CONTACT_EMAIL (
    echo HIBA: e-mail cim nelkul az EDGAR-adatok nem tolthetok le.
    pause
    exit /b 1
)
rem A cim a .env-be kerul (a git nem kovet), hogy ujrainditaskor ne kerdezze ujra.
findstr /b /i "SEC_CONTACT_EMAIL=" ".env" >nul 2>&1 || (echo(& echo SEC_CONTACT_EMAIL=%SEC_CONTACT_EMAIL%)>>".env"
echo.
echo Adatok letoltese - nagyjabol 10-20 perc, a halozattol fugg ...
".venv\Scripts\python.exe" experiments\runners\fetch_data.py
if errorlevel 1 (
    echo.
    echo HIBA: az adatok letoltese nem sikerult, reszletek fent. Ujrainditva a mar letoltott reszek megmaradnak.
    pause
    exit /b 1
)

:tests
".venv\Scripts\python.exe" -m pytest -o addopts="" -q -p no:cacheprovider
if errorlevel 1 (
    echo.
    echo A TESZTEK NEM ZOLDEK - a kiserletet igy ne inditsd el.
    pause
    exit /b 1
)
echo.
echo Kesz: a kornyezet es az adatok rendben, a tesztek zoldek. Kovetkezo lepes: 2_run_V8.bat
pause
