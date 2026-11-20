# QuantiMate V8 és V8.5 – indítás

Előfeltétel: Windows, telepített Python 3.12. A fájlokat a repó gyökérmappájából kell indítani
(dupla kattintás vagy parancssor), ebben a sorrendben:

| Fájl | Mit csinál | Mikor |
|---|---|---|
| `1_setup.bat` | Létrehozza a `.venv` környezetet, telepíti a csomagokat, letölti a bemeneti adatokat és elkészíti a pillanatképet, lefuttatja a teszteket. | gépenként egyszer |
| `2_run_V8.bat` | A V8 notebookja hat sávon, párhuzamosan. | 1. nap |
| `3_run_V8_5.bat` | A V8.5 notebookja ugyanazon a hat sávon, párhuzamosan. | 2. nap, a V8 után |

- A két verzió külön fut, egyszerre mindig csak az egyik: egy indítás hat sávot futtat
  (ár és volatilitás × 1, 5, 21 napos horizont).
- Az indító magától igazodik a géphez: annyi notebook fut egyszerre, ahány logikai processzor van, de legfeljebb hat
  (pl. 4 szálas gépen 4; ha egy sáv végez, a következő automatikusan indul). Az eredmény ettől nem változik,
  mert minden notebook egy szálon számol.
- Megszakítás után ugyanazt a fájlt kell újra elindítani: a kész feladatok megmaradnak.
- A V8 és a V8.5 futása között a kódhoz, a notebookokhoz és a pillanatképhez nem szabad nyúlni (`git pull` sem),
  és mindkettőnek ugyanazon a gépen kell futnia.
- Kevés memória esetén kézzel kevesebb is megadható, parancssorból a repó gyökérmappájában:
  `2_run_V8.bat --max-parallel 3` (a V8.5-nél ugyanígy: `3_run_V8_5.bat --max-parallel 3`).

## Adatok – miért nincsenek a repóban?

A modell kizárólag valós, nyilvánosan letölthető adatból dolgozik, de ezek felhasználási feltételei
nem engedik az újraterjesztést. Ezért a repó egyetlen adatfájlt sem tartalmaz: az `1_setup.bat`
tölti le őket közvetlenül az eredeti forrásokból (`experiments/runners/fetch_data.py`):

| Forrás | Adat |
|---|---|
| Yahoo Finance | napi árfolyamok 2001-02-01-től (kb. 45 ticker), 60 perces NVDA-barok, earnings-naptár |
| SEC EDGAR | NVDA earnings-jelentések időpontjai (8-K), bennfentes tranzakciók (Form 4) |
| FINRA | short interest, Reg SHO napi short volume |
| FRED | EPU-trade index (EPUTRADE) |
| Caldara–Iacoviello | geopolitikai kockázati index (GPR) |

- A SEC egy kontakt e-mail-címet kér a lekérdezésekben (kulcs nem kell). Az `1_setup.bat` ezt egyszer
  megkérdezi, és a `.env`-be menti (a git nem követi).
- A letöltés nagyjából 10–20 perc. Ha megszakad, az `1_setup.bat` újraindítható; a meglévő pillanatképet
  (`data/test2/experiment_snapshot`) soha nem tölti le újra és nem írja felül.
- **Reprodukálhatóság:** a friss letöltés nem bájtra azonos a szakdolgozatban használt pillanatképpel:
  a Yahoo visszamenőleg korrigál, a 60 perces barokat pedig csak az utolsó ~730 napra adja. A futtató
  a pillanatkép ujjlenyomatát rögzíti, így a V8 és a V8.5 egy gépen garantáltan ugyanabból az adatból fut.
- A golden-teszt (`tests/test_golden.py`) fagyasztott adatot igényel, ami szintén nincs a repóban;
  nélküle a teszt kimarad.

## Eredmények – `results/`

| Mappa | Tartalom |
|---|---|
| `results/test2/` | A teljes V8 és V8.5 futás (séma 3, hat sáv, 18 000 ismétlés): sávonként a `parallel/<sáv>/results.csv` és az `experiment.json` (protokoll, pillanatkép- és kód-ujjlenyomat), az összefoglaló `V8_V85_osszehasonlitas.xlsx` és az ábrák (`abrak/`). |
| `results/test1/` | A korábbi, 1-es sémájú futás összesítője: `results.csv`, `main.xlsx`, `experiment.json`. |

A bemeneti pillanatkép (árfolyamadat) jogi okból nem része a repónak. A nyilvános kód a futtatottal
működésében azonos: csak kommentek, docstringek és egy figyelmeztető üzenet szövege tér el, ezért az
`experiment.json`-okban rögzített kód-ujjlenyomat a nyilvános fájlokra nem egyezik.

## Hol tart a futás? – `status.bat`

> **Futás közben is nyugodtan elindítható, nem zavarja a futtatást.** Nem tanít, nem ír fájlt, csak megszámolja a kész eredményeket.

- **Hol:** ugyanabban a repó-gyökérmappában, ahol a `2_run_V8.bat` is van.
- **Hogyan:** dupla kattintás a `status.bat`-ra. Új ablakban nyílik meg, és egy billentyű lenyomására bezárul.
  A futtató (`2_run_V8.bat` / `3_run_V8_5.bat`) ablakát közben **ne zárd be**, mert az leállítja a futást.
- **Mit mutat:** verziónként és sávonként, hogy a 30 dátumpárból hánynál kész a konfiguráció kiválasztása, és az
  1500 ismétlésből hány készült el. Ha van hibanapló, azt is jelzi.
- Ha ritkán hibával megáll (épp egy íródó fájlt olvasott), elég újra elindítani. A futást ez sem érinti.

## Licenc

Minden jog fenntartva, lásd: [LICENSE](LICENSE). A repó megtekintésre és hivatkozásra szolgál;
a tartalom felhasználásához a szerző előzetes írásos engedélye kell.
