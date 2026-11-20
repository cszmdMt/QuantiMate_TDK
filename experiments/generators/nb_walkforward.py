"""Shared generated notebook cells for the approved V8 / V8.5 experiment."""


def cells(version):
    return [
        ("md", f"""# QuantiMate {version} — ismételt előrejelzés, cél és horizont szerint

**Sávonként (cél × horizont) 30 dátumpár × 50 független seed; a két verzió sávonként 3000,
a hat sáv együtt 18 000 sikeres ismétlés.**
Ez a notebook a rögzített kísérleti protokoll (séma 3) végrehajtója. A futtató minden
feladathoz új Python-folyamatot és új kernelt indít. A generálás nem futtat cellákat.

- Befagyasztott, SHA-256-tal ellenőrzött napi adatok; nyers kezdőnap: 2001-02-01.
- Dátumpáronként a lezárás előtti 10 naptári év; indikátorbemelegítés korábbi adatokon.
- Időrendi 70/15/15; a határok előtt 21 nap purge + 5 nap embargo.
- **Hat sáv:** cél = ár vagy volatilitás; horizont = 1, 5 vagy 21 kereskedési nap.
  - **Ár:** korrigált záróár a lezárás utáni h. kereskedési napon; viszonyítási alap a naiv becslés
    (jövőbeli ár = mai záróár); veszteség: százalékos abszolút hiba. Minden ár-előrejelzéshez 90%-os és
    50%-os sáv tartozik: a tanítóminta kvantilisei, a lezáráskori volatilitással skálázva. Egynapos
    horizonton a sávot adaptív konformális kalibrálás igazítja (egy kimenetelt csak akkor használ fel,
    amikor az már ismert); 5 és 21 napon a kalibrálás a validáción rontotta a sáv hatékonyságát, ott a
    sáv kalibrálatlan marad.
  - **Volatilitás:** a következő h nap realizált varianciája (napi becslő: éjszakai hozam² +
    Rogers–Satchell); viszonyítási alap az EWMA-variancia, a tanítóablak átlagos arányával skálázva;
    veszteség: QLIKE. Jelöltek: log-HAR, LightGBM gamma-veszteséggel, és a kettő mértani közepe.
    Ha a validációs győztes a 2. lépcsőn nem ismétlődik meg, előre rögzített tartalékként a HAR kap
    egy próbát; a két lehetséges próba miatt a 2. lépcső szintje itt p < 0,05.
- **Túlillesztés elleni kiválasztás, minden sávon azonosan:** az alapértelmezés a viszonyítási alap.
  Jelöltek: ár-sávon 5–400, volatilitás-sávon 5–1600 fa, zsugorítás a viszonyítási alap felé
  (0,25 / 0,5 / 1), alap- és kis kapacitású
  konfiguráció. Modell csak akkor váltja le, ha (1) a Hansen-féle SPA-próba az ÖSSZES jelöltre együtt
  elveti, hogy egyik sem jobb nála (p < 0,10), és (2) a validációs győztes a tanítóablakon belül,
  független (purged, out-of-fold) mintán is veri (p < 0,10).
- V8.5: a hiperparaméter-keresés kizárólag a tanítóablakon belül, purged 4-fold mintán pontoz; a győztes
  csak PBO ≤ 0,5 mellett versenyezhet a validáción.
- Seedek: 1–50; LightGBM sor- és oszlopmintavétel: 80% / 80%, subsample_freq=1.
- Történelmi teszt: csak train-en illesztett modell. Külső előrejelzés: új modell a teljes ablak
  lezárásig megérett címkéin, újraillesztett HMM-mel, HAR-ral és viszonyítási szinttel.

A forrásnotebook önmagában nem indít tanítást: a futtató által megadott
QUANTIMATE_JOB szükséges. Használat: README.md.
"""),
        ("code", f'''import os
from pathlib import Path

NOTEBOOK_VERSION = {version!r}
JOB_PATH = os.environ.get("QUANTIMATE_JOB")
if JOB_PATH is None:
    print("Nincs kiserleti feladat. Tanitas nem indul. Hasznalat: python experiments/runners/run_experiments.py --help")
else:
    if not Path(JOB_PATH).is_file():
        raise FileNotFoundError("A megadott kiserleti feladat nem talalhato")
    print("Kiserleti feladat:", NOTEBOOK_VERSION, JOB_PATH)
''', ["parameters"]),
        ("md", """## Adat- és szivárgásvédelem

A notebook nem frissít adatforrást. Az input, a kód és a Python-környezet
ujjlenyomata a folytatáskor is ellenőrzött. A jellemzőképzés csak a lezárásig
szeletelt adatokon történik. A lezárás utáni valós érték (ár vagy realizált
variancia) kizárólag az elkészült előrejelzés értékeléséhez kerül elő.

A korábbi, utólagos árugrásból generált earnings-naptár és a prefixfüggő
hónapvége-jelző nem vesz részt ebben a kísérletben. A hitelesített earnings-adatból
csak a már megtörtént esemény óta eltelt idő használható. A rövid lefedettségű
opcionális adatforrások kimaradása a kiválasztás döntésnaplójába kerül; a
pillanatképből hiányzó forrás csak névvel vállalt kivételként fogadható el.
A jelenlegi adatrevíziók és a pooled-univerzum túlélési torzítása továbbra is korlát.
"""),
        ("code", '''if JOB_PATH is not None:
    # This experiment is offline. Accidental legacy refresh calls fail explicitly.
    import requests
    def _offline_request(*args, **kwargs):
        raise RuntimeError("Kiserlet kozben halozati adatfrissites tilos")
    requests.sessions.Session.request = _offline_request
    from quantimate.experiment import execute_job
    RESULT = execute_job(JOB_PATH, NOTEBOOK_VERSION)
    print("Sikeres feladat:", RESULT["job_key"])
    print("Sav:", RESULT["target"], "| horizont (kereskedesi nap):", RESULT["horizon"])
    print("Futasi ido (s):", round(RESULT["duration_seconds"], 2))
'''),
        ("md", """## Eredmények és folytatás

A sikeres eredmény atomikusan mentett JSON-fájl, önálló ellenőrzőösszeggel.
Újraindításkor csak a hiányzó feladatok indulnak. Egy hibás feladat legfeljebb
három próbát kap egy indítás során, azonos seeddel; utána a futtató továbblép.
A státusz csak mind a 18 000 eredménynél teljes; a status.json sávonként is számol.

A results.csv tartalmazza az összes külső előrejelzést, a futásonkénti történelmi
tesztmutatókat, az 50 seed hibáinak átlagát/szórását, valamint külön sorban az
átlagpredikció hibáját. A tracks.csv verziónként és sávonként egy összefoglaló sor.
A történelmi teszt minden napi predikciója és valós értéke külön,
verzió/sáv/dátumpár/seed szerint rendezett CSV-kben marad meg.
"""),
        ("code", '''if JOB_PATH is not None:
    if "forecast" in RESULT:
        import pandas as pd
        display(pd.DataFrame([RESULT["forecast"]]))
        display(pd.DataFrame([RESULT["test_summary"]]))
    else:
        import pandas as pd
        display(pd.DataFrame(RESULT["decision_log"]).drop(columns=["hp", "training_rows"]))
        GATE = RESULT["gate"]
        print("Veszteseg:", GATE["loss_metric"], "| viszonyitasi alap validacios vesztesege:",
              round(GATE["benchmark_validation_loss"], 4))
        print("Legjobb jelolt:", GATE["best_candidate"])
        print("1. lepcso - SPA p-ertek (", GATE["candidates"], "jelolt egyutt):", round(GATE["pvalue"], 4))
        if GATE["replication"] is not None:
            print("2. lepcso - megismetlodes a tanitoablakon belul: p =", round(GATE["replication"]["pvalue"], 4),
                  "| nyereseg (a veszteseg egysegeben):", round(GATE["replication"]["gain"], 4),
                  "| szint:", GATE["stage_two_alpha"])
        if GATE["fallback"] is not None:
            print("2. lepcso - HAR-tartalek: p =", round(GATE["fallback"]["replication"]["pvalue"], 4),
                  "| megismetlodott:", GATE["fallback"]["replication"]["replicated"])
        print("Kapu:", "ATMENT - modell valtja a viszonyitasi alapot" if GATE["passed"]
              else "NEM ment at - marad a viszonyitasi alap")
        if "hpo" in RESULT:
            print("V8.5 kereses (csak tanitoablak):", RESULT["hpo"]["reason"], "| PBO =", round(RESULT["hpo"]["pbo"], 3))
        print("Valasztott kar:", RESULT["selected"]["arm"], "| konfiguracio:", RESULT["selected"]["config"],
              "| fak:", RESULT["selected"]["iterations"], "| zsugoritas:", RESULT["selected"]["shrinkage"],
              "| kombinacio:", RESULT["selected"]["combine"])
        print("Validacios dontesek:", RESULT["decision_count"])
        for note in RESULT["notes"]:
            print(note)
'''),
    ]
