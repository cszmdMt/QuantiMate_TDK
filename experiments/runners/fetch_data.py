"""A kísérlet bemeneti adatainak letöltése az eredeti, nyilvános forrásokból, majd a pillanatkép elkészítése.

A repó jogi okból egyetlen letöltött adatfájlt sem tartalmaz (Yahoo, FINRA, SEC EDGAR, FRED, GPR):
mindenki maga tölti le őket ezzel a szkripttel (az `1_setup.bat` hívja).

Lépések:
  1. az opcionális források a helyi cache-be (alapból `data/`): SEC EDGAR earnings + Form 4,
     FINRA short interest + Reg SHO napi short volume, GPR (Caldara–Iacoviello), FRED EPUTRADE,
     Yahoo 60 perces NVDA-barok;
  2. a napi Yahoo-árfolyamok 2001-02-01-től és a pillanatkép (`run_experiments.py snapshot --download`).

Megjegyzés a reprodukálhatóságról: a friss letöltés nem bájtra azonos egy korábbi pillanatképpel
(a Yahoo visszamenőleg korrigál, a 60 perces barokat csak az utolsó ~730 napra adja). Egy gépen
a V8 és a V8.5 ugyanabból a pillanatképből fut; a meglévő pillanatképet a szkript soha nem írja felül.

A SEC azonosítható User-Agentet kér: a `SEC_CONTACT_EMAIL` a `.env`-ből vagy a környezetből jön.
"""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402
import requests  # noqa: E402

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
# A FRED a böngészőt utánzó User-Agentet nem szolgálja ki, egy egyszerű azonosítót igen.
FRED_HEADERS = {"User-Agent": "QuantiMate-data-fetch/1.0"}
EARNINGS_MIN_MATCH = 0.95


def say(text):
    print(text, flush=True)


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO / ".env", override=False)
    except ImportError:
        pass
    if not os.environ.get("SEC_CONTACT_EMAIL", "").strip():
        raise SystemExit("HIBA: hiányzik a SEC_CONTACT_EMAIL (a .env-ben vagy a környezetben). "
                         "A SEC EDGAR csak kontakt-e-mailt tartalmazó User-Agentet fogad el.")


def fetch_earnings(cache):
    from quantimate.data.earnings import cross_check, fetch_edgar_earnings, fetch_yahoo_earnings
    out = cache / "nvda_earnings_verified.csv"
    edgar = fetch_edgar_earnings(cache_dir=cache / "edgar_cache")
    yahoo = fetch_yahoo_earnings(cache_file=cache / "yahoo_earnings_NVDA.json")
    table = cross_check(edgar, yahoo, start="2015-01-01")
    scheduled = table[table["in_overlap"] & ~table["is_preliminary"]]
    rate = float(scheduled["match"].mean()) if len(scheduled) else 0.0
    if rate < EARNINGS_MIN_MATCH:
        raise SystemExit(f"HIBA: az EDGAR és a Yahoo earnings-dátumai csak {rate:.0%}-ban egyeznek "
                         f"(elvárt: legalább {EARNINGS_MIN_MATCH:.0%}). A CSV nem készült el.")
    table.to_csv(out, index=False, encoding="utf-8")
    return f"{len(table)} jelentés, egyezés {rate:.0%}"


def fetch_form4(cache):
    from quantimate.data.fundamentals import fetch_form4_transactions
    return f"{len(fetch_form4_transactions(cache_dir=cache / 'edgar_cache'))} tranzakció"


def fetch_short_interest(cache):
    from quantimate.data.fundamentals import fetch_short_interest
    return f"{len(fetch_short_interest(cache_dir=cache))} sor"


def fetch_regsho(cache):
    from quantimate.data.fundamentals import fetch_regsho_daily
    return f"{len(fetch_regsho_daily(cache_dir=cache))} nap"


def fetch_gpr(cache):
    from quantimate.features.geo import fetch_gpr_daily
    return f"{len(fetch_gpr_daily(cache_dir=cache))} nap"


def fetch_eputrade(cache):
    response = requests.get(FRED_URL.format(series="EPUTRADE"), headers=FRED_HEADERS, timeout=60)
    response.raise_for_status()
    frame = pd.read_csv(io.StringIO(response.text))
    frame.columns = ["date", "value"]
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    series = frame.dropna().set_index("date")["value"].rename("EPUTRADE")
    if series.empty:
        raise ValueError("FRED EPUTRADE: üres válasz")
    series.to_frame().to_parquet(cache / "fred_EPUTRADE.parquet")
    return f"{len(series)} hónap"


def fetch_intraday(cache):
    from quantimate.data.intraday import fetch_intraday_60m
    bars = fetch_intraday_60m("NVDA", cache_dir=cache)
    return f"{len(bars)} bar ({bars.index.min().date()} - {bars.index.max().date()})"


STEPS = (("SEC EDGAR + Yahoo earnings-naptár", fetch_earnings),
         ("SEC EDGAR Form 4 (bennfentes tranzakciók)", fetch_form4),
         ("FINRA short interest", fetch_short_interest),
         ("FINRA Reg SHO napi short volume (több perc)", fetch_regsho),
         ("GPR napi index", fetch_gpr),
         ("FRED EPUTRADE", fetch_eputrade),
         ("Yahoo 60 perces NVDA-barok", fetch_intraday))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bemeneti adatok letöltése és a pillanatkép elkészítése.")
    parser.add_argument("--cache", type=Path, default=REPO / "data", help="Helyi adat-cache (alap: data/)")
    parser.add_argument("--destination", type=Path, default=REPO / "data" / "test2" / "experiment_snapshot",
                        help="A pillanatkép helye (alap: data/test2/experiment_snapshot)")
    args = parser.parse_args(argv)

    if (args.destination / "snapshot.json").exists():
        say("A pillanatkép már megvan, nem töltök le újra (a meglévőt soha nem írom felül).")
        return 0
    load_env()
    args.cache.mkdir(parents=True, exist_ok=True)
    for number, (title, step) in enumerate(STEPS, 1):
        say(f"[{number}/{len(STEPS) + 1}] {title} ...")
        say("      kész: " + step(args.cache))

    from quantimate.experiment_data import create_snapshot
    say(f"[{len(STEPS) + 1}/{len(STEPS) + 1}] Napi Yahoo-árfolyamok 2001-től és a pillanatkép ...")
    manifest = create_snapshot(args.destination, args.cache, download=True)
    say(f"      kész: {len(manifest['files'])} fájl, azonosító {manifest['snapshot_id'][:16]}")
    for name in manifest["missing_optional"]:
        say("FIGYELEM: hiányzó adatréteg: " + name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
