"""Earnings-naptár hitelesítés (V8-4): két független forrás keresztellenőrzése.

1. forrás (ELSŐDLEGES, hatósági): SEC EDGAR 8-K Item 2.02 filingek —
   a filing dátuma a bejelentés napja, az acceptanceDateTime (ET-re váltva)
   a napszakot adja (piaczárás után / előtt). Gépi, reprodukálható, PIT-tiszta.
2. forrás (keresztellenőrzés): Yahoo visualization API earnings-történet
   (cookie+crumb; a yfinance get_earnings_dates ugyanezt hívja).

Kimenet: data/nvda_earnings_verified.csv — kézi átnézés UTÁN
élesíthető. A pipeline-ba a
config.earnings_source="verified" kapcsolja be (default: "generated",
V6-ekvivalens — a golden-teszt így érintetlen).

PIT-megjegyzés: az ütemezett bejelentési dátum hetekkel előre nyilvános
(lag=-1 a registryben); az EPS-tényadat viszont csak a bejelentéskor ismert —
a CSV-ben riport célra szerepel, jellemzőként csak effective_date-től volna
használható.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

from quantimate.config import DATA_DIR

NVDA_CIK = "0001045810"   # EDGAR-ból visszaellenőrizve: NVIDIA CORP, ticker NVDA
EDGAR_BASE = "https://data.sec.gov/submissions/"
YAHOO_VIZ_URL = "https://query1.finance.yahoo.com/v1/finance/visualization"
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
VERIFIED_CSV = DATA_DIR / "nvda_earnings_verified.csv"


def _sec_headers() -> dict:
    """SEC-etikett: azonosítható User-Agent kontakt-emaillel (.env-ből)."""
    email = os.environ.get("SEC_CONTACT_EMAIL", "").strip()
    if not email:
        raise RuntimeError("SEC_CONTACT_EMAIL hiányzik a .env-ből — az EDGAR "
                           "azonosítható User-Agentet kér (kulcs nem kell).")
    return {"User-Agent": f"QuantiMate research {email}"}


def _cached_json(url: str, cache_file: Path, headers: dict) -> dict:
    """Aznapi cache-ből hálózat nélkül; friss letöltés as-of dátummal mentve."""
    if cache_file.exists():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        if payload.get("_asof") == dt.date.today().isoformat():
            return payload["data"]
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(
        {"_asof": dt.date.today().isoformat(), "_url": url, "data": data}),
        encoding="utf-8")
    time.sleep(0.3)   # SEC rate-limit etikett (max 10 req/s alatt bőven)
    return data


def classify_timing(acceptance_et: pd.Timestamp) -> str:
    """Bejelentés napszaka az ET acceptance-időből: post_close / pre_open / intraday."""
    t = acceptance_et.time()
    if t >= dt.time(16, 0):
        return "post_close"
    if t < dt.time(9, 30):
        return "pre_open"
    return "intraday"


def effective_date(announce_date: pd.Timestamp, timing: str) -> pd.Timestamp:
    """Az első kereskedési nap, amelynek ZÁRÓÁRÁBAN a bejelentés már benne van.
    post_close → következő munkanap; pre_open/intraday → aznap.
    (Munkanap-közelítés; ünnepnapnál a naptár-illesztés a features-oldalon
    searchsorted-del úgyis a következő kereskedési napra képez.)"""
    if timing == "post_close":
        return (announce_date + pd.offsets.BDay(1)).normalize()
    return announce_date.normalize()


def fetch_edgar_earnings(cache_dir: Path | None = None) -> pd.DataFrame:
    """Minden NVDA 8-K Item 2.02 filing az EDGAR submissions API-ról
    (recent + archív lapok). Oszlopok: announce_date, acceptance_et, timing,
    effective_date, accession."""
    cache_dir = Path(cache_dir) if cache_dir is not None else DATA_DIR / "edgar_cache"
    headers = _sec_headers()
    root = _cached_json(EDGAR_BASE + f"CIK{NVDA_CIK}.json",
                        cache_dir / f"CIK{NVDA_CIK}.json", headers)
    assert "NVDA" in root.get("tickers", []), "CIK-ellenőrzés: nem az NVDA-t kaptuk!"

    pages = [root["filings"]["recent"]]
    for f in root["filings"].get("files", []):
        page = _cached_json(EDGAR_BASE + f["name"], cache_dir / f["name"], headers)
        pages.append(page)

    rows = []
    for rec in pages:
        for i in range(len(rec["form"])):
            if rec["form"][i] == "8-K" and "2.02" in (rec["items"][i] or ""):
                acc_utc = pd.Timestamp(rec["acceptanceDateTime"][i])
                acc_et = acc_utc.tz_convert("America/New_York").tz_localize(None)
                announce = pd.Timestamp(rec["filingDate"][i])
                timing = classify_timing(acc_et)
                rows.append({
                    "announce_date": announce,
                    "acceptance_et": acc_et,
                    "timing": timing,
                    "effective_date": effective_date(announce, timing),
                    "accession": rec["accessionNumber"][i],
                })
    df = pd.DataFrame(rows).sort_values("announce_date").reset_index(drop=True)
    # Csak az azonos NAPI duplumot (módosított/ismételt filing) dobjuk — egy
    # negyedévben jogosan lehet több Item 2.02 (pl. előzetes figyelmeztetés
    # 2019-01-28, 2022-08-08 + a rendes jelentés).
    df = df[~df["announce_date"].duplicated(keep="first")]
    return df.reset_index(drop=True)


def fetch_yahoo_earnings(cache_file: Path | None = None,
                         page_size: int = 100) -> pd.DataFrame:
    """NVDA earnings-történet a Yahoo visualization API-ról (cookie+crumb).
    Oszlopok: yahoo_dt_et, yahoo_date, timing_type, eps_estimate, eps_actual."""
    cache_file = (Path(cache_file) if cache_file is not None
                  else DATA_DIR / "yahoo_earnings_NVDA.json")
    if cache_file.exists():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        if payload.get("_asof") == dt.date.today().isoformat():
            return pd.DataFrame(payload["data"]).assign(
                yahoo_dt_et=lambda d: pd.to_datetime(d["yahoo_dt_et"]),
                yahoo_date=lambda d: pd.to_datetime(d["yahoo_date"]))

    s = requests.Session()
    s.headers.update({"User-Agent": BROWSER_UA})
    s.get("https://fc.yahoo.com", timeout=20)                     # cookie
    crumb = s.get("https://query2.finance.yahoo.com/v1/test/getcrumb",
                  timeout=20).text.strip()

    rows, offset = [], 0
    while True:
        body = {"sortField": "startdatetime", "sortType": "DESC",
                "entityIdType": "earnings",
                "includeFields": ["ticker", "eventname", "startdatetime",
                                   "startdatetimetype", "epsestimate",
                                   "epsactual"],
                "query": {"operator": "eq", "operands": ["ticker", "NVDA"]},
                "offset": offset, "size": page_size}
        r = s.post(YAHOO_VIZ_URL, params={"crumb": crumb, "lang": "en-US",
                                          "region": "US"}, json=body, timeout=30)
        r.raise_for_status()
        doc = r.json()["finance"]["result"][0]["documents"][0]
        cols = [c["id"] for c in doc["columns"]]
        batch = [dict(zip(cols, row)) for row in doc["rows"]]
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
        time.sleep(1.0)   # udvarias lapozás

    out = []
    for r_ in rows:
        name = (r_.get("eventname") or "")
        if "Meeting" in name:           # közgyűlés nem earnings
            continue
        ts = pd.Timestamp(r_["startdatetime"])
        ts_et = (ts.tz_localize("UTC") if ts.tzinfo is None else ts
                 ).tz_convert("America/New_York").tz_localize(None)
        out.append({"yahoo_dt_et": ts_et, "yahoo_date": ts_et.normalize(),
                    "timing_type": r_.get("startdatetimetype"),
                    "eps_estimate": r_.get("epsestimate"),
                    "eps_actual": r_.get("epsactual")})
    df = (pd.DataFrame(out).drop_duplicates(subset="yahoo_date")
          .sort_values("yahoo_date").reset_index(drop=True))
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    ser = df.copy()
    ser["yahoo_dt_et"] = ser["yahoo_dt_et"].astype(str)
    ser["yahoo_date"] = ser["yahoo_date"].astype(str)
    cache_file.write_text(json.dumps(
        {"_asof": dt.date.today().isoformat(),
         "data": ser.to_dict(orient="records")}), encoding="utf-8")
    return df


def cross_check(edgar: pd.DataFrame, yahoo: pd.DataFrame,
                start: str = "2015-01-01",
                tolerance_days: int = 1) -> pd.DataFrame:
    """EDGAR-filingek párosítása Yahoo-eseményekkel ±tolerance napon belül.

    A keresztellenőrzés csak a két forrás ÁTFEDÉSI ablakában értelmezhető
    (in_overlap: announce_date ≤ a Yahoo-lefedettség vége) — azon túl az EDGAR
    hatósági forrásként önmagában áll (verified_by="edgar_only").
    is_preliminary: párosítatlan filing, amelyhez ±45 napon belül van párosított
    filing ugyanabból a negyedéves ciklusból → előzetes/figyelmeztető 8-K
    (pl. 2019-01-28 guidance-vágás), nem ütemezett jelentés."""
    ed = edgar[edgar["announce_date"] >= pd.Timestamp(start)].copy()
    ydates = yahoo["yahoo_date"].to_numpy()
    matched_dt, delta = [], []
    for d in ed["announce_date"]:
        if len(ydates) == 0:
            matched_dt.append(pd.NaT); delta.append(None); continue
        diffs = (ydates - d.to_datetime64()).astype("timedelta64[D]").astype(int)
        k = abs(diffs).argmin()
        if abs(diffs[k]) <= tolerance_days:
            matched_dt.append(pd.Timestamp(ydates[k])); delta.append(int(diffs[k]))
        else:
            matched_dt.append(pd.NaT); delta.append(None)
    ed["yahoo_date"] = matched_dt
    ed["yahoo_delta_days"] = delta
    ed["match"] = ed["yahoo_date"].notna()
    ed = ed.merge(yahoo[["yahoo_date", "eps_estimate", "eps_actual"]],
                  on="yahoo_date", how="left")

    yahoo_end = (yahoo["yahoo_date"].max() if len(yahoo) else pd.Timestamp(start))
    ed["in_overlap"] = ed["announce_date"] <= yahoo_end
    matched_dates = ed.loc[ed["match"], "announce_date"].to_numpy()
    ed["is_preliminary"] = [
        (not m) and io and len(md) > 0
        and (abs((md - d.to_datetime64()).astype("timedelta64[D]").astype(int))
             <= 45).any()
        for d, m, io, md in zip(ed["announce_date"], ed["match"],
                                ed["in_overlap"], [matched_dates] * len(ed))]
    ed["verified_by"] = _verified_by_labels(ed)
    return ed


def _verified_by_labels(ed: pd.DataFrame) -> list[str]:
    """verified_by címke soronként: both / edgar_only / preliminary / unmatched."""
    out = []
    for _, r in ed.iterrows():
        if r["match"]:
            out.append("both")
        elif r["is_preliminary"]:
            out.append("preliminary")
        elif not r["in_overlap"]:
            out.append("edgar_only")
        else:
            out.append("unmatched")
    return out


def build_verified_csv(out_path: Path | None = None,
                       start: str = "2015-01-01") -> tuple[pd.DataFrame, float]:
    """A teljes V8-4 folyamat: letöltés → keresztellenőrzés → CSV-írás.
    Visszaadja (tábla, match_arány) — az elfogadási küszöb: match ≥ 0.95, és a CSV-t
    élesítés előtt kézzel át kell nézni."""
    out_path = Path(out_path) if out_path is not None else VERIFIED_CSV
    edgar = fetch_edgar_earnings()
    yahoo = fetch_yahoo_earnings()
    tbl = cross_check(edgar, yahoo, start=start)
    # Küszöb-alap: az átfedési ablak ÜTEMEZETT jelentései (az előzetes 8-K nem
    # Yahoo-earnings-esemény, az átfedésen túl pedig nincs mivel egyezni).
    sched = tbl[tbl["in_overlap"] & ~tbl["is_preliminary"]]
    match_rate = float(sched["match"].mean()) if len(sched) else 0.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(out_path, index=False, encoding="utf-8")
    return tbl, match_rate


def load_verified_earnings(path: Path | None = None, kind: str = "announce",
                           include_preliminary: bool = False) -> pd.DatetimeIndex:
    """A hitelesített CSV betöltése a features-réteg számára.
    kind="announce" → bejelentési napok (days_to_next: előre ismert ütemezés);
    kind="effective" → az első beárazó kereskedési nap (post-earnings jellemzők).
    Az előzetes/figyelmeztető 8-K-k alapból kimaradnak (NEM ütemezett esemény,
    előre nem látható → days_to_next-be tenni look-ahead volna)."""
    path = Path(path) if path is not None else VERIFIED_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Nincs hitelesített earnings-CSV: {path} — futtasd a "
            f"scripts/verify_earnings.py-t és hagyd jóvá az eredményt.")
    df = pd.read_csv(path, parse_dates=["announce_date", "effective_date"])
    if not include_preliminary and "is_preliminary" in df.columns:
        df = df[~df["is_preliminary"]]
    col = "announce_date" if kind == "announce" else "effective_date"
    return pd.DatetimeIndex(sorted(df[col].dropna().unique()))
