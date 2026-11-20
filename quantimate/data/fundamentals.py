"""Fundamentum–tulajdonosi modul (V8-2) — szigorúan point-in-time jelcsaládok.

Három élő jelcsalád (a 13F intézményi koncentráció DOKUMENTÁLT SKIP: a SEC
negyedéves structured-data dumpjai ~10–15 GB / 45 negyedév — a felhasználó
döntése szerint kimarad; negyedéves jel 45+ napos laggal amúgy is ~40
megfigyelést adna):

1. Insider net-buy (Form 4, EDGAR): nyílt piaci vétel−eladás (P/S kódok),
   a FILING dátumára igazítva (nem a tranzakcióéra!), +1 nap elérhetőségi lag
   (az acceptance jellemzően zárás utáni) → rolling 21/63 napos nettó összeg
   darabban és dollárban.
2. Short interest (FINRA API, kétheti): settlement-dátum + 9 munkanapos
   publikációs lag — a t napon a legutóbb PUBLIKÁLT érték él (lépcső + ffill).
3. Napi short-volume arány (FINRA Reg SHO CNMS): shortVolume/totalVolume —
   a napi fájl ~18:00 ET-kor jelenik meg (zárás UTÁN) → shift(1): a t-napi
   jellemző a t−1 nap aránya. CDN-lefedettség: 2019-01-től (előtte 403).

A jellemzők az extras-úton utaznak (use_fundamentals flag, GBDT-A/B) — az
availability-maszkok (SI 2018+, Reg SHO 2019+) NaN-jai miatt, a V8-1 mintájára.
Narratív/spekulatív réteg ("kinek mi az érdeke") a backtestben TILOS — csak
ezek a számszerű proxyk élnek.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from quantimate.config import DATA_DIR
from quantimate.data import pit

NVDA_CIK = "0001045810"
SI_URL = ("https://api.finra.org/data/group/otcMarket/name/"
          "consolidatedShortInterest")
REGSHO_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d}.txt"
REGSHO_START = "2019-01-02"     # a CDN régebbi fájlokra 403-at ad (dokumentálva)
SI_PUB_LAG_BDAYS = 9            # FINRA disszeminációs ütem, konzervatívan

INSIDER_COLS = ("insider_netbuy_sh_21", "insider_netbuy_usd_21",
                "insider_netbuy_sh_63", "insider_netbuy_usd_63")
SI_COLS = ("short_interest_qty", "short_interest_dtc", "short_interest_chg")
SHORTVOL_COLS = ("shortvol_ratio", "shortvol_ratio_ma5")
FUNDAMENTAL_FEATURES = INSIDER_COLS + SI_COLS + SHORTVOL_COLS

# PIT-regisztráció (V8-10)
pit.register(r"insider_netbuy_(sh|usd)_(21|63)",
             source="EDGAR Form 4 (NVDA CIK 1045810)", lag_days=1,
             asof_rule="a FILING dátumára igazított tranzakció-összegek, +1 nap "
                       "shift (zárás utáni acceptance) → rolling 21/63 nap",
             notes="csak nyílt piaci P/S kódok; award/adó (M/F/G) kizárva")
pit.register(r"short_interest_(qty|dtc|chg)",
             source="FINRA consolidatedShortInterest API (kétheti)", lag_days=9,
             asof_rule="settlement + 9 munkanap publikációs lag; a t napon a "
                       "legutóbb publikált érték (lépcső + ffill)",
             notes="lefedettség 2017-12-től; revíziók a revisionFlag-ben")
pit.register(r"shortvol_ratio(_ma5)?",
             source="FINRA Reg SHO CNMS napi fájl", lag_days=1,
             asof_rule="a napi fájl zárás után (~18:00 ET) publikálódik → "
                       "shift(1): t-napi érték = t-1 nap aránya",
             notes="CDN-lefedettség 2019-01-től; hiányzó nap → NaN")


def _sec_headers() -> dict:
    email = os.environ.get("SEC_CONTACT_EMAIL", "").strip()
    if not email:
        raise RuntimeError("SEC_CONTACT_EMAIL hiányzik a .env-ből.")
    return {"User-Agent": f"QuantiMate research {email}"}


# ── 1) Form 4 insider ────────────────────────────────────────────────────────

def parse_form4_xml(xml_text: str) -> list[dict]:
    """Egy ownershipDocument XML nem-derivatív tranzakciói.
    Csak nyílt piaci kódok (P=vétel, S=eladás); A/D előjelezi a darabszámot."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    out = []
    for tx in root.iter("nonDerivativeTransaction"):
        code = tx.findtext("transactionCoding/transactionCode", default="")
        if code not in ("P", "S"):
            continue
        sh = tx.findtext("transactionAmounts/transactionShares/value")
        px = tx.findtext("transactionAmounts/transactionPricePerShare/value")
        ad = tx.findtext("transactionAmounts/transactionAcquiredDisposedCode/value",
                         default="")
        if sh is None:
            continue
        sign = 1.0 if ad == "A" else -1.0
        shares = sign * float(sh)
        price = float(px) if px not in (None, "") else np.nan
        out.append({"code": code, "shares": shares,
                    "value_usd": shares * price if np.isfinite(price) else np.nan})
    return out


def fetch_form4_transactions(cache_dir: Path | None = None,
                             start: str = "2015-01-01",
                             verbose: bool = True) -> pd.DataFrame:
    """Minden NVDA Form 4 (filing_date ≥ start) tranzakció-szintű táblája.
    Az XML-ek darabonként cache-elődnek (data/edgar_cache/form4/); az
    aggregált tábla naponta frissülő parquet."""
    cache_dir = Path(cache_dir) if cache_dir is not None else DATA_DIR / "edgar_cache"
    agg_file = cache_dir / "form4_nvda.parquet"
    if agg_file.exists():
        mtime = dt.date.fromtimestamp(agg_file.stat().st_mtime)
        if mtime == dt.date.today():
            return pd.read_parquet(agg_file)

    headers = _sec_headers()
    xml_dir = cache_dir / "form4"
    xml_dir.mkdir(parents=True, exist_ok=True)

    # A submissions-lapok a V8-4 earnings-modullal közös cache-ben élnek;
    # ha hiányoznak (tiszta gép), itt töltődnek le.
    if not (cache_dir / f"CIK{NVDA_CIK}.json").exists():
        from quantimate.data.earnings import EDGAR_BASE, _cached_json
        root = _cached_json(EDGAR_BASE + f"CIK{NVDA_CIK}.json",
                            cache_dir / f"CIK{NVDA_CIK}.json", headers)
        for f in root["filings"].get("files", []):
            _cached_json(EDGAR_BASE + f["name"], cache_dir / f["name"], headers)

    filings = []
    for f in sorted(cache_dir.glob(f"CIK{NVDA_CIK}*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))["data"]
        rec = data["filings"]["recent"] if "filings" in data else data
        for i in range(len(rec["form"])):
            if rec["form"][i] == "4" and rec["filingDate"][i] >= start:
                filings.append((rec["filingDate"][i],
                                rec["accessionNumber"][i],
                                rec["primaryDocument"][i]))
    filings = sorted(set(filings))
    if verbose:
        print(f"Form 4 filingek: {len(filings)} db ({start} ota)")

    rows = []
    for n, (fdate, acc, prim) in enumerate(filings):
        xml_file = xml_dir / f"{acc}.xml"
        if xml_file.exists():
            text = xml_file.read_text(encoding="utf-8")
        else:
            doc = prim.split("/")[-1]
            url = (f"https://www.sec.gov/Archives/edgar/data/{int(NVDA_CIK)}/"
                   f"{acc.replace('-', '')}/{doc}")
            r = requests.get(url, headers=headers, timeout=30)
            if not r.ok:
                warnings.warn(f"Form4 {acc}: HTTP {r.status_code} — kihagyva.")
                continue
            text = r.text
            xml_file.write_text(text, encoding="utf-8")
            time.sleep(0.12)   # SEC-etikett (<10 req/s)
        for tx in parse_form4_xml(text):
            tx["filing_date"] = pd.Timestamp(fdate)
            rows.append(tx)
        if verbose and (n + 1) % 200 == 0:
            print(f"  ... {n + 1}/{len(filings)} filing feldolgozva")

    tx = pd.DataFrame(rows)
    if len(tx) == 0:
        tx = pd.DataFrame(columns=["filing_date", "code", "shares", "value_usd"])
    tx.to_parquet(agg_file)
    return tx


def insider_features(daily_index: pd.DatetimeIndex,
                     tx: pd.DataFrame) -> pd.DataFrame:
    """Filing-napi nettó vétel → +1 nap lag → rolling 21/63 kereskedési nap."""
    out = pd.DataFrame(index=daily_index)
    if len(tx) == 0:
        for c in INSIDER_COLS:
            out[c] = np.nan
        return out
    daily = tx.groupby("filing_date").agg(sh=("shares", "sum"),
                                          usd=("value_usd", "sum"))
    # A filing-nap utáni ELSŐ kereskedési napon válik jellemzővé (lag=1):
    pos = np.searchsorted(daily_index.values, daily.index.values, side="right")
    sh = pd.Series(0.0, index=daily_index)
    usd = pd.Series(0.0, index=daily_index)
    for p, (s, u) in zip(pos, daily.itertuples(index=False)):
        if p < len(daily_index):
            sh.iloc[p] += s
            usd.iloc[p] += 0.0 if pd.isna(u) else u
    out["insider_netbuy_sh_21"] = sh.rolling(21, min_periods=1).sum()
    out["insider_netbuy_usd_21"] = usd.rolling(21, min_periods=1).sum()
    out["insider_netbuy_sh_63"] = sh.rolling(63, min_periods=1).sum()
    out["insider_netbuy_usd_63"] = usd.rolling(63, min_periods=1).sum()
    # Availability-mask: az első ELÉRHETŐVÉ vált filing (lag utáni pozíció)
    # előtt nincs megfigyelés → NaN, nem 0.
    landed = pos[pos < len(daily_index)]
    first_eff = int(landed.min()) if len(landed) else len(daily_index)
    out.iloc[:first_eff] = np.nan
    return out


# ── 2) FINRA short interest (kétheti) ───────────────────────────────────────

def fetch_short_interest(cache_dir: Path | None = None,
                         symbol: str = "NVDA") -> pd.DataFrame:
    """Teljes kétheti short-interest történet a FINRA API-ról (kulcs nélkül)."""
    cache_dir = Path(cache_dir) if cache_dir is not None else DATA_DIR
    cache_file = cache_dir / f"finra_si_{symbol}.parquet"
    if cache_file.exists():
        mtime = dt.date.fromtimestamp(cache_file.stat().st_mtime)
        if mtime == dt.date.today():
            return pd.read_parquet(cache_file)
    body = {"limit": 5000,
            "compareFilters": [{"compareType": "EQUAL",
                                "fieldName": "symbolCode",
                                "fieldValue": symbol}]}
    r = requests.post(SI_URL, json=body, headers={"Accept": "application/json"},
                      timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df["settlementDate"] = pd.to_datetime(df["settlementDate"])
    df = df.sort_values("settlementDate").reset_index(drop=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_file)
    return df


def short_interest_features(daily_index: pd.DatetimeIndex, si: pd.DataFrame,
                            pub_lag_bdays: int = SI_PUB_LAG_BDAYS) -> pd.DataFrame:
    """Lépcsős as-of illesztés: a t napon a legutóbb PUBLIKÁLT rekord él."""
    out = pd.DataFrame(index=daily_index,
                       columns=list(SI_COLS), dtype=float)
    if len(si) == 0:
        return out
    pub = si["settlementDate"] + pd.offsets.BDay(pub_lag_bdays)
    tbl = pd.DataFrame({
        "qty": si["currentShortPositionQuantity"].astype(float).values,
        "dtc": si["daysToCoverQuantity"].astype(float).values,
        "chg": si["changePercent"].astype(float).values,
    }, index=pd.DatetimeIndex(pub)).sort_index()
    tbl = tbl[~tbl.index.duplicated(keep="last")]
    idx = np.searchsorted(tbl.index.values, daily_index.values, side="right") - 1
    valid = idx >= 0
    out.loc[valid, "short_interest_qty"] = tbl["qty"].to_numpy()[idx[valid]]
    out.loc[valid, "short_interest_dtc"] = tbl["dtc"].to_numpy()[idx[valid]]
    out.loc[valid, "short_interest_chg"] = tbl["chg"].to_numpy()[idx[valid]]
    return out


# ── 3) FINRA Reg SHO napi short-volume ──────────────────────────────────────

def fetch_regsho_daily(cache_dir: Path | None = None, symbol: str = "NVDA",
                       start: str = REGSHO_START,
                       verbose: bool = True) -> pd.DataFrame:
    """Napi shortVolume/totalVolume a CNMS-fájlokból, inkrementális cache-sel.
    Csak a hiányzó napokat tölti le (a szimbólum sorát tartja meg)."""
    cache_dir = Path(cache_dir) if cache_dir is not None else DATA_DIR
    cache_file = cache_dir / f"regsho_{symbol}.parquet"
    have = (pd.read_parquet(cache_file) if cache_file.exists()
            else pd.DataFrame(columns=["date", "short_vol", "total_vol"]))
    have_dates = set(pd.to_datetime(have["date"]).dt.date) if len(have) else set()

    days = pd.bdate_range(start, dt.date.today() - dt.timedelta(days=1))
    todo = [d for d in days if d.date() not in have_dates]
    if verbose and todo:
        print(f"Reg SHO: {len(todo)} hianyzo nap letoltese ...")
    def _save(rows_):
        nonlocal have
        if rows_:
            have = (pd.concat([have, pd.DataFrame(rows_)], ignore_index=True)
                    .sort_values("date").reset_index(drop=True))
            cache_dir.mkdir(parents=True, exist_ok=True)
            have.to_parquet(cache_file)

    rows, misses = [], 0
    for n, d in enumerate(todo):
        try:
            r = requests.get(REGSHO_URL.format(d=d.strftime("%Y%m%d")),
                             timeout=20)
            if not r.ok:
                misses += 1           # ünnepnap / hiányzó fájl — nem hiba
                continue
            for ln in r.text.splitlines():
                parts = ln.split("|")
                if len(parts) >= 5 and parts[1] == symbol:
                    rows.append({"date": d, "short_vol": float(parts[2]),
                                 "total_vol": float(parts[4])})
                    break
        except requests.RequestException:
            misses += 1
        time.sleep(0.1)
        if (n + 1) % 250 == 0:
            _save(rows)               # checkpoint — megszakadó futás folytatható
            rows = []
            if verbose:
                print(f"  ... {n + 1}/{len(todo)} nap (checkpoint mentve)",
                      flush=True)
    _save(rows)
    if verbose:
        print(f"Reg SHO kesz: {len(have)} nap, kihagyott (unnep/403): {misses}")
    return have


def shortvol_features(daily_index: pd.DatetimeIndex,
                      sv: pd.DataFrame) -> pd.DataFrame:
    """shortVolume/totalVolume — shift(1) a zárás utáni publikáció miatt."""
    out = pd.DataFrame(index=daily_index)
    if len(sv) == 0:
        for c in SHORTVOL_COLS:
            out[c] = np.nan
        return out
    ratio = pd.Series(sv["short_vol"].values / sv["total_vol"].values,
                      index=pd.DatetimeIndex(pd.to_datetime(sv["date"])))
    ratio = ratio[~ratio.index.duplicated(keep="last")].sort_index()
    aligned = ratio.reindex(daily_index)
    out["shortvol_ratio"] = aligned.shift(1)
    out["shortvol_ratio_ma5"] = aligned.shift(1).rolling(5, min_periods=3).mean()
    return out


# ── összeállítás ────────────────────────────────────────────────────────────

def build_fundamental_features(daily_index: pd.DatetimeIndex,
                               use_insider: bool = True, use_si: bool = True,
                               use_shortvol: bool = True,
                               cache_dir: Path | None = None,
                               verbose: bool = True) -> pd.DataFrame:
    """A három jelcsalád a napi pipeline-indexre igazítva (availability-mask:
    a lefedettség előtti napok NaN-ok). Elérhetetlen forrás → őszinte skip."""
    frames = []
    if use_insider:
        try:
            tx = fetch_form4_transactions(cache_dir=cache_dir, verbose=verbose)
            frames.append(insider_features(daily_index, tx))
        except Exception as exc:
            warnings.warn(f"Insider (Form 4) skip: {exc}")
    if use_si:
        try:
            si = fetch_short_interest(cache_dir=cache_dir)
            frames.append(short_interest_features(daily_index, si))
        except Exception as exc:
            warnings.warn(f"Short interest skip: {exc}")
    if use_shortvol:
        try:
            sv = fetch_regsho_daily(cache_dir=cache_dir, verbose=verbose)
            frames.append(shortvol_features(daily_index, sv))
        except Exception as exc:
            warnings.warn(f"Reg SHO skip: {exc}")
    if not frames:
        return pd.DataFrame(index=daily_index)
    return pd.concat(frames, axis=1)
