"""Collect-forward gyűjtések (V8.5 / V8-12): a MOST el nem tárolt adat örökre
elveszik — ez a modul napi gyűjtéssel építi a V9+ történeti adatvagyonát.

Két gyűjtés:

1. **60m bar-archívum**: a Yahoo 60m
   végpont gördülő ~730 napos ablakot ad — a `data/{T}_60m.parquet` cache ezért
   idővel ELVESZÍTI a legrégebbi barokat. Az archívum (`data/{T}_60m_archive.parquet`)
   append-only unió: minden futáskor beolvasztja a friss ablakot, a meglévő
   (ablakból már kicsúszott) barokat SOHA nem dobja el. Így az intraday réteg
   (V8-1) train-lefedettsége évről évre nő.

2. **Opciós-lánc snapshot** (V8-12): opciós jellemzők (ATM IV, ~25Δ skew,
   put/call OI) historikus adata ingyen nem létezik → mostantól gyűjtjük.
   Napi kivonat a Yahoo options API-ról `data/options_snapshots/` alá,
   hash-láncba fűzve (`data/ledger/collections.jsonl`, a live_ledger
   lánc-logikájával) — az utólagos módosítás így észlelhető.

**BACKTEST-FELHASZNÁLÁS TILOS** (V8-12): e gyűjtéseknek nincs
történetük a backtest-időszakra; jellemzőként legkorábban a V9-ben, a
felhalmozott ablakon versenyezhetnek. Szándékosan NINCS pit.register — ha egy
oszlop mégis a pipeline-ba kerülne, a PIT-lefedettség teszt bukik.

A snapshot-összegzőben a ~25Δ skew Black–Scholes deltával közelített
(r=0, osztalék nélkül — dokumentált egyszerűsítés; a nyers lánc eltárolódik,
így a pontosabb görögök utólag újraszámolhatók).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from quantimate.config import DATA_DIR
from quantimate.data.data import _safe_name
from quantimate.data.earnings import BROWSER_UA
from quantimate.backtest.live_ledger import append_prediction, verify_chain

OPTIONS_URL = "https://query2.finance.yahoo.com/v7/finance/options/{t}"
MAX_EXPIRY_DAYS = 100          # ~1-3 hónapos terminstruktúra — dokumentált kivonat
COLLECTIONS_LEDGER = DATA_DIR / "ledger" / "collections.jsonl"
SNAPSHOT_DIR = DATA_DIR / "options_snapshots"

SNAPSHOT_COLUMNS = ("expiry", "type", "strike", "last", "bid", "ask",
                    "volume", "open_interest", "implied_vol")


# ── 60m bar-archívum ─────────────────────────────────────────────────────────

def merge_bars(archive: pd.DataFrame | None, fresh: pd.DataFrame) -> pd.DataFrame:
    """Append-only unió: a friss ablak barjai felülírják az azonos időbélyegűt
    (keep-last: az utolsó, már lezárt bar frissülhet), a régiek megmaradnak."""
    frames = [f for f in (archive, fresh) if f is not None and len(f)]
    if not frames:
        return fresh
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def update_60m_archive(ticker: str = "NVDA",
                       cache_dir: Path | None = None) -> dict:
    """A friss 60m ablak beolvasztása az archívumba. Vissza: gyűjtés-összegző."""
    from quantimate.data.intraday import fetch_intraday_60m
    cache_dir = Path(cache_dir) if cache_dir is not None else DATA_DIR
    arch_file = cache_dir / f"{_safe_name(ticker)}_60m_archive.parquet"
    archive = pd.read_parquet(arch_file) if arch_file.exists() else None
    n_before = 0 if archive is None else len(archive)

    fresh = fetch_intraday_60m(ticker, cache_dir=cache_dir)
    merged = merge_bars(archive, fresh)
    merged.to_parquet(arch_file)
    return {"kind": "60m_archive", "ticker": ticker,
            "file": arch_file.name, "n_before": n_before,
            "n_added": len(merged) - n_before, "n_total": len(merged),
            "span": f"{merged.index.min()} -> {merged.index.max()}"}


# ── Opciós-lánc snapshot ─────────────────────────────────────────────────────

def _yahoo_session() -> tuple[requests.Session, str]:
    """Cookie+crumb session (az earnings.fetch_yahoo_earnings mintája)."""
    s = requests.Session()
    s.headers.update({"User-Agent": BROWSER_UA})
    s.get("https://fc.yahoo.com", timeout=20)
    crumb = s.get("https://query2.finance.yahoo.com/v1/test/getcrumb",
                  timeout=20).text.strip()
    return s, crumb


def _contracts_frame(opt_block: dict, expiry: pd.Timestamp) -> pd.DataFrame:
    """Egy lejárat calls+puts kontraktusai a snapshot-sémára hozva."""
    rows = []
    for typ, key in (("call", "calls"), ("put", "puts")):
        for c in opt_block.get(key, []):
            rows.append({
                "expiry": expiry, "type": typ,
                "strike": float(c.get("strike", np.nan)),
                "last": float(c.get("lastPrice", np.nan)),
                "bid": float(c.get("bid", np.nan)),
                "ask": float(c.get("ask", np.nan)),
                "volume": float(c.get("volume") or 0),
                "open_interest": float(c.get("openInterest") or 0),
                "implied_vol": float(c.get("impliedVolatility", np.nan)),
            })
    return pd.DataFrame(rows, columns=list(SNAPSHOT_COLUMNS))


def fetch_options_chain(ticker: str = "NVDA",
                        max_expiry_days: int = MAX_EXPIRY_DAYS
                        ) -> tuple[pd.DataFrame, float]:
    """A max_expiry_days-en belüli lejáratok teljes lánca + a spot ár."""
    s, crumb = _yahoo_session()
    base = OPTIONS_URL.format(t=ticker)
    r = s.get(base, params={"crumb": crumb}, timeout=30)
    r.raise_for_status()
    res = r.json()["optionChain"]["result"][0]
    spot = float(res["quote"]["regularMarketPrice"])
    horizon = dt.date.today() + dt.timedelta(days=max_expiry_days)

    frames = []
    for epoch in res["expirationDates"]:
        expiry = pd.Timestamp(epoch, unit="s")
        if expiry.date() > horizon:
            break
        rr = s.get(base, params={"crumb": crumb, "date": epoch}, timeout=30)
        rr.raise_for_status()
        blocks = rr.json()["optionChain"]["result"][0]["options"]
        if blocks:
            frames.append(_contracts_frame(blocks[0], expiry))
        time.sleep(0.3)   # udvarias lapozás
    chain = (pd.concat(frames, ignore_index=True) if frames
             else pd.DataFrame(columns=list(SNAPSHOT_COLUMNS)))
    return chain, spot


def validate_snapshot(chain: pd.DataFrame) -> None:
    """Snapshot-séma őr: oszlopkészlet + alap-értelmesség."""
    missing = [c for c in SNAPSHOT_COLUMNS if c not in chain.columns]
    assert not missing, f"Snapshot-séma sérült, hiányzó oszlopok: {missing}"
    if len(chain):
        assert chain["type"].isin(["call", "put"]).all(), "ismeretlen típus"
        assert (chain["strike"].dropna() > 0).all(), "nem-pozitív strike"
        assert (chain["open_interest"].dropna() >= 0).all(), "negatív OI"


def _bs_call_delta(spot: float, strike: np.ndarray, iv: np.ndarray,
                   t_years: np.ndarray) -> np.ndarray:
    """Black–Scholes call-delta (r=0, q=0 — dokumentált közelítés a ~25Δ
    strike-kiválasztáshoz; put-delta = call-delta − 1)."""
    from scipy import stats
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(spot / strike) + 0.5 * iv ** 2 * t_years) / (iv * np.sqrt(t_years))
    return stats.norm.cdf(d1)


def options_summary(chain: pd.DataFrame, spot: float,
                    asof: dt.date | None = None) -> dict:
    """Riport-összegző a láncból: ATM IV, ~25Δ skew, put/call OI-arány.

    A legalább 7 napos, legközelebbi lejáraton számol (a lejárat-közeli zaj
    ellen); üres/hiányos láncnál NaN-okat ad — soha nem becsül."""
    out = {"spot": float(spot), "atm_iv": np.nan, "skew_25d": np.nan,
           "putcall_oi": np.nan, "n_contracts": int(len(chain)),
           "n_expiries": int(chain["expiry"].nunique()) if len(chain) else 0}
    if not len(chain):
        return out
    asof = asof or dt.date.today()
    total_call_oi = chain.loc[chain["type"] == "call", "open_interest"].sum()
    if total_call_oi > 0:
        out["putcall_oi"] = float(
            chain.loc[chain["type"] == "put", "open_interest"].sum() / total_call_oi)

    days = (pd.DatetimeIndex(chain["expiry"]).normalize()
            - pd.Timestamp(asof)).days
    ok_exp = sorted(set(chain["expiry"][days >= 7]))
    if not ok_exp:
        return out
    near = chain[(chain["expiry"] == ok_exp[0])
                 & chain["implied_vol"].notna()
                 & (chain["implied_vol"] > 1e-4)].copy()
    if not len(near):
        return out
    t_years = max((pd.Timestamp(ok_exp[0]).date() - asof).days, 1) / 365.0

    atm = near.loc[(near["strike"] - spot).abs().sort_values().index[:4]]
    out["atm_iv"] = float(atm["implied_vol"].mean())

    call_d = _bs_call_delta(spot, near["strike"].to_numpy(),
                            near["implied_vol"].to_numpy(),
                            np.full(len(near), t_years))
    near["delta"] = np.where(near["type"] == "call", call_d, call_d - 1.0)
    calls = near[near["type"] == "call"]
    puts = near[near["type"] == "put"]
    if len(calls) and len(puts):
        c25 = calls.iloc[(calls["delta"] - 0.25).abs().argmin()]
        p25 = puts.iloc[(puts["delta"] + 0.25).abs().argmin()]
        out["skew_25d"] = float(p25["implied_vol"] - c25["implied_vol"])
    return out


def snapshot_options(ticker: str = "NVDA", out_dir: Path | None = None,
                     ledger_path: Path | None = None) -> dict:
    """Napi snapshot: lánc-letöltés → parquet → hash-láncolt napló-bejegyzés.
    Idempotens: aznapi meglévő snapshotot nem tölt le újra."""
    out_dir = Path(out_dir) if out_dir is not None else SNAPSHOT_DIR
    ledger_path = (Path(ledger_path) if ledger_path is not None
                   else COLLECTIONS_LEDGER)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = dt.date.today()
    snap_file = out_dir / f"{_safe_name(ticker)}_options_{today:%Y%m%d}.parquet"
    if snap_file.exists():
        return {"kind": "options_snapshot", "ticker": ticker,
                "file": snap_file.name, "skipped": "aznapi snapshot már létezik"}

    chain, spot = fetch_options_chain(ticker)
    validate_snapshot(chain)
    chain.to_parquet(snap_file)
    sha = hashlib.sha256(snap_file.read_bytes()).hexdigest()
    summary = options_summary(chain, spot, asof=today)

    rec = append_prediction(ledger_path, {
        "kind": "options_snapshot", "ticker": ticker,
        "asof": today.isoformat(), "file": snap_file.name, "sha256": sha,
        **{k: (None if isinstance(v, float) and not np.isfinite(v) else v)
           for k, v in summary.items()},
    })
    return {"kind": "options_snapshot", "ticker": ticker,
            "file": snap_file.name, "sha256": sha, "hash": rec["hash"],
            **summary}


def collect_daily(ticker: str = "NVDA", verbose: bool = True) -> list[dict]:
    """A napi collect-forward kör: 60m archívum +
    opciós snapshot. Hiba esetén őszinte skip — SOHA nem ír szintetikus adatot."""
    results = []
    for name, fn in (("60m archivum", update_60m_archive),
                     ("opcios snapshot", snapshot_options)):
        try:
            res = fn(ticker)
            results.append(res)
            if verbose:
                print(f"  OK {name}: " + ", ".join(
                    f"{k}={v}" for k, v in res.items()
                    if k in ("n_added", "n_total", "n_contracts", "n_expiries",
                             "atm_iv", "skipped", "file")))
        except Exception as exc:
            warnings.warn(f"{name} skip ({exc}) — szintetikus adat nincs.")
    chain_state = verify_chain(COLLECTIONS_LEDGER)
    if verbose and chain_state["n"]:
        print(f"  Gyujtesi lanc: {'OK' if chain_state['valid'] else 'SERULT!'} "
              f"({chain_state['n']} bejegyzes)")
    return results
