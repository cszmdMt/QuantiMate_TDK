"""Keresztmetszeti (pooled) tanítás (V8-6) — a statisztikailag helyes „több adat".

A V8 tanulsága: a mintaszám növelése NEM finomabb mintavétellel (intraday-tréning
tilos, Merton 1980), hanem KERESZTMETSZETTEL történhet: ~40 régóta jegyzett,
likvid név (megacapek + SMH-kör) napi soraiból pooled GBDT tanul, a kiértékelés
viszont VÁLTOZATLANUL kizárólag az NVDA-n, a baseline purged splitjein történik.

Kulcsdöntések (V8-6):
- Per-ticker vol-normált targetek (a targets.py képlete ticker-független) —
  a pooled tér így összemérhető.
- GENERIKUS jellemzőnevek: a cél-ticker oszlopai `tgt_` prefixet kapnak, a
  piaci kontextus (SPY/QQQ/SMH + VIX/VXN/TNX + naptár) minden sorban közös.
  Ticker-identitás: `tick_id` kategória-oszlop (a LightGBM splittelhet rajta).
- A pooled train-sorok dátuma a baseline NVDA-train ablakában marad — a
  purge/embargo védelem így a pooled sorokra is öröklődik (a train-ablak
  targetjei a val kezdete előtt realizálódnak).
- Az earnings- és peer-(AMD)-jellemzők a generikus készletből KIMARADNAK
  (40 tickerre nincs hitelesített naptár — dokumentált eltérés a baseline-tól;
  ezért fut a kontroll-kar: single-name GBDT ugyanezen a generikus készleten,
  ami a jellemzőkészlet-hatást izolálja a mintaszám-hatástól).

**Survivorship-caveat (kötelező kimondás):** a lista MA likvid, régóta jegyzett
nevekből áll — a 2016–2026 ablakban túlélő cégek felé torzít. A pooled modell
általánosítás-becslése ettől optimista lehet; az NVDA-ra szűkített kiértékelést
ez nem érinti (az NVDA nem a lista miatt van a mintában), a riport dokumentálja.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from quantimate.data.data import assemble_universe, fetch_ohlcv, _safe_name
from quantimate.features.features import (calendar_features, compute_index_features,
                       compute_indicators, overnight_intraday)
from quantimate.models.models_gbdt import DEFAULT_LAGS
from quantimate.features.targets import build_targets
from quantimate.data import pit

# Fix, determinisztikus univerzum (ábécé-sorrend; tick_id = pozíció).
# Mind 2015 előtt jegyzett, likvid név: SMH-kör (félvezető) + megacapek.
POOLED_UNIVERSE = (
    "AAPL", "ADBE", "ADI", "AMAT", "AMD", "AMZN", "ASML", "AVGO", "BAC",
    "CRM", "CSCO", "CVX", "GOOGL", "IBM", "INTC", "INTU", "JNJ", "JPM",
    "KLAC", "KO", "LRCX", "MCHP", "META", "MRVL", "MSFT", "MU", "NFLX",
    "NVDA", "NXPI", "ON", "ORCL", "PG", "QCOM", "SWKS", "TER", "TSLA",
    "TSM", "TXN", "WMT", "XOM",
)
CONTEXT_TICKERS = ("SPY", "QQQ", "SMH")
LIGHT_TICKERS = ("^VIX", "^VXN", "^TNX")

# PIT-regisztráció (V8-10): a tgt_* oszlopok ugyanazok a kauzális indikátorok,
# mint a NVDA_* család, generikus névvel; tick_id statikus identitás.
pit.register(r"tgt_(log_ret|ewma_vol|dist_sma20|dist_sma50|rsi14|macd_hist"
             r"|boll_pctB|boll_bw|atr_norm|adx14|di_spread|stoch_k|stoch_d"
             r"|roc10|obv_z|vol_z|r_overnight|r_intraday)(_lag\d+)?",
             source="yahoo_daily (pooled univerzum, V8-6)", lag_days=0,
             asof_rule="kauzális rolling/EWM a t-napi zárásig — a NVDA_* "
                       "családdal azonos képletek, generikus (tgt_) névvel")
pit.register(r"tick_id",
             source="statikus univerzum-lista (V8-6)", lag_days=-1,
             asof_rule="determinisztikus ticker-identitás — nem időfüggő")


def generic_feature_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Egy ticker GENERIKUS jellemző-mátrixa: tgt_* (teljes indikátor-készlet +
    overnight/intraday) + közös piaci kontextus + naptár. Earnings/peer nincs."""
    t = _safe_name(ticker)
    own = pd.concat([compute_indicators(df, t, full=True),
                     overnight_intraday(df, t)], axis=1)
    own.columns = [c.replace(f"{t}_", "tgt_", 1) for c in own.columns]
    frames = [own]
    for c in CONTEXT_TICKERS:
        frames.append(compute_indicators(df, _safe_name(c), full=False))
    for c in LIGHT_TICKERS:
        frames.append(compute_index_features(df, _safe_name(c)))
    frames.append(calendar_features(df.index))
    return pd.concat(frames, axis=1)


def generic_tabular(feats_g: pd.DataFrame, lags=DEFAULT_LAGS) -> pd.DataFrame:
    """A models_gbdt.tabular_view generikus párja: tgt_log_ret lagek."""
    tab = feats_g.copy()
    r = feats_g["tgt_log_ret"]
    for k in lags:
        tab[f"tgt_log_ret_lag{k}"] = r.shift(k)
    return tab


def load_ticker_frame(ticker: str, years: int = 10,
                      cache_dir: Path | None = None) -> pd.DataFrame:
    """Egy ticker + a közös kontextus dátum-illesztett OHLCV-kerete
    (a data.py cache-fegyelmével: aznapi parquet → hálózat nélkül)."""
    ohlcv = [fetch_ohlcv(ticker, years=years, cache_dir=cache_dir)]
    for c in CONTEXT_TICKERS:
        ohlcv.append(fetch_ohlcv(c, years=years, cache_dir=cache_dir))
    light = []
    for c in LIGHT_TICKERS:
        d = fetch_ohlcv(c, years=years, cache_dir=cache_dir)
        light.append(d[[f"{_safe_name(c)}_Close"]])
    return assemble_universe(ohlcv, light, (ticker,) + CONTEXT_TICKERS)


def ticker_design(df: pd.DataFrame, ticker: str, horizons,
                  lags=DEFAULT_LAGS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(tabuláris jellemzők, targetek) egy tickerre — dropna utáni közös indexen.
    A targetek a tgt_log_ret / tgt_ewma_vol párból épülnek (per-ticker
    vol-normálás, a targets.py képletével)."""
    feats_g = generic_feature_frame(df, ticker)
    tab = generic_tabular(feats_g, lags=lags).dropna()
    tg = build_targets(tab["tgt_log_ret"], tab["tgt_ewma_vol"],
                       horizons=horizons)
    return tab, tg


def build_pooled_training(train_dates: pd.DatetimeIndex, horizons,
                          tickers=POOLED_UNIVERSE, years: int = 10,
                          cache_dir: Path | None = None,
                          verbose: bool = True) -> dict:
    """A pooled train-mátrix: minden ticker train-ablakba eső, címkézett sora.

    train_dates: a baseline NVDA-train sorok DÁTUMAI (S.train → feats.index) —
    a purge/embargo védelem így öröklődik. Vissza: X (n, f), Y (n, |H|),
    feature_names, per-ticker sor-statisztika. Elérhetetlen ticker → őszinte
    skip figyelmeztetéssel (mock adat soha)."""
    date_set = pd.DatetimeIndex(train_dates)
    X_parts, Y_parts, rows_per_ticker = [], [], {}
    feature_names = None
    for tick_id, t in enumerate(tickers):        # ábécé-sorrend = determinizmus
        try:
            df_t = load_ticker_frame(t, years=years, cache_dir=cache_dir)
        except Exception as exc:
            warnings.warn(f"{t}: pooled skip ({exc})")
            rows_per_ticker[t] = 0
            continue
        tab, tg = ticker_design(df_t, t, horizons)
        y_cols = [f"y_h{h}" for h in horizons]
        ok = tab.index.isin(date_set) & tg[y_cols].notna().all(axis=1)
        tab_ok = tab[ok]
        if feature_names is None:
            feature_names = list(tab_ok.columns) + ["tick_id"]
        assert list(tab_ok.columns) + ["tick_id"] == feature_names, \
            f"{t}: eltérő generikus oszlopkészlet"
        X_parts.append(np.column_stack(
            [tab_ok.to_numpy(dtype=float),
             np.full(len(tab_ok), float(tick_id))]))
        Y_parts.append(tg.loc[ok, y_cols].to_numpy(dtype=float))
        rows_per_ticker[t] = int(ok.sum())
        if verbose:
            print(f"  OK {t}: {int(ok.sum())} train-sor")
    assert X_parts, "Egyetlen pooled ticker sem elérhető"
    return {"X": np.vstack(X_parts), "Y": np.vstack(Y_parts),
            "feature_names": feature_names,
            "rows_per_ticker": rows_per_ticker}


def nvda_generic_views(df: pd.DataFrame, feats_index: pd.DatetimeIndex,
                       nvda_tick_id: int = POOLED_UNIVERSE.index("NVDA"),
                       lags=DEFAULT_LAGS) -> np.ndarray:
    """Az NVDA generikus tabuláris nézete a BASELINE pipeline indexére igazítva
    (a baseline df-ből számolva — a kiértékelés így a megszokott S splitekkel
    címezhető). A tick_id oszlop az NVDA kódja."""
    feats_g = generic_feature_frame(df, "NVDA")
    tab = generic_tabular(feats_g, lags=lags).reindex(feats_index)
    assert not tab.isna().any().any(), \
        "NVDA generikus nézet: NaN a baseline jellemző-ablakban"
    return np.column_stack([tab.to_numpy(dtype=float),
                            np.full(len(tab), float(nvda_tick_id))])
