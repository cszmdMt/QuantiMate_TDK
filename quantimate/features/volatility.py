"""Range-alapú volatilitás (V7-1): Yang–Zhang becslő + HAR-RV előrejelző.

Yang–Zhang (2000, J. Business 73(3)): drift-független, nyitási ugrásra
robusztus, minimális varianciájú range-alapú σ-becslő:

    σ²_YZ = σ²_overnight + k·σ²_open-close + (1−k)·σ²_RS,
    k = 0.34 / (1.34 + (w+1)/(w−1)),
    RS_t = ln(H/O)·ln(H/C) + ln(L/O)·ln(L/C)   (Rogers–Satchell)

Kauzalitás: a t-napi becslés a t-napi O,H,L,C-t használja — a t-napi ZÁRÁSKOR
ez ismert, és a célváltozó t+1-től indul, tehát az igazítás azonos az
EWMA-változattal (a jelentés 1. ötletének szivárgás-elemzése szerint).

HAR-RV (Corsi 2009): log RV_{t+1} ~ β0 + βd·log RV_t + βw·log RV̄_5 + βm·log RV̄_21.
Az illesztés KIZÁRÓLAG a train-szakaszon; az előrejelzés kauzális.
A jövőre igazítás pozicionális szeleteléssel történik (negatív pandas-shift
nélkül) — az a szabály szerint a targets.py privilégiuma.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SIGMA_FLOOR = 1e-8


def yang_zhang_sigma(df: pd.DataFrame, ticker: str, window: int = 21) -> pd.Series:
    """Napi Yang–Zhang σ (nem évesített), gördülő `window` napos ablakon."""
    O = df[f"{ticker}_Open"]
    H = df[f"{ticker}_High"]
    L = df[f"{ticker}_Low"]
    C = df[f"{ticker}_Close"]

    r_overnight = np.log(O / C.shift(1))
    r_openclose = np.log(C / O)
    rs = (np.log(H / O) * np.log(H / C) + np.log(L / O) * np.log(L / C))

    var_o = r_overnight.rolling(window).var(ddof=1)
    var_c = r_openclose.rolling(window).var(ddof=1)
    var_rs = rs.rolling(window).mean()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    var_yz = var_o + k * var_c + (1 - k) * var_rs
    return np.sqrt(var_yz.clip(lower=SIGMA_FLOOR ** 2)).rename(f"{ticker}_yz_sigma")


def har_features(rv: np.ndarray) -> np.ndarray:
    """HAR jellemzők a t pozícióra: [1, log RV_t, log RV̄_{t-4..t}, log RV̄_{t-20..t}].
    Csak múltra néző gördülő átlagok; a sor eleje (t<20) NaN."""
    rv = np.asarray(rv, dtype=float)
    log_rv = np.log(np.clip(rv, SIGMA_FLOOR ** 2, None))
    s = pd.Series(log_rv)
    daily = s
    weekly = np.log(np.clip(pd.Series(rv).rolling(5).mean(), SIGMA_FLOOR ** 2, None))
    monthly = np.log(np.clip(pd.Series(rv).rolling(21).mean(), SIGMA_FLOOR ** 2, None))
    return np.column_stack([np.ones_like(log_rv), daily.to_numpy(),
                            weekly.to_numpy(), monthly.to_numpy()])


def har_rv_forecast(sigma: pd.Series, train_end_date) -> pd.Series:
    """Egy nappal előretekintő HAR-σ előrejelzés, train-only OLS-illesztéssel.

    Bemenet: napi σ-sor (pl. Yang–Zhang). Belül RV = σ² log-térben modellezve.
    A t-napi kimenet a t-ig ismert információból jelzi a t+1 napi σ-t — ez megy
    a target-normálásba σ̂_t-ként (sigma_mode="yz_har").

    Illesztés: X[t] → y[t+1] párok pozicionális szeleteléssel (X[:-1], y[1:]),
    kizárólag a train_end_date-ig eső CÉL-napokkal.
    """
    rv = (sigma.to_numpy(dtype=float)) ** 2
    X = har_features(rv)                       # (T, 4), eleje NaN
    log_rv = np.log(np.clip(rv, SIGMA_FLOOR ** 2, None))

    X_lag = X[:-1]                             # jellemzők t-ben...
    y_next = log_rv[1:]                        # ...cél: t+1 (pozicionális igazítás)
    target_dates = sigma.index[1:]

    valid = ~np.isnan(X_lag).any(axis=1)
    train_mask = valid & (target_dates <= pd.Timestamp(train_end_date))
    if train_mask.sum() < 60:
        raise ValueError("HAR: túl kevés train-minta az illesztéshez")

    beta, *_ = np.linalg.lstsq(X_lag[train_mask], y_next[train_mask], rcond=None)

    # Kauzális előrejelzés minden t-re: σ̂(t→t+1) = sqrt(exp(X[t]·β))
    pred_log_rv = X @ beta
    sig_hat = np.sqrt(np.exp(pred_log_rv))
    out = pd.Series(sig_hat, index=sigma.index, name=f"{sigma.name}_har")
    # A jellemző-bemelegedés (első ~20 nap) alatt az alap-σ-ra esünk vissza.
    warm = np.isnan(X).any(axis=1)
    out[warm] = sigma.to_numpy()[warm]
    return out.clip(lower=SIGMA_FLOOR)


def daily_range_variance(df: pd.DataFrame, ticker: str) -> pd.Series:
    """Egynapos variancia-becslő OHLC-ből: éjszakai hozam² + Rogers–Satchell.

    v_t = ln(O_t/C_{t-1})² + [ln(H_t/O_t)·ln(H_t/C_t) + ln(L_t/O_t)·ln(L_t/C_t)]

    A Rogers–Satchell-tag (1991) drift-független és nemnegatív; az éjszakai tag a
    nyitási ugrást adja hozzá. Ez a volatilitás-cél „megfigyelt" értéke: a napi
    záró-hozam négyzeténél sokkal kevésbé zajos helyettesítő (Patton 2011 a
    terjedelem-alapú helyettesítőt kifejezetten tárgyalja). A t-napi érték csak
    t-napi és t−1-napi árakat használ, tehát kauzális; az első sor NaN.
    """
    o, h, l, c = (df[f"{ticker}_{k}"] for k in ("Open", "High", "Low", "Close"))
    overnight = np.log(o / c.shift(1))
    rs = np.log(h / o) * np.log(h / c) + np.log(l / o) * np.log(l / c)
    return (overnight ** 2 + rs).clip(lower=SIGMA_FLOOR ** 2).rename(f"{ticker}_range_var")
