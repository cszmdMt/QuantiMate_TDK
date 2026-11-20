"""Frakcionális differenciálás (V7-3): memóriamegőrző stacionarizálás.

A jellemzők vagy szintek (nem stacionerek), vagy hozamok (stacionerek, de a
memóriát eldobják). A frakcionális differenciálás (López de Prado 2018, AFML 5.
fej.; Hosking 1981) a legkisebb olyan d rendet keresi, amelynél a sor már
stacioner (ADF), de a hosszú memória maximálisan megmarad — tipikusan d≈0,3–0,6
az árszintekre.

Fixed-width window (FFD): a súlysor egy tolerancia alatt csonkolódik, és
kizárólag MÚLTBELI értékekre alkalmazódik (a meglévő grep-teszt lefedi). A d*
megválasztása adatfüggő döntés → KIZÁRÓLAG a train-szakaszon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def ffd_weights(d: float, thresh: float = 1e-5, max_width: int = 10000) -> np.ndarray:
    """Fixed-width frakcionális differenciálás súlyai (a legújabb megfigyeléstől
    visszafelé): w_0=1, w_k = -w_{k-1}·(d-k+1)/k, |w_k|<thresh-nél csonkolva."""
    w = [1.0]
    k = 1
    while k < max_width:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])   # legrégebbi → legújabb súly


def frac_diff_ffd(series: pd.Series, d: float, thresh: float = 1e-5) -> pd.Series:
    """FFD-transzformált sor; a bemelegedési (ablak-hosszú) eleje NaN.
    Csak múltbeli értékekre néz (kauzális)."""
    w = ffd_weights(d, thresh)
    width = len(w)
    x = series.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    for i in range(width - 1, len(x)):
        out[i] = np.dot(w, x[i - width + 1: i + 1])
    return pd.Series(out, index=series.index, name=f"{series.name}_ffd")


def find_min_d(series: pd.Series, train_end_date, d_grid=None,
               thresh: float = 1e-5, adf_pvalue: float = 0.05) -> dict:
    """A legkisebb d, amelynél a train-szakasz FFD-sora stacioner (ADF p<küszöb).

    KIZÁRÓLAG a train-szakaszon dönt. Visszaad: {"d", "adf_pvalue",
    "corr_with_original", "grid"} — a corr a memória-megtartás mértéke.
    """
    if d_grid is None:
        d_grid = np.round(np.arange(0.0, 1.01, 0.1), 2)
    train = series.loc[:pd.Timestamp(train_end_date)]
    grid_rows = []
    chosen = None
    for d in d_grid:
        ffd = frac_diff_ffd(train, float(d), thresh).dropna()
        if len(ffd) < 50:
            continue
        pval = float(adfuller(ffd, autolag="AIC")[1])
        common = train.loc[ffd.index]
        corr = float(np.corrcoef(common, ffd)[0, 1]) if ffd.std() > 0 else np.nan
        grid_rows.append({"d": float(d), "adf_p": pval, "corr": corr})
        if chosen is None and pval < adf_pvalue:
            chosen = {"d": float(d), "adf_pvalue": pval,
                      "corr_with_original": corr}
    if chosen is None:                       # egyik d sem stacionárius → d=1
        chosen = {"d": 1.0, "adf_pvalue": grid_rows[-1]["adf_p"] if grid_rows else np.nan,
                  "corr_with_original": grid_rows[-1]["corr"] if grid_rows else np.nan}
    chosen["grid"] = grid_rows
    return chosen
