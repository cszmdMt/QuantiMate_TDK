"""Célváltozó-képzés: több-horizontú, volatilitás-normált kumulatív log-hozam.

y_{t,h} = (Σ_{i=1..h} logret_{t+i}) / (σ_t · √h),   H = {1, 5, 21}

ahol σ_t a t-napi KAUZÁLIS EWMA-volatilitás. A σ_t-t eltároljuk, hogy a
predikció denormalizálható legyen: r̂_{t,h} = ŷ_{t,h} · σ_t · √h.

Ez a modul az EGYETLEN hely, ahol shift(-k) megengedett —
kizárólag célváltozó-képzésre.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SIGMA_FLOOR = 1e-8


def forward_cum_return(log_ret: pd.Series, h: int) -> pd.Series:
    """F_{t,h} = Σ_{i=1..h} logret_{t+i} — a t utáni h nap kumulatív log-hozama.

    rolling(h).sum() a t+h napon = Σ_{j=t+1..t+h} r_j, ezt shift(-h) hozza t-re.
    """
    return log_ret.rolling(h).sum().shift(-h)


def build_targets(log_ret: pd.Series, sigma: pd.Series,
                  horizons=(1, 5, 21)) -> pd.DataFrame:
    """Vol-normált targetek minden horizontra + irány-címke h=1-re.

    Oszlopok: y_h{h} (normált), fwd_ret_h{h} (nyers kumulatív log-hozam),
    direction (h=1 irány) — a sigma külön Series-ként adott, a hívó tárolja.
    """
    sig = sigma.clip(lower=SIGMA_FLOOR)
    out = pd.DataFrame(index=log_ret.index)
    for h in horizons:
        fwd = forward_cum_return(log_ret, h)
        out[f"fwd_ret_h{h}"] = fwd
        out[f"y_h{h}"] = fwd / (sig * np.sqrt(h))
    out["direction"] = (out[f"fwd_ret_h{horizons[0]}"] > 0).astype(int)
    return out


def denormalize(y_norm: np.ndarray, sigma_t: np.ndarray, h: int) -> np.ndarray:
    """ŷ (normált) → nyers kumulatív log-hozam: r̂ = ŷ · σ_t · √h."""
    return np.asarray(y_norm) * np.asarray(sigma_t) * np.sqrt(h)


def reconstruct_price_path(anchor_price: float, med_log_returns: np.ndarray) -> np.ndarray:
    """Ár-pálya a JÓSOLT (denormalizált, h=1) medián log-hozamokból:
    P_t = anchor · exp(cumsum(r̂)). A p·avg_win képlet minden formája tilos."""
    return anchor_price * np.exp(np.cumsum(np.asarray(med_log_returns)))


def forward_realized_variance(variance: pd.Series, h: int) -> pd.Series:
    """RV_{t,h} = Σ_{i=1..h} v_{t+i} — a t utáni h nap realizált varianciája.

    A volatilitás-cél nevezője a hívónál h·σ_t² (az origón ismert EWMA-variancia).
    Ugyanaz az igazítás, mint a forward_cum_return-nél; shift(-h) csak itt.
    """
    return variance.rolling(h).sum().shift(-h)
