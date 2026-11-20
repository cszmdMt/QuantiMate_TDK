"""Végrehajtási realizmus (V7-14): Corwin–Schultz effektív spread + részleges
pozíció-igazítás.

A fix 0,05%-os költség kényelmes, de nem adat-alapú. A **Corwin–Schultz** (2012)
becslő napi High–Low párokból ad effektív spread-idősort — a költség így időben
változó: turbulens napokon drágább kereskedni, ami a stratégiát pont a kritikus
rezsimben teszi realisztikusabbá. A forgalom-csökkentésre a **Gârleanu–Pedersen**
(2013) elv: "aim in front of the target, trade partially" — egyszerű, jól
tesztelhető változata a no-trade sáv (csak akkor kereskedünk, ha |cél−aktuális|>δ).

Kauzalitás: a CS-becslő a t napi H/L-t használ → a t napi ZÁRÁSKORI kereskedés
költségeként kauzális; a τ/δ val-on rögzített.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def corwin_schultz_spread(high: pd.Series, low: pd.Series,
                          window: int = 1) -> pd.Series:
    """Corwin–Schultz (2012) effektív relatív spread-becslés napi H/L-ből.

    β = E[(ln(H_t/L_t))² + (ln(H_{t+1}/L_{t+1}))²] a 2 napos ablakon,
    γ = (ln(H_{t,t+1}^max / L_{t,t+1}^min))²,
    α = (√(2β)−√β)/(3−2√2) − √(γ/(3−2√2)),
    S = 2(e^α−1)/(1+e^α).  A negatív becsléseket 0-ra vágjuk (a szerzők ajánlása).

    Megjegyzés: a t-edik érték a t és t−1 napok H/L-jét használja (kauzális:
    a t napi záráskor mindkettő ismert).
    """
    h = np.log(high.to_numpy(dtype=float))
    l = np.log(low.to_numpy(dtype=float))
    n = len(h)
    S = np.full(n, np.nan)
    const = 3 - 2 * np.sqrt(2)
    for t in range(1, n):
        hl_t = (h[t] - l[t]) ** 2
        hl_p = (h[t - 1] - l[t - 1]) ** 2
        beta = hl_t + hl_p
        h2 = max(h[t], h[t - 1])
        l2 = min(l[t], l[t - 1])
        gamma = (h2 - l2) ** 2
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / const - np.sqrt(gamma / const)
        s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        S[t] = max(s, 0.0)
    out = pd.Series(S, index=high.index, name="cs_spread")
    if window > 1:
        out = out.rolling(window, min_periods=1).mean()
    return out


def dynamic_cost(cs_spread: np.ndarray, fixed_floor: float = 0.0005) -> np.ndarray:
    """Idővariáns költség = max(fix padló, spread/2). A fél-spread a
    tranzakciónkénti effektív ár-hatás közelítése."""
    half_spread = np.nan_to_num(np.asarray(cs_spread, dtype=float), nan=0.0) / 2
    return np.maximum(fixed_floor, half_spread)


def partial_rebalance(target_pos: np.ndarray, tau: float = 0.5) -> np.ndarray:
    """Gârleanu–Pedersen részleges igazítás: az aktuális pozíció csak τ-nyit
    lép a cél felé (τ=1 → azonnali, τ→0 → lomha). Csökkenti a forgalmat."""
    target = np.asarray(target_pos, dtype=float)
    actual = np.empty_like(target)
    prev = 0.0
    for t in range(len(target)):
        prev = prev + tau * (target[t] - prev)
        actual[t] = prev
    return actual


def no_trade_band(target_pos: np.ndarray, delta: float = 0.05) -> np.ndarray:
    """No-trade sáv: csak akkor mozdulunk a célra, ha |cél − aktuális| > δ."""
    target = np.asarray(target_pos, dtype=float)
    actual = np.empty_like(target)
    prev = 0.0
    for t in range(len(target)):
        if abs(target[t] - prev) > delta:
            prev = target[t]
        actual[t] = prev
    return actual


def turnover(positions: np.ndarray) -> float:
    """Teljes forgalom = Σ|Δpozíció| (a break-even-költség és a forgalom-
    csökkentés mérőszáma)."""
    p = np.asarray(positions, dtype=float)
    return float(np.abs(np.diff(np.concatenate([[0.0], p]))).sum())
