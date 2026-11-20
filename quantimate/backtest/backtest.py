"""Backtest és kockázati metrikák + Merton-frakció.

A lényeg: a Target log-hozam, de pozíció-súlyozott tőkegörbéhez EGYSZERŰ
hozam kell: r_simple = expm1(r_log); equity = cumprod(1 + pos·r_simple −
cost·|Δpos|). A risk_report is egyszerű hozamon számol.
"""
from __future__ import annotations

import numpy as np


def run_backtest(positions: np.ndarray, r_log: np.ndarray,
                 cost: float = 0.0005):
    """Tőkegörbe pozíció-súlyozott EGYSZERŰ hozammal és forgalmi költséggel.

    positions[t]: a t-napi hozamot elnyerő tőkehányad (a döntés t−1 végén
    születik); belépéskor az első |Δpos| is költséget fizet.
    Visszaad: (strat_ret_simple, equity, turnover).
    """
    positions = np.asarray(positions, dtype=float)
    r_simple = np.expm1(np.asarray(r_log, dtype=float))
    turnover = np.abs(np.diff(np.concatenate([[0.0], positions])))
    strat_ret = positions * r_simple - cost * turnover
    equity = np.cumprod(1.0 + strat_ret)
    return strat_ret, equity, turnover


def risk_report(returns_simple: np.ndarray, label: str) -> dict:
    """Éves hozam, Sharpe, Sortino, Calmar, MaxDD, VaR/CVaR — egyszerű hozamon."""
    r = np.asarray(returns_simple, dtype=float)
    ann_ret = float((1 + r).prod() ** (252 / len(r)) - 1)
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
    downside = r[r < 0].std() if (r < 0).any() else 0.0
    sortino = float(r.mean() / downside * np.sqrt(252)) if downside > 0 else 0.0
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    mdd = float(((eq - peak) / peak).min())
    calmar = float(ann_ret / abs(mdd)) if mdd < 0 else float("inf")
    var95 = float(-np.percentile(r, 5))
    tail = r[r <= -var95]
    cvar95 = float(-tail.mean()) if len(tail) else 0.0
    return {"Stratégia": label, "Éves hozam": ann_ret, "Sharpe": sharpe,
            "Sortino": sortino, "Calmar": calmar, "MaxDD": mdd,
            "VaR(95%)": var95, "CVaR(95%)": cvar95}


def merton_fraction(mu: np.ndarray, sigma: np.ndarray, *, half: bool = True,
                    cap: float = 0.5, sigma_floor: float = 1e-4) -> np.ndarray:
    """Merton-frakció: f* = μ̂/σ̂² — folytonos hozamú pozíció konzisztens
    méretezése (a bináris Kelly-képlet helyett, A10d).

    Fél-Merton + felső plafon; σ̂ padlóval az alacsony-vol robbanás ellen
    (hibakatalógus 14.).
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), sigma_floor)
    f = mu / sigma ** 2
    if half:
        f = 0.5 * f
    return np.clip(f, 0.0, cap)


def sigma_from_quantiles(q05: np.ndarray, q95: np.ndarray) -> np.ndarray:
    """σ̂ a (q95−q05) sávszélességből: normál eloszlásnál q95−q05 = 2·1.645·σ."""
    return (np.asarray(q95, dtype=float) - np.asarray(q05, dtype=float)) / 3.29


# ── V7-13: kockázati réteg — vol-targeting + CVaR-méretezés + short-oldal ─────

def vol_target_overlay(positions: np.ndarray, sigma_hat: np.ndarray,
                       target_vol: float, *, cap: float = 1.0,
                       floor: float = 1e-4) -> np.ndarray:
    """Vol-targeting overlay (Moreira–Muir 2017): a pozíciót σ_cél/σ̂_t-vel
    skálázza — nyugodt piacon nagyobb, viharosban kisebb kitettség. A σ̂
    KAUZÁLIS (a kvantilis-fejből), a target_vol val-on rögzített hiperparaméter."""
    scale = np.clip(target_vol / np.maximum(np.asarray(sigma_hat), floor), 0, cap / floor)
    out = np.asarray(positions, dtype=float) * scale
    return np.clip(out, -cap, cap)


def cvar_from_quantiles(q05: np.ndarray, sigma_t: np.ndarray, h: int = 1,
                        alpha: float = 0.05) -> np.ndarray:
    """Napi CVaR-becslés a q05 kvantilisből (a jósolt eloszlás közvetlen döntési
    felhasználása). A q05 alatti farkat normál-közelítéssel integráljuk:
    CVaR_α ≈ −(denormalizált q05) · (φ(z_α)/α)/z_α korrekcióval; egyszerű,
    dokumentált feltevéssel a farok-várható értékre.
    """
    from scipy import stats
    z = stats.norm.ppf(alpha)
    # a denormalizált q05 (hozam-tér) = q05_norm · σ_t · √h
    var = -np.asarray(q05, dtype=float) * np.asarray(sigma_t) * np.sqrt(h)
    # ES/VaR arány normál eloszlásnál: (φ(z)/α)/(−z)
    es_over_var = (stats.norm.pdf(z) / alpha) / (-z)
    return var * es_over_var


def cvar_position_cap(positions: np.ndarray, cvar: np.ndarray,
                      cvar_budget: float, *, cap: float = 0.5) -> np.ndarray:
    """A pozíciót úgy vágja, hogy a napi |pozíció|·CVaR ne lépje át a keretet
    (Rockafellar–Uryasev 2000). cvar_budget val-on rögzített."""
    cvar = np.maximum(np.asarray(cvar, dtype=float), 1e-8)
    max_pos = cvar_budget / cvar
    pos = np.asarray(positions, dtype=float)
    return np.clip(np.sign(pos) * np.minimum(np.abs(pos), max_pos), -cap, cap)


# ── V8-11: bizonytalanság-alapú (konformális-sáv) pozícióméretezés ───────────

def ewma_smooth(x: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """Kauzális EWMA-simítás: s_t = λ·s_{t−1} + (1−λ)·x_t (s_0 = x_0).
    A sávszélesség zaja ellen KÖTELEZŐ simítási opció (V8-11)."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    if len(x) == 0:
        return out
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = lam * out[t - 1] + (1 - lam) * x[t]
    return out


def conformal_sizing(med_ret: np.ndarray, band_width: np.ndarray, *,
                     k: float = 1.0, cap: float = 0.5, lam: float = 0.94,
                     width_floor: float = 1e-4) -> np.ndarray:
    """Pozíció ∝ jel / konformálissáv-szélesség: ahol a kalibrált sáv széles
    (nagy bizonytalanság), ott kicsi a kitettség — a jósolt eloszlás TELJES
    alakja kerül döntési felhasználásra, nem csak a mediánja.

    med_ret: denormalizált medián-hozam (jel, long-only: negatív → 0);
    band_width: denormalizált (q95−q05) sávszélesség, EWMA-val simítva;
    k: skála-konstans — KIZÁRÓLAG train+val-on kalibrálható, lásd
    calibrate_sizing_k.
    """
    w = np.maximum(ewma_smooth(band_width, lam=lam), width_floor)
    raw = np.clip(np.asarray(med_ret, dtype=float), 0.0, None) / w
    return np.clip(k * raw, 0.0, cap)


def calibrate_sizing_k(med_ret_val: np.ndarray, band_width_val: np.ndarray,
                       ref_positions_val: np.ndarray, *, cap: float = 0.5,
                       lam: float = 0.94) -> float:
    """A k skála val-illesztése: az átlagos |pozíció| egyezzen a referencia-
    szabályéval (fél-Merton) — így az A/B kitettség-semleges, a KÜLÖNBSÉG a
    méretezés eloszlás-érzékenységéből jön. Val-only fit (walk-forward refit tétel)."""
    base = conformal_sizing(med_ret_val, band_width_val, k=1.0, cap=np.inf,
                            lam=lam)
    mean_base = float(np.mean(base))
    if mean_base <= 0:
        return 1.0
    return float(np.mean(np.abs(ref_positions_val)) / mean_base)


def two_sided_position(p_up: np.ndarray, med_ret: np.ndarray, sigma_hat: np.ndarray,
                       threshold: float, *, cap: float = 0.5, half: bool = True,
                       sigma_floor: float = 1e-4) -> np.ndarray:
    """Kétoldali pozíció: P(emelkedés)≥küszöb → long fél-Merton; P(esés)≥küszöb
    (azaz P(emelkedés)≤1−küszöb) → SHORT (negatív fél-Merton). Egyébként 0.

    Őszinte lábjegyzet (a notebookban): a short kölcsönzési költséget és az NVDA
    erős driftjét nem modellezzük külön — a short-oldal koncepció-demonstráció.
    """
    p_up = np.asarray(p_up, dtype=float)
    f = merton_fraction(np.abs(med_ret), sigma_hat, half=half, cap=cap,
                        sigma_floor=sigma_floor)
    pos = np.zeros_like(p_up)
    pos[p_up >= threshold] = f[p_up >= threshold]
    short_mask = p_up <= (1 - threshold)
    pos[short_mask] = -f[short_mask]
    return pos
