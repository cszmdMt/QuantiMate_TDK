"""Kiértékelés-tudomány (A9, B4): Pesaran–Timmermann, Diebold–Mariano,
IC-guard, CRPS-közelítés, Wilson-CI, Deflated Sharpe Ratio."""
from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportion_confint


def pesaran_timmermann(y_true_dir: np.ndarray, y_pred_dir: np.ndarray) -> dict:
    """PT-teszt az iránypontosság szignifikanciájára (A9 képlet szerint).

    Degenerált (konstans) előrejelzésnél a nevező ≤ 0 → stat/p = NaN,
    'n/a' megjegyzéssel (hibakatalógus 11.).
    """
    y = np.asarray(y_true_dir, dtype=float)
    yh = np.asarray(y_pred_dir, dtype=float)
    n = len(y)
    p_hat = float((y == yh).mean())
    py, pyh = float(y.mean()), float(yh.mean())
    p_star = py * pyh + (1 - py) * (1 - pyh)
    v_hat = p_star * (1 - p_star) / n
    v_star = ((2 * py - 1) ** 2 * pyh * (1 - pyh)
              + (2 * pyh - 1) ** 2 * py * (1 - py)
              + 4 * py * pyh * (1 - py) * (1 - pyh) / n) / n
    denom = v_hat - v_star
    if denom <= 0 or pyh in (0.0, 1.0) or py in (0.0, 1.0):
        return {"stat": float("nan"), "pvalue": float("nan"),
                "acc": p_hat, "acc_expected": p_star, "note": "n/a (degenerált)"}
    stat = (p_hat - p_star) / np.sqrt(denom)
    pval = float(1 - stats.norm.cdf(stat))     # egyoldali: jobb-e a véletlennél
    return {"stat": float(stat), "pvalue": pval, "acc": p_hat,
            "acc_expected": p_star, "note": ""}


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray,
                    h: int = 1) -> dict:
    """DM-teszt két modell veszteség-sorára (d = loss_a − loss_b).

    Newey–West variancia (lag = h−1) + Harvey-féle kis mintás korrekció.
    Negatív stat → az 'a' modell vesztesége kisebb.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    d_bar = d.mean()
    gamma0 = np.mean((d - d_bar) ** 2)
    var = gamma0
    for lag in range(1, h):
        cov = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        var += 2 * (1 - lag / h) * cov
    if var <= 0:
        return {"stat": float("nan"), "pvalue": float("nan"), "note": "n/a"}
    dm = d_bar / np.sqrt(var / n)
    harvey = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm *= harvey
    pval = float(2 * (1 - stats.t.cdf(abs(dm), df=n - 1)))
    return {"stat": float(dm), "pvalue": pval, "note": ""}


def spearman_ic(pred: np.ndarray, realized: np.ndarray) -> dict:
    """Information Coefficient — konstans jóslat-vektornál NaN-t jelent
    explicit megjegyzéssel (hibakatalógus 17.)."""
    pred = np.asarray(pred, dtype=float)
    realized = np.asarray(realized, dtype=float)
    if len(pred) < 3 or pred.std() == 0 or realized.std() == 0:
        return {"ic": float("nan"), "pvalue": float("nan"),
                "note": "n/a (konstans vektor vagy N<3)"}
    ic, p = stats.spearmanr(pred, realized)
    return {"ic": float(ic), "pvalue": float(p), "note": ""}


def crps_from_quantiles(y_true: np.ndarray, q_pred: np.ndarray,
                        quantiles) -> float:
    """CRPS-közelítés a kvantilis-rácsból: CRPS ≈ 2·átlagolt pinball a
    kvantilisek felett (standard kvadratúra-közelítés)."""
    from quantimate.models.models import pinball_loss_np
    return 2.0 * pinball_loss_np(y_true, q_pred, quantiles=tuple(quantiles))


def wilson_ci(count: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson-féle CI a lefedettség-arányra (hibakatalógus 12.)."""
    lo, hi = proportion_confint(count, n, alpha=alpha, method="wilson")
    return float(lo), float(hi)


def deflated_sharpe(sr_observed: float, sr_trials: np.ndarray, n_obs: int,
                    skew: float, kurt_raw: float) -> dict:
    """Deflated Sharpe Ratio (Bailey–López de Prado).

    sr_observed és sr_trials NAPI (nem évesített) Sharpe-ok; n_obs a napok
    száma; kurt_raw a NEM-excess kurtózis. SR0 = a próbált konfigurációk
    maximumának várható értéke zérus valódi jel mellett; DSR = PSR(SR0).
    """
    sr_trials = np.asarray(sr_trials, dtype=float)
    n_trials = len(sr_trials)
    if n_trials < 2:
        return {"dsr": float("nan"), "sr0": float("nan"),
                "note": "n/a (kevés trial)"}
    v = float(np.var(sr_trials, ddof=1))
    gamma = 0.5772156649015329
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    sr0 = float(np.sqrt(v) * ((1 - gamma) * z1 + gamma * z2))
    denom = 1 - skew * sr_observed + (kurt_raw - 1) / 4 * sr_observed ** 2
    if denom <= 0:
        return {"dsr": float("nan"), "sr0": sr0, "note": "n/a (nevező ≤ 0)"}
    psr = stats.norm.cdf((sr_observed - sr0) * np.sqrt(n_obs - 1)
                         / np.sqrt(denom))
    return {"dsr": float(psr), "sr0": sr0, "note": ""}


def direction_metrics(y_true_dir, p_up, threshold: float = 0.5) -> dict:
    """Egységes irány-metrika csomag: Acc, F1, MCC (+degenerált jelzés), Brier."""
    from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                                 matthews_corrcoef)
    y = np.asarray(y_true_dir, dtype=int)
    p = np.asarray(p_up, dtype=float)
    pred = (p >= threshold).astype(int)
    degenerate = len(np.unique(pred)) == 1
    return {
        "Accuracy": float(accuracy_score(y, pred)),
        "F1": float(f1_score(y, pred)),
        "MCC": float(matthews_corrcoef(y, pred)),
        "Brier": float(brier_score_loss(y, p)),
        "degenerált": degenerate,
    }


def best_threshold_by_mcc(y_true_dir, p_up, grid=None) -> float:
    """A val-on MCC-t maximalizáló döntési küszöb (A9) — a teszten csak
    egyszer, változtatás nélkül alkalmazható."""
    from sklearn.metrics import matthews_corrcoef
    y = np.asarray(y_true_dir, dtype=int)
    p = np.asarray(p_up, dtype=float)
    if grid is None:
        grid = np.round(np.arange(0.40, 0.6001, 0.01), 2)
    best_t, best_m = 0.5, -np.inf
    for t in grid:
        pred = (p >= t).astype(int)
        if len(np.unique(pred)) == 1:
            continue
        m = matthews_corrcoef(y, pred)
        if m > best_m:
            best_t, best_m = float(t), float(m)
    return best_t
