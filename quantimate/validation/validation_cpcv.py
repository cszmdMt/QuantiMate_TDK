"""Validáció-tudomány (V7-15): CPCV, PBO (CSCV) és Hansen-féle SPA-teszt.

- **CPCV** (López de Prado, AFML 12. fej.): N csoportból kombinatorikusan sok
  purged train/teszt-útvonal — a teljesítmény ELOSZLÁSA egyetlen szám helyett.
- **PBO** (Bailey et al. 2015, CSCV-logit): annak valószínűsége, hogy az
  IS-győztes OOS a medián alatt teljesít. Bemenete a konfigurációk NAPI
  stratégia-hozam mátrixa (nálunk: a REGISTRY val-időszaki hozamai — refit
  nélkül, a jelentés előírása szerint).
- **SPA** (Hansen 2005): „sok stratégiát próbáltunk" együttes teszt a
  benchmark ellen, stationary bootstrappel (Politis–Romano 1994).

Determinizmus: minden bootstrap/kombinatorika seedelt generátorral.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


# ── CPCV ─────────────────────────────────────────────────────────────────────

def cpcv_splits(n: int, n_groups: int = 8, k_test: int = 2,
                purge: int = 21, embargo: int = 5):
    """Kombinatorikus purged útvonal-generátor.

    A címkézett [0, n) tartomány n_groups összefüggő blokkra oszlik; minden
    k_test-elemű blokk-kombináció teszt, a train a maradék, a teszt-blokkok
    körüli purge+embargo sávok MINDKÉT oldalon kizárva (konzervatív AFML-purge:
    a train-címkék előre, a soros korreláció hátra sem érhet át).

    Yield: (train_idx, test_idx) párok — C(n_groups, k_test) útvonal.
    """
    bounds = np.linspace(0, n, n_groups + 1).astype(int)
    gap = purge + embargo
    for combo in combinations(range(n_groups), k_test):
        test_idx = np.concatenate([np.arange(bounds[g], bounds[g + 1])
                                   for g in combo])
        train_mask = np.ones(n, dtype=bool)
        for g in combo:
            lo = max(0, bounds[g] - gap)
            hi = min(n, bounds[g + 1] + gap)
            train_mask[lo:hi] = False
        yield np.where(train_mask)[0], test_idx


# ── PBO (CSCV) ───────────────────────────────────────────────────────────────

def pbo_cscv(returns_matrix: np.ndarray, n_blocks: int = 16) -> dict:
    """Probability of Backtest Overfitting a CSCV-logit módszerrel.

    returns_matrix: (T, N) — N konfiguráció napi hozama ugyanazon a T napon.
    A T napot n_blocks összefüggő blokkra osztjuk; minden felezős
    blokk-kombinációra: IS-győztes (Sharpe) → OOS relatív rang ω;
    PBO = P(ω < 0.5); λ = logit(ω) eloszlása is visszaadva.
    """
    R = np.asarray(returns_matrix, dtype=float)
    T, N = R.shape
    assert n_blocks % 2 == 0, "n_blocks legyen páros"
    bounds = np.linspace(0, T, n_blocks + 1).astype(int)
    blocks = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_blocks)]

    # Blokk-szintű elégséges statisztikák (vektorizált kombináció-értékelés)
    s1 = np.stack([R[b].sum(axis=0) for b in blocks])          # (B, N)
    s2 = np.stack([(R[b] ** 2).sum(axis=0) for b in blocks])   # (B, N)
    cnt = np.array([len(b) for b in blocks], dtype=float)

    def sharpe(idx):
        n_obs = cnt[idx].sum()
        mu = s1[idx].sum(axis=0) / n_obs
        var = s2[idx].sum(axis=0) / n_obs - mu ** 2
        sd = np.sqrt(np.maximum(var, 1e-18))
        return mu / sd

    omegas = []
    for is_combo in combinations(range(n_blocks), n_blocks // 2):
        is_idx = np.array(is_combo)
        oos_idx = np.array([b for b in range(n_blocks) if b not in is_combo])
        winner = int(np.argmax(sharpe(is_idx)))
        oos_sr = sharpe(oos_idx)
        rank = float((oos_sr < oos_sr[winner]).sum() + 1)      # 1..N
        omegas.append(rank / (N + 1))
    omegas = np.asarray(omegas)
    lam = np.log(omegas / (1 - omegas))
    return {"pbo": float((omegas < 0.5).mean()),
            "lambda_mean": float(lam.mean()),
            "n_combinations": len(omegas), "omegas": omegas}


# ── SPA (Hansen 2005) ────────────────────────────────────────────────────────

def stationary_bootstrap_indices(T: int, avg_block: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """Politis–Romano stationary bootstrap indexsor (geometriai blokkhossz)."""
    p = 1.0 / avg_block
    idx = np.empty(T, dtype=int)
    t = rng.integers(0, T)
    for i in range(T):
        idx[i] = t
        if rng.random() < p:
            t = rng.integers(0, T)
        else:
            t = (t + 1) % T
    return idx


def spa_test(strategy_returns: np.ndarray, benchmark_returns: np.ndarray,
             n_boot: int = 1000, avg_block: float = 10.0,
             seed: int = 42) -> dict:
    """Hansen-féle SPA (consistent p-érték) a stratégia-készlet vs. benchmark
    párra. H0: egyik stratégia sem jobb a benchmarknál.

    strategy_returns: (T, K); benchmark_returns: (T,).
    d_{k,t} = stratégia_k hozama − benchmark hozama.
    """
    S = np.asarray(strategy_returns, dtype=float)
    if S.ndim == 1:
        S = S[:, None]
    b = np.asarray(benchmark_returns, dtype=float)
    d = S - b[:, None]                                        # (T, K)
    T, K = d.shape
    d_bar = d.mean(axis=0)

    rng = np.random.default_rng(seed)
    boot_means = np.empty((n_boot, K))
    for i in range(n_boot):
        idx = stationary_bootstrap_indices(T, avg_block, rng)
        boot_means[i] = d[idx].mean(axis=0)

    omega = np.sqrt(np.maximum((boot_means - d_bar).var(axis=0, ddof=1) * 1.0,
                               1e-18)) * np.sqrt(T)           # ω̂_k ≈ sd(√T·d̄*)
    # Hansen (2005): T = max(max_k √T·d̄_k/ω̂_k, 0) — a nullánál vágás nélkül egy
    # a benchmarknál ROSSZABB készlet p-értéke ~0,5 lenne a helyes 1 helyett.
    t_stat = max(float(np.max(np.sqrt(T) * d_bar / omega)), 0.0)

    # Hansen-féle konzisztens újraközpontozás: a "nem reménytelenül rossz"
    # stratégiák saját átlagukra, a többiek 0-ra központozódnak.
    thresh = np.sqrt(omega ** 2 / T * 2 * np.log(np.log(max(T, 3))))
    keep = d_bar >= -thresh
    center = np.where(keep, d_bar, 0.0)

    t_boot = np.maximum(np.max(np.sqrt(T) * (boot_means - center[None, :]) / omega[None, :],
                               axis=1), 0.0)
    pval = float((t_boot >= t_stat).mean())
    return {"stat": t_stat, "pvalue": pval,
            "best_k": int(np.argmax(d_bar)), "d_bar": d_bar,
            "n_kept": int(keep.sum())}
