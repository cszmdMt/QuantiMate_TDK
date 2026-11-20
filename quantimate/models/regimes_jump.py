"""Statisztikai jump-modell (V7-8, Nystrup–Lindström–Madsen 2020).

A Gauss-HMM likelihood-alapú illesztése érzékeny az inicializálásra és vastag
szélű adaton félrespecifikált. A jump-modell a rezsim-becslést **klaszterezés +
állapotváltási büntetés** feladatként oldja meg:

    min_{s,μ}  Σ_t ||x_t − μ_{s_t}||²  +  λ Σ_t 1[s_t ≠ s_{t-1}]

robusztusabb, perzisztensebb rezsimeket ad. Az illesztés (μ és a train-hozzárendelés)
KIZÁRÓLAG train-en; a jellemzőként használt állapot **online (kauzális)
dekódolással** áll elő — a t-edik állapot csak [0..t]-ből.

**Szivárgás:** ugyanaz a csapda, mint a V5 simított HMM-poszteriora — az
OFFLINE (teljes soros) dekódolás a jövőbe lát. Ezért a jellemző a FORWARD
(kauzális) dekódolásból jön; erre kötelező prefix-invariancia pytest.
"""
from __future__ import annotations

import numpy as np


def _assign_states_dp(X, mu, lam):
    """Offline DP (Viterbi-szerű) állapot-hozzárendelés — CSAK a train-illesztéshez
    (a fit belső lépése). A jellemző NEM ezt használja."""
    T, K = len(X), len(mu)
    emit = np.sum((X[:, None, :] - mu[None, :, :]) ** 2, axis=2)   # (T, K)
    cost = np.empty((T, K))
    back = np.zeros((T, K), dtype=int)
    cost[0] = emit[0]
    for t in range(1, T):
        for k in range(K):
            trans = cost[t - 1] + lam * (np.arange(K) != k)
            j = int(np.argmin(trans))
            back[t, k] = j
            cost[t, k] = emit[t, k] + trans[j]
    states = np.empty(T, dtype=int)
    states[-1] = int(np.argmin(cost[-1]))
    for t in range(T - 2, -1, -1):
        states[t] = back[t + 1, states[t + 1]]
    return states


class JumpModel:
    def __init__(self, n_states: int, lam: float, means: np.ndarray):
        self.n_states = n_states
        self.lam = lam
        self.means_ = means

    def forward_states(self, X: np.ndarray) -> np.ndarray:
        """KAUZÁLIS forward dekódolás: state_t = argmin_k F_k(t), ahol
        F_k(t) = ||x_t−μ_k||² + min_j[F_j(t−1) + λ·1(j≠k)].
        F_k(t) kizárólag x[0..t]-től függ → prefix-invariáns."""
        X = np.asarray(X, dtype=float)
        T, K = len(X), self.n_states
        emit = np.sum((X[:, None, :] - self.means_[None, :, :]) ** 2, axis=2)
        F = np.empty((T, K))
        F[0] = emit[0]
        for t in range(1, T):
            prev_min = np.array([np.min(F[t - 1] + self.lam * (np.arange(K) != k))
                                 for k in range(K)])
            F[t] = emit[t] + prev_min
        return np.argmin(F, axis=1)

    def forward_proba(self, X: np.ndarray) -> np.ndarray:
        """Lágy állapot-valószínűség a forward költségek softmaxából (kauzális),
        a HMM szűrt poszterior analógiája — jellemzőként használható."""
        X = np.asarray(X, dtype=float)
        T, K = len(X), self.n_states
        emit = np.sum((X[:, None, :] - self.means_[None, :, :]) ** 2, axis=2)
        F = np.empty((T, K))
        F[0] = emit[0]
        for t in range(1, T):
            prev_min = np.array([np.min(F[t - 1] + self.lam * (np.arange(K) != k))
                                 for k in range(K)])
            F[t] = emit[t] + prev_min
        neg = -(F - F.min(axis=1, keepdims=True))
        e = np.exp(neg)
        return e / e.sum(axis=1, keepdims=True)


def fit_jump_model(X_train: np.ndarray, n_states: int = 3, lam: float = 50.0,
                   n_iter: int = 20, seed: int = 42) -> JumpModel:
    """Koordináta-leszállás: (a) állapot-hozzárendelés offline DP-vel,
    (b) μ_k = a k-hoz rendelt pontok átlaga. KIZÁRÓLAG train-en."""
    X = np.asarray(X_train, dtype=float)
    rng = np.random.default_rng(seed)
    # k-means++ szerű inicializálás
    idx = rng.choice(len(X), n_states, replace=False)
    mu = X[idx].copy()
    prev_states = None
    for _ in range(n_iter):
        states = _assign_states_dp(X, mu, lam)
        for k in range(n_states):
            m = states == k
            if m.any():
                mu[k] = X[m].mean(axis=0)
        if prev_states is not None and np.array_equal(states, prev_states):
            break
        prev_states = states
    # Rendezés az 1. jellemző (hozam) átlaga szerint — reprodukálható címkézés
    order = np.argsort(mu[:, 0])
    return JumpModel(n_states, lam, mu[order])


def persistence(states: np.ndarray) -> float:
    """Átlagos tartózkodási idő (rezsim-perzisztencia): a szegmensek átlagos hossza.
    A jump-modellnek itt kell vernie a HMM-et (kevesebb, hosszabb szegmens)."""
    states = np.asarray(states)
    switches = int((np.diff(states) != 0).sum())
    return len(states) / (switches + 1)
