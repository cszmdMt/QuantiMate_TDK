"""HPO a GBDT-győztesre, kemény őrökkel (V8-8).

Kereső: SEEDELT VÉLETLEN KERESÉS (Bergstra–Bengio 2012) — az Optuna helyett,
mert új kötelező függőség nem vehető fel, és a determinizmus elsőbbséget
élvez; fix trial-budget mellett a TPE-előny marginális.

Belső validáció: NESTED PURGED K-FOLD kizárólag a train-szakaszon — a K
összefüggő blokk mindegyike egyszer val-fold (cpcv_splits, k_test=1), a
purge+embargo sáv mindkét oldalon kizárva. Egy trial pontszáma a foldokon
összefűzött out-of-fold pinball-veszteség átlaga; a fold-mátrixból PBO (CSCV)
számolható — az őr: a HPO-győztes PBO-ja nem jelezhet érdemi túlillesztést.

A KÜLSŐ val-halmazhoz a keresés nem nyúl: a győztes konfiguráció egyetlen
újratanítást kap a szokásos úton (train + val-ES), és a szokásos kapun
(Δval-pinball ≤ −0,002 ÉS DM p<0,10) versenyez a default hiperparaméterekkel.
"""
from __future__ import annotations

import numpy as np

from quantimate.models.models_gbdt import DEFAULT_HYPERPARAMS, train_gbdt_quantiles
from quantimate.validation.validation_cpcv import cpcv_splits

# Keresési tér — a default (400 fa, lr=0.05, 31 levél, ...) a tér belsejében.
SEARCH_SPACE = {
    "learning_rate": ("log", 0.02, 0.15),
    "num_leaves": ("int", 15, 127),
    "min_child_samples": ("int", 10, 80),
    "subsample": ("float", 0.6, 1.0),
    "colsample_bytree": ("float", 0.6, 1.0),
    "reg_alpha": ("log", 1e-4, 1.0),
    "reg_lambda": ("log", 1e-4, 1.0),
}


def sample_config(rng: np.random.Generator) -> dict:
    """Egy hiperparaméter-konfiguráció a keresési térből (seedelt)."""
    hp = {}
    for name, (kind, lo, hi) in SEARCH_SPACE.items():
        if kind == "log":
            hp[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        elif kind == "int":
            hp[name] = int(rng.integers(lo, hi + 1))
        else:
            hp[name] = float(rng.uniform(lo, hi))
    if "subsample" in hp and hp["subsample"] < 1.0:
        hp["subsample_freq"] = 1          # enélkül a subsample hatástalan
    return hp


def _pinball_rows(y: np.ndarray, q: np.ndarray, quants) -> np.ndarray:
    """Soronkénti átlagos pinball (horizont- és kvantilis-átlag)."""
    losses = []
    for q_i, tau in enumerate(quants):
        diff = y - q[:, :, q_i]
        losses.append(np.maximum(tau * diff, (tau - 1) * diff))
    return np.mean(np.stack(losses, axis=-1), axis=(1, 2))


def oof_losses(X: np.ndarray, Y: np.ndarray, train_pos: np.ndarray,
               horizons, quantiles, hyperparams: dict | None, *,
               n_folds: int = 4, purge: int = 21, embargo: int = 5,
               seed: int = 42, allow_nan: bool = False,
               early_stopping_rounds: int = 50) -> np.ndarray:
    """Egy konfiguráció out-of-fold veszteség-sora a train-szakaszon.

    A train_pos tartományt n_folds összefüggő blokkra osztjuk; minden blokk
    egyszer fold-val (ES is ezen fut), a fold-train a purge+embargo-val
    megtisztított maradék. Vissza: a train_pos hosszú OOF pinball-sor
    (minden sor pontosan egy fold val-jában szerepel)."""
    n = len(train_pos)
    out = np.full(n, np.nan)
    for tr_rel, te_rel in cpcv_splits(n, n_groups=n_folds, k_test=1,
                                      purge=purge, embargo=embargo):
        tr_abs, te_abs = train_pos[tr_rel], train_pos[te_rel]
        m = train_gbdt_quantiles(X, Y, tr_abs, te_abs, horizons, quantiles,
                                 seed=seed, hyperparams=hyperparams,
                                 early_stopping_rounds=early_stopping_rounds,
                                 allow_nan=allow_nan)
        q_te = m.predict_quantiles(X[te_abs])
        out[te_rel] = _pinball_rows(Y[te_abs], q_te, quantiles)
    assert not np.isnan(out).any(), "OOF-lefedettségi lyuk (fold-particionálás)"
    return out


def random_search(X: np.ndarray, Y: np.ndarray, train_pos: np.ndarray,
                  horizons, quantiles, *, n_trials: int = 30,
                  n_folds: int = 4, seed: int = 42,
                  allow_nan: bool = False, base_hp: dict | None = None,
                  verbose: bool = True) -> dict:
    """Seedelt véletlen keresés; a 0. trial mindig a DEFAULT konfiguráció
    (így a győztesnek a defaultot OOF-ban is vernie kell). base_hp minden
    trialra rákényszerített felülírás (SMOKE: kis fa-szám). Vissza:
    best_hp, best_score, trials (konfig+score), loss_matrix (n_train, n_trial)."""
    rng = np.random.default_rng(seed)
    configs = [dict(DEFAULT_HYPERPARAMS)] + [sample_config(rng)
                                             for _ in range(n_trials - 1)]
    if base_hp:
        configs = [{**c, **base_hp} for c in configs]
    losses, trials = [], []
    for i, hp in enumerate(configs):
        row = oof_losses(X, Y, train_pos, horizons, quantiles, hp,
                         n_folds=n_folds, seed=seed, allow_nan=allow_nan)
        losses.append(row)
        trials.append({"trial": i, "score": float(row.mean()),
                       "default": i == 0, **hp})
        if verbose:
            print(f"  trial {i:>2}/{n_trials - 1}: OOF-pinball "
                  f"{row.mean():.4f}{'  (default)' if i == 0 else ''}")
    loss_matrix = np.column_stack(losses)          # (n_train, n_trials)
    scores = loss_matrix.mean(axis=0)
    best = int(np.argmin(scores))
    return {"best_hp": configs[best], "best_trial": best,
            "best_score": float(scores[best]),
            "default_score": float(scores[0]),
            "trials": trials, "loss_matrix": loss_matrix}
