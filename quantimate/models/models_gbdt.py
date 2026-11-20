"""LightGBM kvantilis-baseline (V7-5) — a hiányzó legerősebb tabuláris ellenfél.

Az M5 verseny (pont- és kvantilis-ág) győztes módszerei LightGBM-alapúak
(Makridakis et al. 2022, IJF 38(4)); közepes tabuláris adaton a fák a mély
hálókat tipikusan verik (Grinsztajn et al. 2022, NeurIPS). Ha a GBDT veri az
NN-eket, azt az őszinte kimondás elve szerint vállalni kell.

Determinizmus: deterministic=True, force_row_wise=True, rögzített seed,
num_threads=1 — így két futás bitre azonos (pytest bizonyítja).

Bemenet: TABULÁRIS nézet — az utolsó napi jellemzővektor + kiválasztott
log-hozam lagok. A fa nem igényel skálázást, ezért a NYERS (nem RobustScaler-ezett)
jellemzőkön fut — dokumentált eltérés a szekvencia-modellektől; a split-fegyelem
(purged pozíciók) azonos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_LAGS = (2, 3, 5, 10, 21)

DETERMINISM_PARAMS = dict(
    deterministic=True,
    force_row_wise=True,
    num_threads=1,        # bit-reprodukálhatóság > sebesség (dokumentálva)
    verbosity=-1,
)

DEFAULT_HYPERPARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=1.0,
    colsample_bytree=1.0,
)


def tabular_view(feats: pd.DataFrame, target: str,
                 lags=DEFAULT_LAGS) -> pd.DataFrame:
    """Utolsó napi jellemzők + log-hozam lagok (pozitív shift — csak múlt)."""
    tab = feats.copy()
    r = feats[f"{target}_log_ret"]
    for k in lags:
        tab[f"{target}_log_ret_lag{k}"] = r.shift(k)
    return tab


class GBDTQuantileModel:
    """Kvantilisenként × horizontonként külön LGBMRegressor; a kimenet a
    kvantilis-tengelyen RENDEZVE (kvantilis-keresztezés elleni V6-korrekció)."""

    def __init__(self, models: dict, horizons, quantiles):
        self.models = models
        self.horizons = tuple(horizons)
        self.quantiles = tuple(quantiles)

    def predict_quantiles(self, X_tab: np.ndarray) -> np.ndarray:
        out = np.empty((len(X_tab), len(self.horizons), len(self.quantiles)))
        for h_i, h in enumerate(self.horizons):
            for q_i, q in enumerate(self.quantiles):
                out[:, h_i, q_i] = self.models[(h, q)].predict(X_tab)
        return np.sort(out, axis=-1)

    def best_iterations(self) -> dict:
        return {k: getattr(m, "best_iteration_", None) or m.n_estimators
                for k, m in self.models.items()}


def train_gbdt_quantiles(X_tab: np.ndarray, Y: np.ndarray,
                         train_pos: np.ndarray, val_pos: np.ndarray,
                         horizons, quantiles, *, seed: int = 42,
                         early_stopping_rounds: int = 50,
                         hyperparams: dict | None = None,
                         allow_nan: bool = False) -> GBDTQuantileModel:
    """15 kis kvantilis-modell tanítása; early stopping a val-on (val-döntés).
    A train/val pozíciók a purged splitekből jönnek.

    allow_nan=True (V8): az availability-maszkos jellemzők (intraday ~2023-tól,
    fundamentum 2018/2019-től) NaN-jait a LightGBM natívan kezeli — a V7-es
    NaN-assert ilyenkor a lag-bemelegedés hibát nem tudná megkülönböztetni a
    jogos maszktól, ezért kapcsolható."""
    import lightgbm as lgb

    hp = dict(DEFAULT_HYPERPARAMS)
    if hyperparams:
        hp.update(hyperparams)

    X_tab = np.asarray(X_tab, dtype=float)
    used = np.concatenate([train_pos, val_pos])
    if not allow_nan:
        assert not np.isnan(X_tab[used]).any(), \
            "NaN a tabuláris nézet használt soraiban (lag-bemelegedés hiba)"

    models = {}
    for h_i, h in enumerate(horizons):
        for q in quantiles:
            m = lgb.LGBMRegressor(objective="quantile", alpha=float(q),
                                  seed=seed, **hp, **DETERMINISM_PARAMS)
            m.fit(X_tab[train_pos], Y[train_pos, h_i],
                  eval_set=[(X_tab[val_pos], Y[val_pos, h_i])],
                  eval_metric="quantile",
                  callbacks=[lgb.early_stopping(early_stopping_rounds,
                                                verbose=False)])
            models[(h, float(q))] = m
    return GBDTQuantileModel(models, horizons, [float(q) for q in quantiles])


def shap_contributions(model: GBDTQuantileModel, X_tab: np.ndarray,
                       horizon, quantile: float = 0.5) -> np.ndarray:
    """Egzakt TreeSHAP a LightGBM beépített pred_contrib-jával (F2-17-hez;
    nincs külön shap-függőség). Visszaad: (B, n_feat+1) — az utolsó oszlop a bázis."""
    booster = model.models[(horizon, float(quantile))].booster_
    return np.asarray(booster.predict(X_tab, pred_contrib=True))
