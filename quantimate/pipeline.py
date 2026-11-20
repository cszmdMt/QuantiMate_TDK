"""Közös pipeline-összeállítás: a V6 notebook adat→jellemző→target→
split→HMM→skálázás lépéseinek PONTOS megfelelője egyetlen függvényben.

Ezt hívja a golden-run teszt és a mainV7.ipynb is — így a notebook és a
tesztek garantáltan ugyanazt a kódutat futtatják. Az alapértelmezett config
V6-ekvivalens; a V7-blokkok flag mögött kapcsolódnak be.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantimate.config import PipelineConfig
from quantimate.features.features import build_features, generate_earnings_candidates
from quantimate.models.regimes import fit_hmm, hmm_filtered_proba, label_regimes
from quantimate.validation.splits import Splits, fit_train_scaler, make_purged_splits
from quantimate.features.targets import build_targets

EARNINGS_SIGMA = 2.5   # dokumentált V6-döntés (4σ → 6/40 negyedév, túl ritka)
HMM_SEED = 42


@dataclass
class PipelineArtifacts:
    """A pipeline minden downstream-hez szükséges terméke."""
    df: pd.DataFrame
    feats: pd.DataFrame
    feature_cols: list
    regime_cols: list
    X_scaled: np.ndarray
    scaler: object
    S: Splits
    tg: pd.DataFrame
    Y_all: np.ndarray
    dir_all: np.ndarray
    fwd1_all: np.ndarray
    sig_all: np.ndarray          # a TARGET-normáló σ (sigma_mode szerint)
    hmm: object
    regime_names: list
    filtered_o: np.ndarray
    earn_dates: pd.DatetimeIndex
    n_labeled: int
    T: int
    sigma_mode: str = "ewma"
    extras: dict = field(default_factory=dict)


def _target_sigma(feats: pd.DataFrame, df: pd.DataFrame, cfg: PipelineConfig,
                  train_end_pos: int) -> tuple[pd.Series, dict]:
    """A target-normáló σ_t a sigma_mode flag szerint (a σ-definíció váltása
    új target-teret nyit — az összevetés denormalizált térben történik).

    A YZ/HAR a TELJES df-en számolódik (a jellemző-tartomány előtti history a
    bemelegedést fedezi), majd a feats-indexre szűkül; a HAR illesztése a
    train-szakasz utolsó dátumáig tartó CÉL-napokon történik.
    """
    extras: dict = {}
    if cfg.sigma_mode == "ewma":
        return feats[f"{cfg.target}_ewma_vol"], extras
    if cfg.sigma_mode in ("yz", "yz_har"):
        from quantimate.features.volatility import har_rv_forecast, yang_zhang_sigma
        yz_full = yang_zhang_sigma(df, cfg.target, window=cfg.yz_window)
        if cfg.sigma_mode == "yz":
            sig = yz_full.reindex(feats.index)
        else:
            train_end_date = feats.index[train_end_pos - 1]
            sig = har_rv_forecast(yz_full, train_end_date).reindex(feats.index)
        assert not sig.isna().any(), "σ NaN a jellemző-tartományban (bemelegedés-hiba)"
        extras["sigma_series"] = sig
        return sig, extras
    raise ValueError(f"Ismeretlen sigma_mode: {cfg.sigma_mode}")


def build_pipeline(df: pd.DataFrame, cfg: PipelineConfig,
                   horizons: tuple | None = None,
                   verbose: bool = True) -> PipelineArtifacts:
    """A V6 notebook 4–8. szakaszának megfelelő, determinisztikus összeállítás.

    Sorrend (bitre a V6 szerint): earnings-jelöltek → jellemzők → EGYSZERI
    dropna → targetek → purged splitek → HMM (train-only fit, kauzális szűrés,
    obs = [log_ret, EWMA-vol] MINDIG — a sigma_mode a targetet normálja, nem a
    rezsim-megfigyelést) → rezsim-oszlopok → train-only scaler.
    """
    horizons = tuple(horizons if horizons is not None else cfg.horizons)
    purge = 21   # mindig a teljes H_max-hoz igazítva (SMOKE-ban is)

    # V8-4: hitelesített earnings-naptár (EDGAR+Yahoo,
    # hitelesített CSV) — a days_to_next az ELŐRE ISMERT bejelentési dátumra
    # számol (announce); default: V6-ekvivalens generált jelöltek (golden!).
    if cfg.earnings_source == "verified":
        from quantimate.data.earnings import load_verified_earnings
        earn_dates = load_verified_earnings(kind="announce")
    else:
        earn_dates = generate_earnings_candidates(df[f"{cfg.target}_Close"],
                                                  threshold_sigma=EARNINGS_SIGMA)
    feats_full = build_features(df, cfg.target, cfg.ohlcv_tickers,
                                cfg.light_tickers, earn_dates)

    # V7-2: makro-jellemzők (EPU/VIX-term/FRED, release-lag fegyelemmel)
    if cfg.use_macro:
        from quantimate.features.macro import build_macro_features
        macro = build_macro_features(feats_full.index)
        feats_full = pd.concat([feats_full, macro], axis=1)

    # V8-3: geopolitikai jellemzők (GPR napi + EPU-trade, lag-fegyelemmel) —
    # teljes historikus lefedettség (1985+), ezért a macro-mintát követi.
    if cfg.use_geo:
        from quantimate.features.geo import build_geo_features
        geo = build_geo_features(feats_full.index, use_gpr=cfg.geo_gpr,
                                 use_tpu=cfg.geo_tpu)
        feats_full = pd.concat([feats_full, geo], axis=1)

    # V7-3: frakcionális differenciálás — d* a train-szakaszból (dátum-alapú
    # cutoff a végleges split ELŐTT, hogy elkerüljük a körkörös függést;
    # a d* kizárólag train-adaton dől el)
    if cfg.use_fracdiff:
        from quantimate.features.fracdiff import find_min_d, frac_diff_ffd
        # Looser thresh (1e-3) → rövidebb FFD-ablak → kevesebb bemelegedési
        # sorveszteség (a memória-megőrzés vs. mintaszám dokumentált kompromisszuma).
        FFD_THRESH = 1e-3
        base_idx = feats_full.dropna().index
        cutoff = base_idx[int(len(base_idx) * cfg.train_frac)]
        for col in (f"{cfg.target}_Close_log", f"{cfg.target}_rsi14"):
            src = (np.log(df[f"{cfg.target}_Close"]) if col.endswith("_log")
                   else feats_full[f"{cfg.target}_rsi14"])
            src = src.reindex(feats_full.index)
            d_res = find_min_d(src.dropna(), cutoff, thresh=FFD_THRESH)
            feats_full[f"{col}_ffd"] = frac_diff_ffd(
                src, d_res["d"], thresh=FFD_THRESH).reindex(feats_full.index)

    feats = feats_full.dropna().copy()
    assert (feats.index == df.index[-len(feats):]).all(), \
        "A jellemző-mátrix nem összefüggő a dropna után"

    log_ret = feats[f"{cfg.target}_log_ret"]
    ewma_sigma = feats[f"{cfg.target}_ewma_vol"]

    T = len(feats)
    n_labeled = T - max(horizons)

    S = make_purged_splits(n_labeled, window=cfg.window,
                           train_frac=cfg.train_frac, val_frac=cfg.val_frac,
                           purge=purge, embargo=cfg.embargo)
    train_end = int(S.train.max()) + 1

    # Target-normáló σ (V7-1: EWMA/YZ/HAR) — a HAR fitje train-only
    sigma, extras = _target_sigma(feats, df, cfg, train_end_pos=train_end)

    # V7-1b: HAR-σ előrejelzés JELLEMZŐKÉNT (flag; train-only fit — a
    # walk-forward refit-checklist tétele)
    if cfg.use_har_feature:
        from quantimate.features.volatility import har_rv_forecast, yang_zhang_sigma
        yz_full = yang_zhang_sigma(df, cfg.target, window=cfg.yz_window)
        har_feat = har_rv_forecast(yz_full, feats.index[train_end - 1])
        feats["har_sigma"] = har_feat.reindex(feats.index)
        assert not feats["har_sigma"].isna().any()

    tg = build_targets(log_ret, sigma, horizons=horizons)
    y_cols = [f"y_h{h}" for h in horizons]
    assert tg[y_cols].iloc[:n_labeled].notna().all().all()

    # HMM: fit CSAK train-en, kauzális szűrés a teljes soron
    hmm_obs = np.column_stack([log_ret.to_numpy(), ewma_sigma.to_numpy()])
    hmm = fit_hmm(hmm_obs[:train_end], n_components=3, seed=HMM_SEED)
    filtered = hmm_filtered_proba(hmm, hmm_obs)
    order, regime_names = label_regimes(hmm)
    filtered_o = filtered[:, order]

    regime_cols = []
    for i in range(3):
        feats[f"regime_p{i}"] = filtered_o[:, i]
        regime_cols.append(f"regime_p{i}")

    feature_cols = list(feats.columns)
    X_full_mat = feats[feature_cols].to_numpy(dtype=float)
    scaler, X_scaled = fit_train_scaler(X_full_mat, S.train, clip=cfg.clip_sigma)

    # V8-1: intraday-aggregátumok az EXTRAS-ban utaznak (a GBDT-A/B útja) —
    # nem az X_scaled-ben: az availability-mask NaN-jai (60m csak ~730 nap)
    # az NN/scaler utat törnék; a LightGBM natívan kezeli őket.
    if cfg.use_intraday:
        from quantimate.data.intraday import build_intraday_features
        extras["intraday_feats"] = build_intraday_features(feats.index,
                                                           ticker=cfg.target)

    # V8-2: fundamentum–tulajdonosi jelcsaládok — szintén extras-út (a
    # lefedettségi NaN-ok miatt), a GBDT-A/B fogyasztja az F2-ben.
    if cfg.use_fundamentals:
        from quantimate.data.fundamentals import build_fundamental_features
        extras["fundamental_feats"] = build_fundamental_features(
            feats.index, use_insider=cfg.fund_insider, use_si=cfg.fund_si,
            use_shortvol=cfg.fund_shortvol, verbose=verbose)

    Y_all = np.column_stack([tg[f"y_h{h}"].to_numpy() for h in horizons])
    dir_all = tg["direction"].to_numpy()
    fwd1_all = tg["fwd_ret_h1"].to_numpy()
    sig_all = sigma.clip(lower=1e-8).to_numpy()

    if verbose:
        print(f"Pipeline (sigma_mode={cfg.sigma_mode}): {T} sor, "
              f"{len(feature_cols)} jellemző, {n_labeled} címkézett nap")
        for name, block in (("train", S.train), ("val", S.val), ("test", S.test)):
            print(f"  {name:<6} {len(block):>5} minta "
                  f"({feats.index[block.min()].date()} → {feats.index[block.max()].date()})")

    return PipelineArtifacts(
        df=df, feats=feats, feature_cols=feature_cols, regime_cols=regime_cols,
        X_scaled=X_scaled, scaler=scaler, S=S, tg=tg, Y_all=Y_all,
        dir_all=dir_all, fwd1_all=fwd1_all, sig_all=sig_all, hmm=hmm,
        regime_names=regime_names, filtered_o=filtered_o,
        earn_dates=earn_dates, n_labeled=n_labeled, T=T,
        sigma_mode=cfg.sigma_mode, extras=extras)
