"""V8 / V8.5 repeated-seed experiment over (target, horizon) tracks. Execution is always explicit.

All model selection sees validation only. Test observations never tune a model.
The final refit is a separate fresh model on mature labels through the cutoff.

Tracks (protocol schema 3): target "price" or "volatility", horizon 1, 5 or 21 sessions.
- price: forecast of the adjusted close; benchmark = last close (naive); loss = APE.
  Every price forecast also carries a 90% and a 50% band: the training quantiles
  of the normalized target, corrected by adaptive conformal inference.
- volatility: forecast of the realized variance of the next h sessions (range-based
  daily proxy); benchmark = rescaled EWMA variance; loss = QLIKE.

Selection is built to refuse overfitted models, identically on every track:
- the benchmark is the default;
- candidates span small-to-large tree counts, shrinkage toward the benchmark and
  a low-capacity configuration (price: also a constant median; volatility: also
  a log-HAR model and the geometric mean of HAR and the tree model);
- a candidate replaces the benchmark only if (1) Hansen's SPA test, run jointly
  over ALL candidates on validation, rejects "nothing beats the benchmark", and
  (2) the validation winner also beats it out-of-fold inside the training window;
- V8.5 searches hyper-parameters by purged K-fold inside the training window
  and admits the winner to validation only under a PBO guard.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd

from quantimate.experiment_data import CORE, LIGHT, Snapshot
from quantimate.experiment_protocol import (ACI_GAMMA, BAND_LEVELS, BASE_CONFIGS, BENCHMARK_ARM,
    BENCHMARK_ARMS, CONFORMAL_HORIZONS, DATE_PAIRS, EMBARGO, GATE_ALPHA, HAR_WINDOWS, HPO_FOLDS,
    HPO_TRIALS, ITERATION_GRID, PBO_BLOCKS, PBO_LIMIT, PURGE, QUANTILES, REPLICATION_FOLDS, SEEDS,
    SELECTION_SEED, SHRINKAGE, SPA_BOOTSTRAPS, TRACKS, VERSIONS, VOLATILITY_COMBINE,
    VOLATILITY_FALLBACK,
    export_individual, price_metrics, read_json, result_valid, save_result, spa_block,
    variance_metrics)
from quantimate.features.features import compute_indicators, compute_index_features, overnight_intraday, calendar_features
from quantimate.models.models_gbdt import tabular_view
from quantimate.features.pooled import POOLED_UNIVERSE, generic_feature_frame, generic_tabular
from quantimate.models.regimes import fit_hmm_scaled, hmm_filtered_proba, label_regimes
from quantimate.validation.splits import make_purged_splits
from quantimate.features.targets import build_targets, forward_realized_variance
from quantimate.features.volatility import daily_range_variance
from quantimate.validation.conformal import aci_bands_delayed

VALIDATION_METRIC = {"price": "validation_mape_pct", "volatility": "validation_qlike"}
LOSS_METRIC = {"price": "ape_pct", "volatility": "qlike"}
RATIO_FLOOR = 1e-6          # lower bound of a predicted variance ratio before taking logs
EARNINGS_LONG_CAP = 80      # sessions; longer than one reporting quarter (about 63)


def set_determinism():
    """Reset the non-GBDT RNGs used here; each model receives its explicit seed.

    TensorFlow is not part of this GBDT-only experiment. BLAS thread limits and
    PYTHONHASHSEED are set before the fresh kernel starts, by the runner.
    """
    random.seed(42)
    np.random.seed(42)


@dataclass
class Design:
    """y is the model target: price -> h-session log return / (sigma * sqrt(h));
    volatility -> realized variance of the next h sessions / (h * sigma^2)."""
    x: pd.DataFrame
    y: pd.Series
    close: pd.Series
    sigma: pd.Series
    target_dates: pd.Series
    target: str = "price"
    horizon: int = 1
    variance: pd.Series = None      # daily range-based variance, full session calendar
    realized: pd.Series = None      # sum of the next h daily variances

    @property
    def labeled(self):
        return self.x.index[self.y.reindex(self.x.index).notna()]


def baseline_features(frame):
    pieces = [compute_indicators(frame, t, full=t == "NVDA") for t in CORE]
    pieces += [compute_index_features(frame, t.replace("^", "")) for t in LIGHT]
    pieces += [overnight_intraday(frame, "NVDA"),
               calendar_features(frame.index).drop(columns="cal_month_end")]
    # No hindsight-generated earnings dates; no prefix-dependent month-end dummy.
    return tabular_view(pd.concat(pieces, axis=1), "NVDA")


def make_design(frame, cutoff, *, generic=False, ticker="NVDA", target="price", horizon=1):
    if (target, horizon) not in TRACKS:
        raise ValueError(f"Unknown track: {target} / {horizon}")
    if len(frame) == 0 or frame.index[-1] > pd.Timestamp(cutoff):
        raise ValueError("Feature builder received post-cutoff data")
    x = (generic_tabular(generic_feature_frame(frame, ticker)).drop(columns="cal_month_end")
         if generic else baseline_features(frame))
    prefix = "tgt" if generic else "NVDA"
    # Build labels on the original session calendar BEFORE dropping incomplete features.
    sigma = x[prefix + "_ewma_vol"].clip(lower=1e-8)
    variance = daily_range_variance(frame, ticker)
    realized = forward_realized_variance(variance, horizon)
    if target == "price":
        y = build_targets(x[prefix + "_log_ret"], sigma, horizons=(horizon,))[f"y_h{horizon}"]
    else:
        y = realized / (horizon * sigma ** 2)
        # HAR terms (Corsi 2009) relative to the EWMA variance: causal rolling means.
        x = x.assign(**{f"har_ratio_{w}": variance.rolling(w).mean() / sigma ** 2 for w in HAR_WINDOWS})
    dates = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    dates.iloc[:-horizon] = frame.index[horizon:].to_numpy()
    start = pd.Timestamp(cutoff) - pd.DateOffset(years=10)
    x = x.replace([np.inf, -np.inf], np.nan).dropna().loc[start:cutoff]
    if ticker == "NVDA" and pd.Timestamp(cutoff) not in x.index:
        raise ValueError("No complete feature row at cutoff")
    if generic:
        x["tick_id"] = float(POOLED_UNIVERSE.index(ticker))
    return Design(x, y, frame[ticker + "_Close"], sigma, dates, target, horizon, variance, realized)


def past_earnings(index, calendar, long_counter=False):
    events = pd.DatetimeIndex(pd.to_datetime(calendar["effective_date"])).sort_values().unique()
    # Discard future events before mapping; never clip a future event onto the last row.
    events = events[events <= index[-1]]
    positions = np.searchsorted(index.values, events.values)
    positions = positions[positions < len(index)]
    out = pd.DataFrame(index=index, columns=["earn_days_since_prev"], dtype=float)
    if len(positions):
        previous = np.searchsorted(positions, np.arange(len(index)), side="right") - 1
        valid = previous >= 0
        since = np.arange(len(index))[valid] - positions[previous[valid]]
        out.loc[valid, "earn_days_since_prev"] = np.minimum(since, 30)
        if long_counter:
            # Volatility tracks: reports recur about every 63 sessions, so the time since
            # the LAST report tells how near the next one is. Past information only.
            out.loc[valid, "earn_days_since_prev_long"] = np.minimum(since, EARNINGS_LONG_CAP)
    if long_counter and "earn_days_since_prev_long" not in out:
        out["earn_days_since_prev_long"] = np.nan
    return out


def optional_features(snapshot, index, version, target="price"):
    from quantimate.data.fundamentals import insider_features, short_interest_features, shortvol_features
    blocks, notes = {}, []
    calendar = snapshot.optional("nvda_earnings_verified.csv")
    if calendar is not None:
        blocks["earnings"] = past_earnings(index, calendar, long_counter=target == "volatility")
    tx = snapshot.optional("edgar_cache/form4_nvda.parquet")
    if tx is not None:
        tx["filing_date"] = pd.to_datetime(tx["filing_date"])
        blocks["insider"] = insider_features(index, tx.loc[tx.filing_date <= index[-1]])
    si = snapshot.optional("finra_si_NVDA.parquet")
    if si is not None:
        si["settlementDate"] = pd.to_datetime(si["settlementDate"])
        blocks["short_interest"] = short_interest_features(index, si.loc[si.settlementDate <= index[-1]])
    sv = snapshot.optional("regsho_NVDA.parquet")
    if sv is not None:
        sv["date"] = pd.to_datetime(sv["date"])
        blocks["short_volume"] = shortvol_features(index, sv.loc[sv.date <= index[-1]])
    if version == "V8":
        from quantimate.data.intraday import build_intraday_daily
        from quantimate.features.geo import gpr_features
        bars = snapshot.optional("NVDA_60m.parquet")
        if bars is not None:
            local = bars.index.tz_convert("America/New_York").tz_localize(None) if bars.index.tz is not None else bars.index
            daily = build_intraday_daily(bars.loc[local.normalize() <= index[-1]])
            blocks["intraday"] = daily.reindex(index)
        gpr = snapshot.optional("gpr_daily.parquet")
        if gpr is not None:
            blocks["geo"] = gpr_features(index, gpr.loc[:index[-1]])
        epu = snapshot.optional("fred_EPUTRADE.parquet")
        if epu is not None:
            s = epu.iloc[:, 0].sort_index().loc[:index[-1]]
            available = s.index + pd.offsets.MonthEnd(0) + pd.Timedelta(days=7)
            pos = np.searchsorted(available.values, index.values, side="right") - 1
            valid = pos >= 0
            out = pd.DataFrame(index=index, columns=["eputrade_level", "eputrade_chg12"], dtype=float)
            out.loc[valid, "eputrade_level"] = s.to_numpy()[pos[valid]]
            out.loc[valid, "eputrade_chg12"] = s.diff(12).to_numpy()[pos[valid]]
            blocks["epu_trade"] = out
    notes += ["Missing snapshot source: " + p for p in snapshot.manifest["missing_optional"]]
    # Empty pre-coverage blocks must keep numeric dtypes as well.
    return {name: block.astype(float) for name, block in blocks.items()}, notes


def add_hmm(design, train_dates):
    x = design.x.copy()
    obs = x[["NVDA_log_ret", "NVDA_ewma_vol"]].to_numpy(dtype=float)
    train_pos = x.index.get_indexer(train_dates)
    if (train_pos < 0).any():
        raise ValueError("HMM training date mismatch")
    set_determinism()
    # Standardised with fit-window statistics only; see fit_hmm_scaled for why.
    hmm, center, scale = fit_hmm_scaled(obs[train_pos], n_components=3, seed=42, n_iter=300)
    order, _ = label_regimes(hmm)
    probabilities = hmm_filtered_proba(hmm, (obs - center) / scale)[:, order]
    for i in range(3):
        x[f"regime_p{i}"] = probabilities[:, i]
    return x


def build_arms(snapshot, cutoff, version, design, fit_dates):
    base = add_hmm(design, fit_dates)
    arms = {"baseline": base}
    full_index = snapshot.daily("NVDA", cutoff).index
    blocks, notes = optional_features(snapshot, full_index, version, design.target)
    available = {}
    for name, block in blocks.items():
        block = block.reindex(base.index).replace([np.inf, -np.inf], np.nan)
        # Never choose an arm supported only by future/test-period data.
        columns = [c for c in block if block.loc[fit_dates, c].notna().sum() >= 50
                   and pd.notna(block.loc[pd.Timestamp(cutoff), c])]
        if columns:
            available[name] = block[columns]
            arms[name] = pd.concat([base, block[columns]], axis=1)
        else:
            notes.append(name + ": skipped (fewer than 50 observed training rows or missing cutoff value)")
    if version == "V8" and available:
        arms["all_available"] = pd.concat([base, *available.values()], axis=1)
    if version == "V8.5":
        generic = make_design(snapshot.universe(cutoff, generic=True), cutoff, generic=True,
                              target=design.target, horizon=design.horizon)
        view = generic.x.reindex(base.index)
        if view.isna().any().any():
            raise ValueError("Generic NVDA features do not cover the baseline evaluation dates")
        arms["generic_nvda"] = view
        arms["pooled"] = view.copy()
    return arms, notes


def pooled_training(snapshot, cutoff, fit_dates, columns, design):
    xs, ys, counts = [], [], {}
    for ticker in POOLED_UNIVERSE:
        frame = snapshot.universe(cutoff, ticker=ticker, generic=True)
        d = make_design(frame, cutoff, generic=True, ticker=ticker,
                        target=design.target, horizon=design.horizon)
        dates = d.labeled.intersection(fit_dates)
        if len(dates):
            if (d.target_dates.loc[dates] > pd.Timestamp(cutoff)).any():
                raise ValueError("Immature pooled label")
            xs.append(d.x.loc[dates, columns])
            ys.append(d.y.loc[dates])
        counts[ticker] = len(dates)
    if not xs:
        raise ValueError("No pooled training data")
    return pd.concat(xs, ignore_index=True), pd.concat(ys, ignore_index=True), counts


def training_data(arm, x, design, fit_dates, snapshot, cutoff):
    if arm == "pooled":
        return pooled_training(snapshot, cutoff, fit_dates, list(x.columns), design)
    return x.loc[fit_dates], design.y.loc[fit_dates], {"NVDA": len(fit_dates)}


def lgbm_params(hp, seed, iterations):
    params = dict(learning_rate=0.05, num_leaves=31, min_child_samples=20,
                  reg_alpha=0.0, reg_lambda=0.0)
    params.update(hp)
    params.update(n_estimators=iterations, subsample=0.8, colsample_bytree=0.8,
                  subsample_freq=1, random_state=seed, bagging_seed=seed,
                  feature_fraction_seed=seed, data_random_seed=seed,
                  deterministic=True, force_row_wise=True, n_jobs=1, verbosity=-1)
    return params


def fit_quantiles(x, y, hp, seed, iterations, quantiles=QUANTILES):
    from lightgbm import LGBMRegressor
    set_determinism()
    params = lgbm_params(hp, seed, iterations)
    models = []
    for quantile in quantiles:
        model = LGBMRegressor(objective="quantile", alpha=quantile, **params)
        model.fit(x, y)
        models.append(model)
    return models


def fit_variance(x, y, hp, seed, iterations):
    """Variance-ratio model. The gamma deviance is the QLIKE loss, so the fitted value
    is the conditional MEAN of the ratio - the quantity QLIKE scores (a median would
    be biased low on a right-skewed variance)."""
    from lightgbm import LGBMRegressor
    set_determinism()
    model = LGBMRegressor(objective="gamma", **lgbm_params(hp, seed, iterations))
    model.fit(x, y)
    return [model]


def fit_candidate(design, x, y, hp, seed, iterations):
    if design.target == "volatility":
        return fit_variance(x, y, hp, seed, iterations)
    return fit_quantiles(x, y, hp, seed, iterations)


def predict_log_returns(models, x, design, dates, iterations):
    """Median h-session log return: middle of the sorted quantile predictions x sigma x sqrt(h)."""
    values = np.column_stack([m.predict(x.loc[dates], num_iteration=iterations) for m in models])
    median = np.sort(values, axis=1)[:, QUANTILES.index(0.5)]
    return median * design.sigma.loc[dates].to_numpy() * np.sqrt(design.horizon)


def log_ratio(predicted, level):
    """Log of a predicted variance ratio relative to the benchmark level."""
    return np.log(np.clip(np.asarray(predicted, dtype=float), RATIO_FLOOR, None) / level)


def model_gain(models, x, design, dates, iterations, level):
    """A candidate's forecast as a log deviation from the benchmark ("gain"):
    price -> predicted log return; volatility -> log(model variance / benchmark variance)."""
    if design.target == "volatility":
        return log_ratio(models[0].predict(x.loc[dates], num_iteration=iterations), level)
    return predict_log_returns(models, x, design, dates, iterations)


def prices_from_returns(log_returns, design, dates, shrinkage=1.0):
    """shrinkage in [0, 1] scales the predicted return toward the naive forecast (0)."""
    prices = design.close.loc[dates].to_numpy() * np.exp(shrinkage * np.asarray(log_returns, dtype=float))
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Invalid predicted adjusted price")
    return prices


def reference_level(design, fit_dates):
    """The only fitted part of a benchmark: the mean variance ratio of the fit window.
    The price benchmark (last close) has nothing to fit."""
    if design.target != "volatility":
        return None
    return float(np.mean(design.y.loc[fit_dates]))


def benchmark_values(design, level, dates):
    """No-model forecast: last close, or EWMA variance x fit-window mean ratio x h."""
    if design.target == "volatility":
        return level * design.horizon * design.sigma.loc[dates].to_numpy() ** 2
    return design.close.loc[dates].to_numpy()


def values_from_gain(design, level, dates, gain, shrinkage=1.0):
    """forecast = benchmark x exp(shrinkage x gain); shrinkage 0 is the benchmark itself."""
    if design.target != "volatility":
        return prices_from_returns(gain, design, dates, shrinkage)
    values = benchmark_values(design, level, dates) * np.exp(shrinkage * np.asarray(gain, dtype=float))
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Invalid predicted variance")
    return values


def constant_log_returns(design, fit_dates, dates):
    """Zero-tree limit of the quantile booster: the training median of the target."""
    return (float(np.median(design.y.loc[fit_dates])) * design.sigma.loc[dates].to_numpy()
            * np.sqrt(design.horizon))


def har_matrix(design, dates):
    """[1, log mean variance over the last 1 / 5 / 22 sessions] - information up to each date."""
    columns = [np.log(design.variance.rolling(w).mean()).loc[dates].to_numpy() for w in HAR_WINDOWS]
    matrix = np.column_stack([np.ones(len(dates))] + columns)
    if not np.isfinite(matrix).all():
        raise ValueError("HAR regressors unavailable for the requested dates")
    return matrix


def fit_har(design, fit_dates):
    """Log-HAR by least squares on the fit window; Duan's smearing factor turns the
    log forecast back into a mean without assuming normal residuals."""
    regressors = har_matrix(design, fit_dates)
    target = np.log(design.realized.loc[fit_dates].to_numpy() / design.horizon)
    beta, *_ = np.linalg.lstsq(regressors, target, rcond=None)
    return beta, float(np.mean(np.exp(target - regressors @ beta)))


def har_variance(design, fit_dates, dates):
    beta, smear = fit_har(design, fit_dates)
    return np.exp(har_matrix(design, dates) @ beta) * smear * design.horizon


def har_gain(design, fit_dates, dates, level):
    return np.log(har_variance(design, fit_dates, dates) / benchmark_values(design, level, dates))


def combined_gain(design, gain, combine, fit_dates, dates, level):
    """combine="har": geometric mean of the tree model and HAR (mean of the log gains)."""
    if combine == "har":
        return 0.5 * (gain + har_gain(design, fit_dates, dates, level))
    return gain


def ape_pct(prices, actual):
    return np.abs(prices - actual) / actual * 100


def qlike_values(forecast, actual):
    return np.log(forecast) + actual / forecast


def loss_values(design, forecast, actual):
    return qlike_values(forecast, actual) if design.target == "volatility" else ape_pct(forecast, actual)


def finite_or_none(value):
    """Saved results are strict JSON (no NaN): an undefined statistic becomes null."""
    value = float(value)
    return value if np.isfinite(value) else None


def actual_prices(design, dates):
    return design.close.loc[pd.DatetimeIndex(design.target_dates.loc[dates])].to_numpy()


def actual_values(design, dates):
    if design.target == "volatility":
        return design.realized.loc[dates].to_numpy()
    return actual_prices(design, dates)


def strip_fixed(raw_hp):
    """The agreed 80% sampling ratios and the iteration grid override any proposal."""
    return {k: v for k, v in raw_hp.items()
            if k not in ("subsample", "colsample_bytree", "subsample_freq", "n_estimators")}


def nested_hpo(x, design, train_dates):
    """V8.5 random search scored ONLY inside the training window.

    Purged K-fold (each contiguous block is held out once, purge+embargo on both
    sides); the score is the out-of-fold loss of the point model (price: APE of the
    median model; volatility: QLIKE of the gamma model), i.e. the same loss the gate
    uses. The validation window is never read here, so the search cannot fit
    validation noise. PBO (CSCV) over the trial x iteration loss matrix is the
    guard: above PBO_LIMIT the winner is discarded.
    """
    from quantimate.models.hpo import sample_config
    from quantimate.validation.validation_cpcv import cpcv_splits, pbo_cscv
    rng = np.random.default_rng(SELECTION_SEED)
    configs = [{}] + [strip_fixed(sample_config(rng)) for _ in range(HPO_TRIALS - 1)]
    features = x.loc[train_dates]
    scale = design.sigma.loc[train_dates].to_numpy() * np.sqrt(design.horizon)
    actual = actual_values(design, train_dates)
    folds = list(cpcv_splits(len(train_dates), n_groups=HPO_FOLDS, k_test=1, purge=PURGE, embargo=EMBARGO))
    grid = ITERATION_GRID[design.target]
    loss = np.full((len(train_dates), len(configs), len(grid)), np.nan)
    naive = np.full(len(train_dates), np.nan)
    for fit_pos, held_pos in folds:
        level = reference_level(design, train_dates[fit_pos])
        naive[held_pos] = loss_values(design, benchmark_values(design, level, train_dates[held_pos]),
                                      actual[held_pos])
    for c, hp in enumerate(configs):
        for fit_pos, held_pos in folds:
            fit_dates, held = train_dates[fit_pos], train_dates[held_pos]
            level = reference_level(design, fit_dates)
            if design.target == "volatility":
                model = fit_variance(features.iloc[fit_pos], design.y.loc[fit_dates], hp,
                                     SELECTION_SEED, max(grid))[0]
            else:
                model = fit_quantiles(features.iloc[fit_pos], design.y.loc[fit_dates],
                                      hp, SELECTION_SEED, max(grid), quantiles=(0.5,))[0]
            for i, iterations in enumerate(grid):
                predicted = model.predict(features.iloc[held_pos], num_iteration=iterations)
                gain = (log_ratio(predicted, level) if design.target == "volatility"
                        else predicted * scale[held_pos])
                loss[held_pos, c, i] = loss_values(design, values_from_gain(design, level, held, gain),
                                                   actual[held_pos])
            del model
    if np.isnan(loss).any() or np.isnan(naive).any():
        raise ValueError("Out-of-fold coverage hole in nested search")
    scores = loss.mean(axis=0)                       # (config, iteration)
    best = int(np.argmin(scores.min(axis=1)))
    flat = loss.reshape(len(train_dates), -1)
    pbo = float(pbo_cscv(-flat, n_blocks=PBO_BLOCKS)["pbo"])
    # Does ANY trial beat the benchmark out-of-fold? Adjusted for the whole
    # search, so it can serve as the winner's replication evidence.
    versus_naive = spa_gate(list(flat.T), naive, design.horizon)
    accepted = best != 0 and pbo <= PBO_LIMIT
    reason = ("default configuration already best out-of-fold" if best == 0 else
              "PBO above limit: search winner discarded" if pbo > PBO_LIMIT else
              "winner admitted to validation")
    return dict(accepted=accepted, reason=reason, pbo=pbo, pbo_limit=PBO_LIMIT,
                best_trial=best, best_hp=configs[best], trials=len(configs), folds=HPO_FOLDS,
                default_oof_score=float(scores[0].min()), best_oof_score=float(scores[best].min()),
                naive_oof_score=float(naive.mean()), oof_spa_pvalue=versus_naive["pvalue"],
                scope="training window only")


def spa_gate(losses, naive_loss, horizon=1):
    """Hansen (2005) SPA over all candidates against the benchmark. The bootstrap
    block grows with the horizon because h-session targets overlap."""
    from quantimate.validation.validation_cpcv import spa_test
    block = spa_block(horizon)
    result = spa_test(-np.column_stack(losses), -naive_loss, n_boot=SPA_BOOTSTRAPS,
                      avg_block=block, seed=SELECTION_SEED)
    return dict(test="Hansen SPA (consistent), stationary bootstrap", alpha=GATE_ALPHA,
                pvalue=float(result["pvalue"]), statistic=float(result["stat"]),
                candidates=len(losses), bootstraps=SPA_BOOTSTRAPS, average_block=block)


def replicate_in_training(snapshot, cutoff, version, design, train_dates, candidate, hpo=None,
                          alpha=GATE_ALPHA):
    """Second gate stage: does the validation winner also beat the benchmark on
    an INDEPENDENT sample, out-of-fold inside the training window?

    Why: on validation alone a model can be significantly better for reasons
    tied to that one period (relative forecast performance is unstable over
    time, Giacomini & Rossi 2010). A real edge must replicate. Each contiguous
    block of the training window is held out once; the model (with its HMM, HAR
    coefficients and benchmark level) is refitted on the other blocks with
    purge+embargo on both sides. The p-value is the one-sided stationary-bootstrap
    test of "no better than the benchmark" for this single, pre-chosen candidate,
    so no search adjustment is needed. The exception is the V8.5 search winner,
    whose configuration was itself picked on these folds: it inherits the
    search-adjusted p-value of nested_hpo.
    """
    from quantimate.validation.validation_cpcv import cpcv_splits
    arm, combine = candidate["arm"], candidate.get("combine", "none")
    naive_parts, loss_parts, fold_gain = [], [], []
    for fit_pos, held_pos in cpcv_splits(len(train_dates), n_groups=REPLICATION_FOLDS, k_test=1,
                                         purge=PURGE, embargo=EMBARGO):
        fit_dates, held = train_dates[fit_pos], train_dates[held_pos]
        level = reference_level(design, fit_dates)
        if arm == "constant":
            gain = constant_log_returns(design, fit_dates, held)
        elif arm == "har":
            gain = har_gain(design, fit_dates, held, level)
        else:
            arms, _ = build_arms(snapshot, cutoff, version, design, fit_dates)
            # Optional columns can be unavailable on a shorter fold; LightGBM handles the NaN.
            x = arms.get(arm, arms["baseline"]).reindex(columns=candidate["features"])
            train_x, train_y, _ = training_data(arm, x, design, fit_dates, snapshot, cutoff)
            models = fit_candidate(design, train_x, train_y, candidate["hp"], SELECTION_SEED,
                                   candidate["iterations"])
            gain = model_gain(models, x, design, held, candidate["iterations"], level)
            del models
            gain = combined_gain(design, gain, combine, fit_dates, held, level)
        actual = actual_values(design, held)
        naive = loss_values(design, benchmark_values(design, level, held), actual)
        loss = loss_values(design, values_from_gain(design, level, held, gain, candidate["shrinkage"]), actual)
        naive_parts.append(naive)
        loss_parts.append(loss)
        fold_gain.append(float((naive - loss).mean()))
    naive, loss = np.concatenate(naive_parts), np.concatenate(loss_parts)
    searched = candidate["config"] == "hpo"
    pvalue = (float(hpo["oof_spa_pvalue"]) if searched
              else spa_gate([loss], naive, design.horizon)["pvalue"])
    gain = float((naive - loss).mean())
    result = dict(sample="purged K-fold out-of-fold, inside the training window", folds=REPLICATION_FOLDS,
                  days=len(naive), gain=gain, fold_gain=fold_gain,
                  pvalue=pvalue, alpha=alpha, search_adjusted=searched,
                  replicated=bool(pvalue < alpha and loss.mean() < naive.mean()))
    if design.target == "price":
        result.update(gain_pct=gain, fold_gain_pct=fold_gain)   # percentage points of price APE
    return result


NAIVE = dict(arm="naive", config="none", hp={}, iterations=0, shrinkage=0.0, combine="none", features=[])


def benchmark_choice(target):
    """The selection outcome when the gate does not pass: the benchmark, nothing fitted."""
    return dict(NAIVE, arm=BENCHMARK_ARM[target])


def select(snapshot, cutoff, version, design, splits):
    from quantimate.validation.evaluation import diebold_mariano
    labeled = design.labeled
    train_dates, val_dates = labeled[splits.train], labeled[splits.val]
    volatility = design.target == "volatility"
    metric = VALIDATION_METRIC[design.target]
    grid = ITERATION_GRID[design.target]
    arms, notes = build_arms(snapshot, cutoff, version, design, train_dates)
    level = reference_level(design, train_dates)
    actual = actual_values(design, val_dates)
    naive_loss = loss_values(design, benchmark_values(design, level, val_dates), actual)
    log, losses = [], []

    def consider(gain, **candidate):
        for shrinkage in SHRINKAGE:
            loss = loss_values(design, values_from_gain(design, level, val_dates, gain, shrinkage), actual)
            mean = float(loss.mean())
            log.append(dict(candidate, shrinkage=shrinkage, validation_loss=mean, **{metric: mean}))
            losses.append(loss)

    simple = dict(config="none", hp={}, iterations=0, combine="none", features=[],
                  training_rows={"NVDA": len(train_dates)})
    if volatility:
        har = har_gain(design, train_dates, val_dates, level)
        consider(har, arm="har", **simple)
    else:
        consider(constant_log_returns(design, train_dates, val_dates), arm="constant", **simple)
    hpo = None
    for arm, x in arms.items():
        configs = list(BASE_CONFIGS)
        if version == "V8.5" and arm == "baseline":
            hpo = nested_hpo(x, design, train_dates)
            if hpo["accepted"]:
                configs.append(("hpo", hpo["best_hp"]))
        train_x, train_y, counts = training_data(arm, x, design, train_dates, snapshot, cutoff)
        for config, raw_hp in configs:
            hp = strip_fixed(raw_hp)
            models = fit_candidate(design, train_x, train_y, hp, SELECTION_SEED, max(grid))
            for iterations in grid:
                gain = model_gain(models, x, design, val_dates, iterations, level)
                for combine in (VOLATILITY_COMBINE if volatility else ("none",)):
                    consider(gain if combine == "none" else 0.5 * (gain + har), arm=arm,
                             config=config, hp=hp, iterations=iterations, combine=combine,
                             features=list(x.columns), training_rows=counts)
            del models
    naive_mean = float(naive_loss.mean())
    best = int(np.argmin([row["validation_loss"] for row in log]))
    gate = spa_gate(losses, naive_loss, design.horizon)
    dm = diebold_mariano(losses[best], naive_loss, h=design.horizon)
    gate.update(benchmark_validation_loss=naive_mean, loss_metric=LOSS_METRIC[design.target],
                best_candidate={k: log[best][k] for k in
                                ("arm", "config", "iterations", "shrinkage", "combine", "validation_loss", metric)},
                best_vs_naive_dm=dict(statistic=finite_or_none(dm["stat"]),
                                      pvalue=finite_or_none(dm["pvalue"]),
                                      note="pairwise, NOT adjusted for the search; information only"))
    if not volatility:
        gate["naive_validation_mape_pct"] = naive_mean
    # Stage 1: the joint test on validation. Stage 2 (only if stage 1 rejects):
    # the winner must replicate on an independent sample. The benchmark stays
    # otherwise; keeping it is a documented, honest outcome, not a failed run.
    gate["validation_passed"] = bool(gate["pvalue"] < GATE_ALPHA
                                     and log[best]["validation_loss"] < naive_mean)
    # Volatility: a pre-registered second look. If the validation winner does not
    # replicate, the plain HAR model (full weight) is tested the same way. Two
    # possible looks share the level, so each is tested at alpha / 2.
    fallback = (next(i for i, row in enumerate(log)
                     if row["arm"] == VOLATILITY_FALLBACK and row["shrinkage"] == 1.0)
                if volatility else None)
    stage_two = GATE_ALPHA / 2 if volatility else GATE_ALPHA
    gate.update(replication=None, fallback=None, stage_two_alpha=stage_two)
    chosen = None
    if gate["validation_passed"]:
        gate["replication"] = replicate_in_training(snapshot, cutoff, version, design, train_dates,
                                                    log[best], hpo, stage_two)
        if gate["replication"]["replicated"]:
            chosen = best
        elif fallback is not None and fallback != best and log[fallback]["validation_loss"] < naive_mean:
            replication = replicate_in_training(snapshot, cutoff, version, design, train_dates,
                                                log[fallback], None, stage_two)
            gate["fallback"] = dict(candidate={k: log[fallback][k] for k in
                                               ("arm", "shrinkage", "validation_loss")},
                                    replication=replication)
            if replication["replicated"]:
                chosen = fallback
    gate["passed"] = chosen is not None
    selected = dict(log[chosen]) if gate["passed"] else dict(
        benchmark_choice(design.target), validation_loss=naive_mean, training_rows={}, **{metric: naive_mean})
    for row in log:
        del row["features"]          # kept once, on the selected candidate
    result = dict(selected=selected, gate=gate, decision_log=log, decision_count=len(log), notes=notes,
                  split_info=split_info(design, splits), selection_seed=SELECTION_SEED)
    if hpo is not None:
        result["hpo"] = hpo
    return result


def split_info(design, splits):
    info = dict(window_start=str(design.x.index[0].date()),
                window_end=str(design.x.index[-1].date()),
                nominal_fractions=[0.70, 0.15, 0.15], purge=PURGE, embargo=EMBARGO,
                target=design.target, horizon=design.horizon)
    for name in ("train", "val", "test"):
        dates = design.labeled[getattr(splits, name)]
        info[name] = dict(n=len(dates), first=str(dates[0].date()), last=str(dates[-1].date()),
                          last_target=str(design.target_dates.loc[dates[-1]].date()))
    return info


def price_bands(design, train_dates, calibration_dates, dates):
    """90% and 50% price bands for `dates`, uncalibrated ("raw") and conformally calibrated.

    Base band: the training quantiles of the normalized target, scaled by the origin
    volatility - the benchmark's own predictive distribution, no model. Calibration
    (only at the horizons in CONFORMAL_HORIZONS; elsewhere the band IS the reference
    band, see the protocol for the validation evidence): adaptive conformal inference
    over the stream calibration_dates -> dates. A row's outcome enters the correction
    only once it is observed, h sessions after its origin, so every band uses past
    information only. `dates` may end with an origin whose target is not yet known
    (the external forecast).
    """
    stream = calibration_dates.append(dates)
    positions = design.close.index.get_indexer(stream)
    if (positions < 0).any():
        raise ValueError("Band dates outside the session calendar")
    y = design.y.reindex(stream).to_numpy(dtype=float)
    close = design.close.loc[dates].to_numpy()
    scale = design.sigma.loc[dates].to_numpy() * np.sqrt(design.horizon)
    to_price = lambda normalized: close * np.exp(normalized * scale)
    train_y, out, part = design.y.loc[train_dates].to_numpy(), {}, slice(len(calibration_dates), None)
    for name, low, high in BAND_LEVELS:
        q_low, q_high = np.quantile(train_y, [low, high])
        raw_lower, raw_upper = to_price(np.full(len(dates), q_low)), to_price(np.full(len(dates), q_high))
        if design.horizon in CONFORMAL_HORIZONS:
            lower, upper, _ = aci_bands_delayed(y, np.full(len(stream), q_low), np.full(len(stream), q_high),
                                                positions, len(calibration_dates), 1 - (high - low),
                                                design.horizon, gamma=ACI_GAMMA)
            lower, upper = to_price(lower[part]), to_price(upper[part])
        else:
            lower, upper = raw_lower.copy(), raw_upper.copy()
        out[name] = dict(lower=lower, upper=upper, raw_lower=raw_lower, raw_upper=raw_upper)
    return out


def band_fields(bands, position, actual):
    """One row's band columns. The raw bounds are stored for the 90% band only."""
    fields = {}
    for name, band in bands.items():
        lower, upper = float(band["lower"][position]), float(band["upper"][position])
        raw_lower, raw_upper = float(band["raw_lower"][position]), float(band["raw_upper"][position])
        fields.update({f"lower_{name}": lower, f"upper_{name}": upper,
                       f"inside_{name}": int(lower <= actual <= upper),
                       f"raw_inside_{name}": int(raw_lower <= actual <= raw_upper)})
        if name == BAND_LEVELS[0][0]:
            fields.update({f"raw_lower_{name}": raw_lower, f"raw_upper_{name}": raw_upper})
    return fields


def band_summary(rows):
    """Coverage of both bands; width and the Winkler interval score of the 90% band in
    percent of the origin price (lower is better: width plus a penalty for misses)."""
    name, low, high = BAND_LEVELS[0]
    alpha = 1 - (high - low)
    origin = np.array([r["origin_adj_close"] for r in rows])
    actual = np.array([r["actual_adj_close"] for r in rows])
    summary = {}
    for level, _, _ in BAND_LEVELS:
        summary[f"test_coverage_{level}"] = float(np.mean([r[f"inside_{level}"] for r in rows]))
        summary[f"test_raw_coverage_{level}"] = float(np.mean([r[f"raw_inside_{level}"] for r in rows]))
    for prefix, label in (("", ""), ("raw_", "raw_")):
        lower = np.array([r[f"{prefix}lower_{name}"] for r in rows])
        upper = np.array([r[f"{prefix}upper_{name}"] for r in rows])
        miss = np.maximum(lower - actual, 0) + np.maximum(actual - upper, 0)
        summary[f"test_{label}width_{name}_pct"] = float(np.mean((upper - lower) / origin) * 100)
        summary[f"test_{label}interval_score_{name}_pct"] = float(
            np.mean((upper - lower + 2 / alpha * miss) / origin) * 100)
    return summary


def evaluation_rows(design, values, dates, level, bands=None):
    actual = actual_values(design, dates)
    reference = benchmark_values(design, level, dates)
    rows = []
    for i, (date, prediction, truth) in enumerate(zip(dates, values, actual)):
        origin = float(design.close.loc[date])
        row = dict(origin_date=str(date.date()), target_date=str(design.target_dates.loc[date].date()),
                   origin_adj_close=origin)
        if design.target == "volatility":
            row.update(variance_metrics(float(prediction), float(truth), float(reference[i])))
        else:
            row.update(predicted_adj_close=float(prediction), actual_adj_close=float(truth),
                       **price_metrics(float(prediction), float(truth), origin))
            if bands is not None:
                row.update(band_fields(bands, i, float(truth)))
        rows.append(row)
    return rows


def external_actual(snapshot, design, cutoff, forecast):
    """The observed target of the external forecast; read only after the forecast exists."""
    frame = snapshot.daily("NVDA")
    if design.target == "price":
        return float(frame.loc[pd.Timestamp(forecast), "NVDA_Close"])
    start = frame.index.get_loc(pd.Timestamp(cutoff)) + 1
    window = daily_range_variance(frame, "NVDA").iloc[start:start + design.horizon]
    if len(window) != design.horizon or window.index[-1] != pd.Timestamp(forecast):
        raise ValueError("External realized variance window does not match the forecast date")
    return float(window.sum())


def repeat(snapshot, cutoff, forecast, version, seed, design, splits, selection):
    chosen = selection["selected"]
    labeled = design.labeled
    train_dates, val_dates, test_dates = labeled[splits.train], labeled[splits.val], labeled[splits.test]
    arm, iterations, hp = chosen["arm"], chosen["iterations"], chosen["hp"]
    shrinkage, combine = float(chosen.get("shrinkage", 1.0)), chosen.get("combine", "none")
    volatility = design.target == "volatility"
    cutoff_date = pd.DatetimeIndex([cutoff])
    # Independent final refit: no weight reuse; cutoff row has no mature target.
    refit_dates = labeled
    if (design.target_dates.loc[refit_dates] > pd.Timestamp(cutoff)).any():
        raise ValueError("Post-cutoff labels in final refit")
    level, refit_level = reference_level(design, train_dates), reference_level(design, refit_dates)
    if arm in BENCHMARK_ARMS:
        # Gate not passed: the benchmark itself is the forecast; no model is fitted.
        test_gain, gain, counts = np.zeros(len(test_dates)), np.zeros(1), {}
    elif arm == "constant":
        test_gain = constant_log_returns(design, train_dates, test_dates)
        gain = constant_log_returns(design, refit_dates, cutoff_date)
        counts = {"NVDA": len(refit_dates)}
    elif arm == "har":
        test_gain = har_gain(design, train_dates, test_dates, level)
        gain = har_gain(design, refit_dates, cutoff_date, refit_level)
        counts = {"NVDA": len(refit_dates)}
    else:
        arms, _ = build_arms(snapshot, cutoff, version, design, train_dates)
        x = arms[arm].loc[:, chosen["features"]]
        train_x, train_y, _ = training_data(arm, x, design, train_dates, snapshot, cutoff)
        model = fit_candidate(design, train_x, train_y, hp, seed, iterations)
        test_gain = combined_gain(design, model_gain(model, x, design, test_dates, iterations, level),
                                  combine, train_dates, test_dates, level)
        del model
        if arm in ("generic_nvda", "pooled"):
            refit_x = x
        else:
            refit_base = add_hmm(design, refit_dates)
            refit_x = x.copy()
            for col in ("regime_p0", "regime_p1", "regime_p2"):
                refit_x[col] = refit_base[col]
        train_x, train_y, counts = training_data(arm, refit_x, design, refit_dates, snapshot, cutoff)
        model = fit_candidate(design, train_x, train_y, hp, seed, iterations)
        gain = combined_gain(design, model_gain(model, refit_x, design, cutoff_date, iterations, refit_level),
                             combine, refit_dates, cutoff_date, refit_level)
        del model
    test_values = values_from_gain(design, level, test_dates, test_gain, shrinkage)
    prediction = float(values_from_gain(design, refit_level, cutoff_date, gain, shrinkage)[0])
    reference = float(benchmark_values(design, refit_level, cutoff_date)[0])
    # Bands: one causal stream, validation -> test -> cutoff (its outcome is still unknown).
    bands = None if volatility else price_bands(design, train_dates, val_dates, test_dates.append(cutoff_date))
    test = evaluation_rows(design, test_values, test_dates, level, bands)
    summary = dict(loss_metric=LOSS_METRIC[design.target],
                   test_loss=float(np.mean([r["loss"] for r in test])),
                   test_reference_loss=float(np.mean([r["reference_loss"] for r in test])),
                   test_reference_win_rate=float(np.mean([r["beats_reference"] for r in test])))
    if volatility:
        rmse = lambda key: float(np.sqrt(np.mean([(r[key] - r["actual_value"]) ** 2 for r in test])))
        har_test = har_variance(design, train_dates, test_dates)
        summary.update(test_rmse_vol_pct=rmse("predicted_value"),
                       test_reference_rmse_vol_pct=rmse("reference_value"),
                       # Information: the standard econometric benchmark, fitted on train only.
                       test_har_loss=float(qlike_values(har_test, actual_values(design, test_dates)).mean()))
    else:
        summary.update(test_mape_pct=float(np.mean([r["ape_pct"] for r in test])),
                       test_mae_usd=float(np.mean([r["absolute_error_usd"] for r in test])),
                       test_direction_hit_rate=float(np.mean([r["direction_hit"] for r in test])),
                       **band_summary(test))
    # The external observed target is opened only after the forecast has been produced.
    actual = external_actual(snapshot, design, cutoff, forecast)
    origin = float(design.close.loc[pd.Timestamp(cutoff)])
    point = dict(origin_date=cutoff, target_date=forecast, origin_adj_close=origin)
    if volatility:
        point.update(variance_metrics(prediction, actual, reference))
    else:
        point.update(predicted_adj_close=prediction, actual_adj_close=actual,
                     **price_metrics(prediction, actual, origin),
                     **band_fields(bands, len(test_dates), actual))
    return dict(arm=arm, config=chosen.get("config"), iterations=iterations, shrinkage=shrinkage,
                combine=combine, gate_pvalue=selection.get("gate", {}).get("pvalue"),
                target=design.target, horizon=design.horizon, forecast=point, historical_test=test,
                test_summary=summary, refit_rows=counts, split_info=split_info(design, splits))


def execute_job(job_path, notebook_version):
    """Called only by an explicitly executed experiment notebook in a fresh kernel."""
    from quantimate.experiment_runner import code_fingerprint, environment_fingerprint
    started = time.perf_counter()
    job = read_json(job_path)
    manifest = read_json(Path(job["output"]) / "experiment.json")
    if job["experiment_id"] != manifest["experiment_id"]:
        raise ValueError("Wrong experiment identity")
    if code_fingerprint(Path(__file__).resolve().parents[1]) != manifest["code"]:
        raise ValueError("Source changed during experiment")
    if environment_fingerprint() != manifest["environment"]:
        raise ValueError("Runtime dependencies changed during experiment")
    version, pair_id = job["version"], job["pair_id"]
    target, horizon = job["target"], job["horizon"]
    if version not in VERSIONS or version != notebook_version or not 1 <= pair_id <= len(DATE_PAIRS):
        raise ValueError("Job does not match notebook / protocol")
    if (target, horizon) not in TRACKS:
        raise ValueError("Unknown track")
    if job["phase"] not in ("select", "repeat"):
        raise ValueError("Unknown phase")
    if job["phase"] == "repeat" and job["seed"] not in SEEDS:
        raise ValueError("Invalid repetition seed")
    cutoff, next_day = DATE_PAIRS[pair_id - 1]
    snapshot = Snapshot(job["snapshot"])
    if snapshot.manifest["snapshot_id"] != manifest["snapshot_id"]:
        raise ValueError("Wrong snapshot")
    # Calendar lookup only: the session h sessions after the cutoff. No price is read.
    forecast = snapshot.target_date(cutoff, horizon)
    if horizon == 1 and forecast != next_day:
        raise ValueError("Protocol date pair is not the next observed session")
    design = make_design(snapshot.universe(cutoff), cutoff, target=target, horizon=horizon)
    splits = make_purged_splits(len(design.labeled), window=1, train_frac=0.70, val_frac=0.15,
                                purge=PURGE, embargo=EMBARGO)
    if min(len(splits.train), len(splits.val), len(splits.test)) < 30:
        raise ValueError("Insufficient data for purged 70/15/15 split")
    if job["phase"] == "select":
        result = select(snapshot, cutoff, version, design, splits)
    else:
        selection_path = Path(job["selection_path"])
        selection_key = job["job_key"].rsplit("/", 1)[0] + "/selection"
        if not result_valid(selection_path, job["experiment_id"], selection_key):
            raise ValueError("Missing selection")
        selection = read_json(selection_path)
        if selection["result_hash"] != job["selection_hash"]:
            raise ValueError("Frozen selection changed")
        result = repeat(snapshot, cutoff, forecast, version, job["seed"], design, splits, selection)
        result["selection_hash"] = selection["result_hash"]
    result.update(status="success", experiment_id=job["experiment_id"], job_key=job["job_key"],
                  version=version, pair_id=pair_id, target=target, horizon=horizon,
                  cutoff_date=cutoff, forecast_date=forecast,
                  seed=job["seed"], snapshot_id=manifest["snapshot_id"],
                  duration_seconds=time.perf_counter() - started)
    save_result(job["result_path"], result)
    if job["phase"] == "repeat":
        export_individual(result, job["output"])
    return result
