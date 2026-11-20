"""Frozen protocol and durable result formats; importing never trains anything."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

PROTOCOL_VERSION = 3
RAW_START = "2001-02-01"
VERSIONS = ("V8", "V8.5")
SEEDS = tuple(range(1, 51))
# Schema 3: a track is one (target, horizon) study; every track repeats the full
# 30 date pairs x 50 seeds x 2 versions design.
TARGETS = ("price", "volatility")
HORIZONS = (1, 5, 21)
TRACKS = tuple((target, horizon) for target in TARGETS for horizon in HORIZONS)
PURGE, EMBARGO = 21, 5
# The "no model" outcome of the gate, per target.
BENCHMARK_ARM = {"price": "naive", "volatility": "ewma"}
BENCHMARK_ARMS = tuple(BENCHMARK_ARM.values())
SELECTION_SEED = 42
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
# Schema 2 (anti-overfitting selection). Every value below is fixed a priori,
# from the cited literature / repo precedent, never tuned on a test window.
# The grid reaches down to very small ensembles: in schema 1 the smallest
# allowed size (50) won every single date pair, i.e. the optimum was off-grid.
ITERATIONS = (5, 10, 20, 50, 100, 200, 400)
# Volatility tracks search a longer grid (2026-11-15). In the schema-3 verification
# (V8, 30 pairs) the low-capacity configuration won with the LARGEST allowed size
# (400) in 21 of 30 pairs at h=1, in 12 at h=5 and in 7 at h=21: the mirror image
# of the schema-1 symptom above, the optimum may lie beyond the grid (at a
# learning rate of 0.02, 400 trees are few). The evidence is the position of the
# VALIDATION winner; no test loss was used. The extension is not verified by a
# run yet. Price tracks keep the original grid: no price candidate replicated at
# any size, and price_h01 stays identical to schema 2.
VOLATILITY_ITERATIONS = ITERATIONS + (800, 1600)
ITERATION_GRID = {"price": ITERATIONS, "volatility": VOLATILITY_ITERATIONS}
# Shrinkage of the predicted return toward the benchmark (zero return = naive
# last price), cf. Campbell & Thompson (2008): forecasts restricted toward the
# benchmark survive out of sample where unrestricted ones do not.
SHRINKAGE = (0.25, 0.5, 1.0)
# Low-capacity alternative to the legacy defaults (31 leaves, min 20 rows per
# leaf on ~1700 rows). LightGBM tuning guide: on small data limit max_depth,
# keep num_leaves <= 2**max_depth and raise min_data_in_leaf. With the quantile
# objective leaf values are re-fitted as residual quantiles, so L1/L2 penalties
# barely bind; leaf size, depth and learning rate are the effective controls.
BASE_CONFIGS = (
    ("default", {}),
    ("regularized", dict(learning_rate=0.02, num_leaves=4, max_depth=2, min_child_samples=200)),
)
# A fitted model replaces the naive forecast only if Hansen's (2005) SPA test
# rejects "no candidate beats the benchmark" on validation. SPA accounts for
# having searched over all candidates at once (data snooping), which a pairwise
# Diebold-Mariano test of the validation winner would not. alpha follows the
# repo's existing gate precedent (p < 0.10).
GATE_ALPHA = 0.10
# Second gate stage: the validation winner must ALSO beat the benchmark on an
# independent sample, out-of-fold inside the training window (same alpha).
# Relative forecast performance is unstable over time (Giacomini & Rossi 2010):
# significance in one validation period does not imply a persistent edge.
REPLICATION_FOLDS = 4
SPA_BOOTSTRAPS = 2000
SPA_AVG_BLOCK = 10.0
# V8.5 hyper-parameter search: purged K-fold inside the training window only;
# the winner may enter validation only if PBO (Bailey et al. 2015) <= 0.5.
HPO_TRIALS = 30
HPO_FOLDS = 4
PBO_LIMIT = 0.5
PBO_BLOCKS = 16
# h-session targets overlap, so forecast losses are serially dependent over h-1
# sessions; the bootstrap block must be longer than that dependence.
SPA_BLOCK_PER_HORIZON = 2.0
# Volatility track. Benchmark: the EWMA variance known at the origin, rescaled by
# its training-window mean ratio to the realized variance ("no model"). Loss:
# QLIKE on the variance, which ranks forecasts correctly under a noisy realized
# proxy, unlike absolute-error losses (Patton 2011). Candidates: log-HAR
# (Corsi 2009), LightGBM with the gamma deviance (the same loss), and their
# geometric mean. On validation data alone (all 30 pairs) each of the three beat
# the benchmark in 24-28 pairs at h=1 and 22-29 pairs at h=5; the blend was best.
HAR_WINDOWS = (1, 5, 22)
VOLATILITY_COMBINE = ("none", "har")
# Pre-registered fallback of the second gate stage on volatility tracks: when the
# validation winner (picked among hundreds of candidates) does not replicate,
# the plain HAR model is tested the same way before falling back to the
# benchmark. HAR is the standard benchmark of the realized-volatility literature
# and has four coefficients. Two possible looks, so each is tested at alpha / 2.
VOLATILITY_FALLBACK = "har"
# Price bands: the training quantiles of the normalized target, scaled by the
# origin EWMA volatility ("reference band"). Model quantiles are NOT used: on
# validation alone they lost to the reference band in pinball loss (h=5: 29 of
# 30 pairs worse; h=21: 26-27 worse; h=1: no difference).
# Calibration: adaptive conformal inference (Gibbs & Candes 2021), feedback
# delayed by the horizon. Judged on validation halves only, by coverage AND by
# the interval score (a proper scoring rule, so a band cannot win by inflating):
#   h=1 : coverage error 0.026 -> 0.015 (90%) and 0.042 -> 0.026 (50%) at an
#         unchanged interval score -> calibration is applied;
#   h=5 : interval score 2-3% worse (90% band), better in only 3-6 of 30 pairs;
#   h=21: interval score 8-35% worse, better in only 2-3 of 30 pairs
#         -> the reference band is used as it is (about 18 and 9 independent
#         h-session outcomes in a calibration half are too few to estimate a
#         correction). Larger steps (0.02, 0.05) track coverage more closely but
#         widen the band at every horizon; 0.005 is also the step of the paper.
BAND_LEVELS = ((90, 0.05, 0.95), (50, 0.25, 0.75))
ACI_GAMMA = 0.005
CONFORMAL_HORIZONS = (1,)
DATE_PAIRS = (
    ("2021-01-29", "2021-02-01"), ("2021-03-09", "2021-03-10"),
    ("2021-05-12", "2021-05-13"), ("2021-07-12", "2021-07-13"),
    ("2021-09-09", "2021-09-10"), ("2021-11-09", "2021-11-10"),
    ("2022-01-28", "2022-01-31"), ("2022-03-08", "2022-03-09"),
    ("2022-05-11", "2022-05-12"), ("2022-07-11", "2022-07-12"),
    ("2022-09-08", "2022-09-09"), ("2022-11-08", "2022-11-09"),
    ("2023-01-27", "2023-01-30"), ("2023-03-07", "2023-03-08"),
    ("2023-05-10", "2023-05-11"), ("2023-07-17", "2023-07-18"),
    ("2023-09-07", "2023-09-08"), ("2023-11-07", "2023-11-08"),
    ("2024-01-26", "2024-01-29"), ("2024-03-05", "2024-03-06"),
    ("2024-05-08", "2024-05-09"), ("2024-07-15", "2024-07-16"),
    ("2024-09-12", "2024-09-13"), ("2024-11-05", "2024-11-06"),
    ("2025-01-31", "2025-02-03"), ("2025-03-11", "2025-03-12"),
    ("2025-05-07", "2025-05-08"), ("2025-07-14", "2025-07-15"),
    ("2025-09-11", "2025-09-12"), ("2025-11-11", "2025-11-12"),
)
RUNS_PER_TRACK = len(VERSIONS) * len(DATE_PAIRS) * len(SEEDS)
REQUIRED_RUNS = RUNS_PER_TRACK * len(TRACKS)
PROTOCOL = dict(schema=PROTOCOL_VERSION, raw_start=RAW_START,
                date_pairs=DATE_PAIRS, versions=VERSIONS, seeds=SEEDS,
                targets=TARGETS, horizons=HORIZONS,
                years=10, split=[0.70, 0.15, 0.15],
                purge=PURGE, embargo=EMBARGO, selection_seed=42,
                target_definition=dict(
                    price="h-session log return / (EWMA volatility at origin * sqrt(h))",
                    volatility="sum of the next h daily range-based variances"
                               " / (h * EWMA variance at origin)"),
                daily_variance="overnight log return squared + Rogers-Satchell",
                selection_metric=dict(price="validation_adjusted_close_MAPE",
                                      volatility="validation_realized_variance_QLIKE"),
                quantiles=QUANTILES, iteration_grid=ITERATION_GRID,
                shrinkage_grid=SHRINKAGE, base_configs=BASE_CONFIGS,
                benchmark=dict(price="naive_last_adjusted_close",
                               volatility="ewma_variance_times_training_mean_ratio"),
                volatility_candidates=dict(har_windows=HAR_WINDOWS, gbdt_objective="gamma",
                                           combine=VOLATILITY_COMBINE,
                                           second_stage_fallback=VOLATILITY_FALLBACK),
                price_bands=dict(levels=BAND_LEVELS, base="training_quantiles_of_normalized_target",
                                 calibration="adaptive_conformal_delayed_feedback",
                                 gamma=ACI_GAMMA, calibrated_horizons=CONFORMAL_HORIZONS),
                gate=dict(test="hansen_spa_consistent", alpha=GATE_ALPHA,
                          bootstraps=SPA_BOOTSTRAPS, average_block=SPA_AVG_BLOCK,
                          block_per_horizon=SPA_BLOCK_PER_HORIZON,
                          bootstrap_seed=SELECTION_SEED),
                replication=dict(sample="purged_kfold_inside_training_window",
                                 folds=REPLICATION_FOLDS, alpha=GATE_ALPHA,
                                 test="one_sided_stationary_bootstrap"),
                v85_hpo=dict(trials=HPO_TRIALS, purged_folds=HPO_FOLDS,
                             scope="training_window_only", pbo_limit=PBO_LIMIT,
                             pbo_blocks=PBO_BLOCKS),
                subsample=0.8,
                colsample_bytree=0.8, subsample_freq=1, threads=1,
                max_attempts_per_invocation=3)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, ensure_ascii=True, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def point_metrics(predicted, actual, origin):
    if not all(math.isfinite(x) and x > 0 for x in (predicted, actual, origin)):
        raise ValueError("Prices must be finite, positive adjusted closes")
    ae, baseline_ae = abs(predicted - actual), abs(origin - actual)
    sign = lambda x: (x > 0) - (x < 0)
    tie = math.isclose(ae, baseline_ae, rel_tol=1e-12, abs_tol=1e-12)
    return dict(absolute_error_usd=ae, ape_pct=100 * ae / actual,
                direction_hit=int(sign(predicted - origin) == sign(actual - origin)),
                baseline_predicted_adj_close=origin,
                baseline_ape_pct=100 * baseline_ae / actual,
                beats_baseline=int(ae < baseline_ae and not tie),
                ties_baseline=int(tie))


def price_metrics(predicted, actual, origin):
    """Price observation with the track-independent loss columns added."""
    m = point_metrics(predicted, actual, origin)
    return dict(m, value_unit="USD", predicted_value=predicted, actual_value=actual,
                reference_value=origin, loss=m["ape_pct"], reference_loss=m["baseline_ape_pct"],
                beats_reference=m["beats_baseline"], ties_reference=m["ties_baseline"])


def qlike(forecast, actual):
    """QLIKE loss of a variance forecast (Patton 2011): log f + actual / f."""
    return math.log(forecast) + actual / forecast


def variance_metrics(predicted, actual, reference):
    """Variance observation. Values are shown as h-session volatility in percent."""
    if not all(math.isfinite(x) and x > 0 for x in (predicted, actual, reference)):
        raise ValueError("Variances must be finite and positive")
    loss, reference_loss = qlike(predicted, actual), qlike(reference, actual)
    tie = math.isclose(loss, reference_loss, rel_tol=1e-12, abs_tol=1e-12)
    return dict(value_unit="vol_pct", predicted_value=100 * math.sqrt(predicted),
                actual_value=100 * math.sqrt(actual), reference_value=100 * math.sqrt(reference),
                predicted_variance=predicted, actual_variance=actual, reference_variance=reference,
                loss=loss, reference_loss=reference_loss,
                beats_reference=int(loss < reference_loss and not tie), ties_reference=int(tie))


def spa_block(horizon):
    """Average stationary-bootstrap block for an h-session target."""
    return max(SPA_AVG_BLOCK, SPA_BLOCK_PER_HORIZON * horizon)


def track_name(target, horizon):
    if (target, horizon) not in TRACKS:
        raise ValueError(f"Unknown track: {target} / {horizon}")
    return f"{target}_h{horizon:02d}"


def external_phase(horizon):
    return "external_next_day" if horizon == 1 else f"external_{horizon}_sessions"


def job_key(version, pair_id, seed=None, target="price", horizon=1):
    leaf = "selection" if seed is None else f"seed_{seed:02d}"
    return f"{version.replace('.', '_')}/{track_name(target, horizon)}/pair_{pair_id:02d}/{leaf}"


def result_valid(path, experiment_id, key):
    path = Path(path)
    if not path.exists():
        return False
    obj = read_json(path)
    expected = obj.get("result_hash")
    body = {k: v for k, v in obj.items() if k != "result_hash"}
    if (obj.get("status") != "success" or obj.get("experiment_id") != experiment_id
            or obj.get("job_key") != key or digest(body) != expected):
        raise ValueError(f"Conflicting or damaged saved result: {path}")
    return True


def save_result(path, body):
    atomic_json(path, {**body, "result_hash": digest(body)})


CSV_FIELDS = ["record_type", "phase", "experiment_id", "version", "pair_id",
              "cutoff_date", "forecast_date", "run_id", "seed", "ticker",
              "origin_date", "target_date", "origin_adj_close",
              "predicted_adj_close", "actual_adj_close", "absolute_error_usd",
              "ape_pct", "direction_hit", "baseline_predicted_adj_close",
              "baseline_ape_pct", "beats_baseline", "ties_baseline",
              "n_success", "n_required", "mape_mean_pct", "mape_std_pct",
              "mae_mean_usd", "mae_std_usd", "direction_hit_rate",
              "baseline_win_rate", "baseline_tie_rate", "prediction_std_usd",
              "test_mape_pct", "test_mae_usd", "test_direction_hit_rate",
              "selected_arm", "selected_config", "selected_iterations",
              "selected_shrinkage", "gate_pvalue", "selection_hash",
              "snapshot_id", "duration_seconds", "status",
              # Schema 3: track identity and target-independent columns.
              "target", "horizon", "selected_combine", "value_unit", "predicted_value",
              "actual_value", "reference_value", "loss", "reference_loss",
              "beats_reference", "ties_reference", "predicted_variance", "actual_variance",
              "reference_variance", "loss_mean", "loss_std", "reference_win_rate",
              # Calibrated price band (conformal) next to the uncalibrated reference band.
              "lower_90", "upper_90", "inside_90", "lower_50", "upper_50", "inside_50",
              "raw_lower_90", "raw_upper_90", "raw_inside_90", "raw_inside_50", "inside_90_rate",
              "loss_metric", "test_loss", "test_reference_loss", "test_reference_win_rate",
              "test_coverage_90", "test_raw_coverage_90", "test_coverage_50",
              "test_raw_coverage_50", "test_width_90_pct", "test_raw_width_90_pct",
              "test_interval_score_90_pct", "test_raw_interval_score_90_pct",
              "test_rmse_vol_pct", "test_reference_rmse_vol_pct", "test_har_loss"]
TRACK_FIELDS = ["version", "target", "horizon", "loss_metric", "pairs", "runs",
                "pairs_model_selected", "test_loss", "test_reference_loss", "test_gain",
                "pairs_test_better", "test_har_loss", "external_loss", "external_reference_loss",
                "pairs_external_better", "test_coverage_90", "test_raw_coverage_90",
                "test_coverage_50", "test_raw_coverage_50", "test_width_90_pct",
                "external_inside_90_rate"]


def write_csv(path, rows, fields=CSV_FIELDS):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        # UTF-8 BOM + semicolon + decimal comma for Hungarian Excel.
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fields, delimiter=";", extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: (format(v, ".15g").replace(".", ",")
                                     if isinstance(v, float) else v)
                                 for k, v in row.items()})
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def export_individual(result, output):
    """Per-run CSVs are durable immediately, even before the combined export."""
    r, output = result, Path(output)
    meta = {k: r[k] for k in ("experiment_id", "version", "pair_id", "cutoff_date",
            "forecast_date", "seed", "snapshot_id", "duration_seconds", "selection_hash",
            "target", "horizon")}
    key = r["job_key"]
    meta.update(run_id=key, ticker="NVDA", selected_arm=r["arm"],
                selected_config=r.get("config"), selected_iterations=r["iterations"],
                selected_shrinkage=r.get("shrinkage"), selected_combine=r.get("combine"),
                gate_pvalue=r.get("gate_pvalue"), status="success",
                record_type="individual", n_success=1, n_required=len(SEEDS))
    point = {**meta, **r["forecast"], **r["test_summary"], "phase": external_phase(r["horizon"])}
    write_csv(output / "individual" / (key + ".csv"), [point])
    write_csv(output / "historical_test" / (key + ".csv"),
              ({**meta, **x, "phase": "historical_test"} for x in r["historical_test"]))
    return point


def _group_rows(manifest, version, target, horizon, pair_id, group):
    """Seed summary and, separately, the error of the seed-mean prediction."""
    cutoff, next_day = DATE_PAIRS[pair_id - 1]
    forecast = group[0]["forecast_date"] if group else (next_day if horizon == 1 else None)
    base = dict(record_type="seed_summary", phase=external_phase(horizon),
                experiment_id=manifest["experiment_id"], version=version,
                target=target, horizon=horizon,
                pair_id=pair_id, cutoff_date=cutoff, forecast_date=forecast,
                ticker="NVDA", n_success=len(group), n_required=len(SEEDS),
                status="complete" if len(group) == len(SEEDS) else "incomplete")
    if not group:
        return [base]
    metrics = [("loss", "loss_mean", "loss_std")]
    if target == "price":
        metrics += [("ape_pct", "mape_mean_pct", "mape_std_pct"),
                    ("absolute_error_usd", "mae_mean_usd", "mae_std_usd")]
    for metric, mean_key, std_key in metrics:
        vals = [r[metric] for r in group]
        base[mean_key] = statistics.mean(vals)
        base[std_key] = statistics.stdev(vals) if len(vals) > 1 else None
    reference = group[0]
    same = ("actual_value", "reference_value", "selection_hash")
    if any(tuple(r[k] for k in same) != tuple(reference[k] for k in same) for r in group):
        raise ValueError("Inconsistent aggregation group")
    base.update(value_unit=reference["value_unit"],
                reference_win_rate=statistics.mean(r["beats_reference"] for r in group))
    ensemble = {k: v for k, v in base.items()
                if not k.startswith(("mape_", "mae_", "loss_")) and k != "reference_win_rate"}
    ensemble.update(record_type="ensemble_mean", origin_date=cutoff, target_date=forecast)
    if target == "price":
        prices = [r["predicted_adj_close"] for r in group]
        pred = statistics.mean(prices)
        ensemble.update(predicted_adj_close=pred, actual_adj_close=reference["actual_adj_close"],
                        origin_adj_close=reference["origin_adj_close"],
                        prediction_std_usd=statistics.stdev(prices) if len(prices) > 1 else None,
                        **price_metrics(pred, reference["actual_adj_close"],
                                        reference["origin_adj_close"]))
        base["direction_hit_rate"] = statistics.mean(r["direction_hit"] for r in group)
        base["baseline_win_rate"] = statistics.mean(r["beats_baseline"] for r in group)
        base["baseline_tie_rate"] = statistics.mean(r["ties_baseline"] for r in group)
        if reference.get("inside_90") is not None:
            base["inside_90_rate"] = statistics.mean(r["inside_90"] for r in group)
    else:
        # QLIKE elicits the mean of the variance, so seeds are averaged as variances.
        pred = statistics.mean(r["predicted_variance"] for r in group)
        ensemble.update(origin_adj_close=reference["origin_adj_close"],
                        **variance_metrics(pred, reference["actual_variance"],
                                           reference["reference_variance"]))
    return [base, ensemble]


def _track_row(version, target, horizon, groups):
    """Pair-level means. Overlapping historical test windows are NOT independent samples."""
    pairs = [g for g in groups.values() if g]
    if not pairs:
        return dict(version=version, target=target, horizon=horizon, pairs=0, runs=0)
    avg = statistics.mean
    per_pair = lambda key: [avg(r[key] for r in g) for g in pairs]
    present = lambda key: avg(per_pair(key)) if pairs[0][0].get(key) is not None else None
    test, reference = per_pair("test_loss"), per_pair("test_reference_loss")
    external, external_reference = per_pair("loss"), per_pair("reference_loss")
    row = dict(version=version, target=target, horizon=horizon,
               loss_metric=pairs[0][0]["loss_metric"], pairs=len(pairs),
               runs=sum(len(g) for g in pairs),
               pairs_model_selected=sum(g[0]["selected_arm"] not in BENCHMARK_ARMS for g in pairs),
               test_loss=avg(test), test_reference_loss=avg(reference),
               test_gain=avg(reference) - avg(test),
               pairs_test_better=sum(a < b for a, b in zip(test, reference)),
               external_loss=avg(external), external_reference_loss=avg(external_reference),
               pairs_external_better=sum(a < b for a, b in zip(external, external_reference)))
    for key in ("test_har_loss", "test_coverage_90", "test_raw_coverage_90", "test_coverage_50",
                "test_raw_coverage_50", "test_width_90_pct"):
        row[key] = present(key)
    if pairs[0][0].get("inside_90") is not None:
        row["external_inside_90_rate"] = avg(per_pair("inside_90"))
    return row


def count_runs(output, experiment_id):
    """Valid saved repetitions per version/track; reads results only."""
    output, counts = Path(output), {}
    for version in VERSIONS:
        for target, horizon in TRACKS:
            keys = [job_key(version, pair, seed, target, horizon)
                    for pair in range(1, len(DATE_PAIRS) + 1) for seed in SEEDS]
            counts[f"{version}/{track_name(target, horizon)}"] = sum(
                result_valid(output / "results" / (key + ".json"), experiment_id, key) for key in keys)
    return counts


def status_of(counts):
    total = sum(counts.values())
    return dict(successful_runs=total, required_runs=REQUIRED_RUNS,
                status="complete" if total == REQUIRED_RUNS else "incomplete",
                runs_per_track=RUNS_PER_TRACK, tracks=counts)


def export_results(output):
    """results.csv: every external forecast; tracks.csv: one summary row per version and track.
    Separate daily test CSVs avoid Excel's row limit."""
    output = Path(output)
    manifest = read_json(output / "experiment.json")
    rows, track_rows, counts = [], [], {}
    for version in VERSIONS:
        for target, horizon in TRACKS:
            groups = {}
            for pair_id in range(1, len(DATE_PAIRS) + 1):
                group = []
                for seed in SEEDS:
                    key = job_key(version, pair_id, seed, target, horizon)
                    path = output / "results" / (key + ".json")
                    if not result_valid(path, manifest["experiment_id"], key):
                        continue
                    group.append(export_individual(read_json(path), output))
                rows += group + _group_rows(manifest, version, target, horizon, pair_id, group)
                groups[pair_id] = group
            counts[f"{version}/{track_name(target, horizon)}"] = sum(len(g) for g in groups.values())
            track_rows.append(_track_row(version, target, horizon, groups))
    write_csv(output / "results.csv", rows)
    write_csv(output / "tracks.csv", track_rows, TRACK_FIELDS)
    status = status_of(counts)
    atomic_json(output / "status.json", status)
    return status
