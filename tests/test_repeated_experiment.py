"""No-training checks only. Real cached data is used for feature/date checks."""
from pathlib import Path
import ast
import csv

import numpy as np
import pandas as pd
import pytest

from quantimate import experiment as engine
from quantimate import experiment_protocol as p
from quantimate import experiment_runner as runner
from quantimate.experiment_data import (OPTIONAL, Snapshot, check_optional, create_snapshot,
                                        session_after, verify_snapshot)
from quantimate.features.volatility import daily_range_variance
from quantimate.validation.conformal import aci_bands, aci_bands_delayed
from quantimate.validation.splits import make_purged_splits

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def forbid_training_and_network(monkeypatch):
    import lightgbm
    import hmmlearn.hmm
    import nbclient
    import requests
    def forbidden(*args, **kwargs):
        raise AssertionError("Model training, notebook execution and downloads are forbidden in this test suite")
    monkeypatch.setattr(lightgbm.LGBMRegressor, "fit", forbidden)
    monkeypatch.setattr(hmmlearn.hmm.GaussianHMM, "fit", forbidden)
    monkeypatch.setattr(nbclient.NotebookClient, "execute", forbidden)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)


@pytest.fixture
def cached_frame():
    names = ["NVDA", "SPY", "QQQ", "AMD", "SMH", "VIX", "VXN", "TNX"]
    if not all((REPO / "data" / (n + ".parquet")).exists() for n in names):
        pytest.skip("Real local market cache unavailable")
    frames = [pd.read_parquet(REPO / "data" / (n + ".parquet")) for n in names]
    index = frames[0].index
    parts = [f.reindex(index) for f in frames[:5]]
    parts += [f[[n + "_Close"]].reindex(index).ffill(limit=3) for n, f in zip(names[5:], frames[5:])]
    return pd.concat(parts, axis=1)


def test_protocol_count_and_seeds():
    assert len(p.DATE_PAIRS) == len(set(p.DATE_PAIRS)) == 30
    assert len(p.VERSIONS) * len(p.DATE_PAIRS) * len(p.SEEDS) == p.RUNS_PER_TRACK == 3000
    assert p.TRACKS == (("price", 1), ("price", 5), ("price", 21),
                        ("volatility", 1), ("volatility", 5), ("volatility", 21))
    assert p.REQUIRED_RUNS == 18000
    # Labels of the longest horizon must not cross a purged split boundary.
    assert p.PURGE >= max(p.HORIZONS)
    assert p.SEEDS == tuple(range(1, 51))
    assert p.DATE_PAIRS[0] == ("2021-01-29", "2021-02-01")
    assert p.DATE_PAIRS[-1] == ("2025-11-11", "2025-11-12")


def test_metrics_and_baseline_tie():
    m = p.point_metrics(110.0, 100.0, 90.0)
    assert m["ape_pct"] == 10
    assert m["absolute_error_usd"] == 10
    assert m["direction_hit"] == 1
    assert m["beats_baseline"] == 0
    assert m["ties_baseline"] == 1
    assert p.point_metrics(100.0, 100.0, 90.0)["beats_baseline"] == 1
    assert p.point_metrics(90.0, 100.0, 90.0)["direction_hit"] == 0


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_prices_rejected(bad):
    with pytest.raises(ValueError):
        p.point_metrics(bad, 1, 1)


def test_nominal_splits_and_purge():
    s = make_purged_splits(1000, window=1, purge=21, embargo=5)
    assert (s.i_train, s.i_val) == (700, 850)
    assert (len(s.train), len(s.val), len(s.test)) == (674, 124, 150)
    assert s.val[0] - s.train[-1] == 27
    assert s.test[0] - s.val[-1] == 27


def test_features_prefix_invariance_real_cache(cached_frame):
    cutoff = pd.Timestamp("2021-01-29")
    whole = engine.baseline_features(cached_frame)
    prefix = engine.baseline_features(cached_frame.loc[:cutoff])
    pd.testing.assert_frame_equal(whole.loc[:cutoff], prefix, check_freq=False, check_exact=True)
    assert "cal_month_end" not in prefix
    assert not any(c.startswith("earn_") for c in prefix)


def test_label_maturity_and_one_session_real_cache(cached_frame):
    cutoff = "2025-11-11"
    d = engine.make_design(cached_frame.loc[:cutoff], cutoff)
    assert pd.Timestamp(cutoff) not in d.labeled
    assert d.x.index[-1] == pd.Timestamp(cutoff)
    assert (d.target_dates.loc[d.labeled] <= pd.Timestamp(cutoff)).all()
    actual = engine.actual_prices(d, d.labeled)
    reconstructed = d.close.loc[d.labeled].to_numpy() * np.exp(
        d.y.loc[d.labeled].to_numpy() * d.sigma.loc[d.labeled].to_numpy())
    np.testing.assert_allclose(reconstructed, actual, rtol=1e-13)
    calendar = cached_frame.loc[:cutoff].index
    positions = calendar.get_indexer(d.labeled)
    assert pd.DatetimeIndex(d.target_dates.loc[d.labeled]).equals(calendar[positions + 1])


def test_post_cutoff_frame_rejected(cached_frame):
    with pytest.raises(ValueError, match="post-cutoff"):
        engine.make_design(cached_frame, "2021-01-29")


def test_earnings_prefix_invariance_real_calendar(cached_frame):
    if not (REPO / "data/nvda_earnings_verified.csv").exists():
        pytest.skip("Verified real earnings cache unavailable")
    calendar = pd.read_csv(REPO / "data/nvda_earnings_verified.csv")
    index = cached_frame.index
    cutoff = pd.Timestamp("2021-01-29")
    full = engine.past_earnings(index, calendar)
    prefix = engine.past_earnings(index[index <= cutoff], calendar)
    pd.testing.assert_frame_equal(full.loc[:cutoff], prefix, check_freq=False, check_exact=True)


def test_source_notebooks_are_unexecuted_and_valid():
    import nbformat
    for name in runner.NOTEBOOKS.values():
        nb = nbformat.read(REPO / name, as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == "code":
                assert cell.execution_count is None
                assert cell.outputs == []
                ast.parse(cell.source)


def test_atomic_success_resume_and_tamper(tmp_path):
    file = tmp_path / "result.json"
    body = dict(status="success", experiment_id="test", job_key="V8/pair_01/seed_01")
    assert not p.result_valid(file, "test", body["job_key"])
    p.save_result(file, body)
    assert p.result_valid(file, "test", body["job_key"])
    corrupted = p.read_json(file)
    corrupted["seed"] = 999
    p.atomic_json(file, corrupted)
    with pytest.raises(ValueError, match="damaged"):
        p.result_valid(file, "test", body["job_key"])


def test_retry_same_job_three_attempts_and_resume(tmp_path):
    key = p.job_key("V8", 1, 7)
    job = dict(output=str(tmp_path), job_key=key, experiment_id="test", seed=7,
               result_path=str(tmp_path / "result.json"))
    attempts = []
    def launcher(path):
        observed = p.read_json(path)
        attempts.append(observed)
        if len(attempts) < 3:
            raise RuntimeError("unit-test failure injection")
        p.save_result(job["result_path"], dict(status="success", experiment_id="test", job_key=key))
    assert runner.execute_with_retries(job, launcher=launcher)
    assert len(attempts) == 3
    assert all(x == job for x in attempts)
    assert runner.execute_with_retries(job, launcher=lambda _: pytest.fail("Completed job restarted"))
    assert len((tmp_path / "failures" / (key + ".jsonl")).read_text().splitlines()) == 2


def test_retry_exhaustion_is_incomplete(tmp_path):
    key = p.job_key("V8.5", 2, 3)
    job = dict(output=str(tmp_path), job_key=key, experiment_id="test", seed=3,
               result_path=str(tmp_path / "result.json"))
    attempts = []
    def fail(path):
        attempts.append(path)
        raise RuntimeError("unit-test failure injection")
    assert not runner.execute_with_retries(job, launcher=fail)
    assert len(attempts) == 3
    assert not Path(job["result_path"]).exists()


def _saved_run(tmp_path, seed, target, horizon, forecast, summary):
    key = p.job_key("V8", 1, seed, target, horizon)
    body = dict(status="success", experiment_id="test", job_key=key, version="V8",
                pair_id=1, seed=seed, cutoff_date="2021-01-29", forecast_date=forecast["target_date"],
                target=target, horizon=horizon, snapshot_id="fixture", duration_seconds=1.0,
                selection_hash="fixture", arm=p.BENCHMARK_ARM[target], iterations=0,
                forecast=forecast, historical_test=[forecast], test_summary=summary)
    p.save_result(tmp_path / "results" / (key + ".json"), body)


def test_csv_adjacent_prices_and_ensemble_error(tmp_path):
    # Arithmetic fixtures only, stored in pytest's temporary directory; no market data generation.
    p.atomic_json(tmp_path / "experiment.json", dict(experiment_id="test"))
    for seed, predicted in ((1, 90.0), (2, 110.0)):
        point = dict(origin_date="2021-01-29", target_date="2021-02-01",
                     origin_adj_close=95.0, predicted_adj_close=predicted, actual_adj_close=100.0,
                     lower_90=80.0, upper_90=105.0, inside_90=1, inside_50=0,
                     **p.price_metrics(predicted, 100.0, 95.0))
        _saved_run(tmp_path, seed, "price", 1, point,
                   dict(loss_metric="ape_pct", test_loss=10.0, test_reference_loss=5.0,
                        test_mape_pct=10.0, test_mae_usd=10.0, test_direction_hit_rate=0.5,
                        test_coverage_90=0.9, test_raw_coverage_90=0.8))
    for seed, variance in ((1, 2e-4), (2, 6e-4)):
        point = dict(origin_date="2021-01-29", target_date="2021-02-05", origin_adj_close=95.0,
                     **p.variance_metrics(variance, 4e-4, 3e-4))
        _saved_run(tmp_path, seed, "volatility", 5, point,
                   dict(loss_metric="qlike", test_loss=-6.6, test_reference_loss=-6.5, test_har_loss=-6.55))
    status = p.export_results(tmp_path)
    assert (status["successful_runs"], status["required_runs"], status["status"]) == (4, 18000, "incomplete")
    assert status["tracks"]["V8/price_h01"] == status["tracks"]["V8/volatility_h05"] == 2
    assert sum(status["tracks"].values()) == 4 and len(status["tracks"]) == 12
    assert p.status_of(p.count_runs(tmp_path, "test")) == status
    with (tmp_path / "results.csv").open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        columns, rows = reader.fieldnames, list(reader)
    assert columns.index("actual_adj_close") == columns.index("predicted_adj_close") + 1
    group = [r for r in rows if (r["version"], r["pair_id"], r["target"]) == ("V8", "1", "price")
             and r["horizon"] == "1"]
    summary = next(r for r in group if r["record_type"] == "seed_summary")
    ensemble = next(r for r in group if r["record_type"] == "ensemble_mean")
    assert summary["mape_mean_pct"] == summary["loss_mean"] == "10"
    assert ensemble["ape_pct"] == "0"  # Mean error and error of mean are different quantities.
    assert summary["n_success"] == "2" and summary["inside_90_rate"] == "1"
    assert summary["status"] == ensemble["status"] == "incomplete"
    # Volatility: seeds are averaged as variances (QLIKE scores the mean), shown as volatility.
    group = [r for r in rows if r["target"] == "volatility" and r["horizon"] == "5" and r["pair_id"] == "1"]
    ensemble = next(r for r in group if r["record_type"] == "ensemble_mean")
    assert ensemble["predicted_variance"] == "0,0004" and ensemble["predicted_value"] == "2"
    assert ensemble["phase"] == "external_5_sessions" and ensemble["predicted_adj_close"] == ""
    assert len(list((tmp_path / "historical_test").rglob("*.csv"))) == 4
    with (tmp_path / "tracks.csv").open(encoding="utf-8-sig", newline="") as f:
        tracks = {(r["version"], r["target"], r["horizon"]): r for r in csv.DictReader(f, delimiter=";")}
    assert len(tracks) == 12
    price = tracks[("V8", "price", "1")]
    assert (price["pairs"], price["runs"], price["pairs_model_selected"]) == ("1", "2", "0")
    assert (price["test_gain"], price["test_coverage_90"], price["external_inside_90_rate"]) == ("-5", "0,9", "1")
    assert tracks[("V8", "volatility", "5")]["test_har_loss"] == "-6,55"
    assert tracks[("V8.5", "price", "21")]["pairs"] == "0"


def test_snapshot_checksum_and_path_guard(tmp_path):
    file = tmp_path / "data.txt"
    file.write_text("immutable")
    body = dict(schema=1, files={"data.txt": p.file_hash(file)})
    body["snapshot_id"] = p.digest(body)
    p.atomic_json(tmp_path / "snapshot.json", body)
    assert verify_snapshot(tmp_path)["snapshot_id"] == body["snapshot_id"]
    file.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        verify_snapshot(tmp_path)


def test_old_cache_does_not_silently_shorten_window(tmp_path):
    if not (REPO / "data/NVDA.parquet").exists():
        pytest.skip("Real local market cache unavailable")
    nvda = pd.read_parquet(REPO / "data/NVDA.parquet")
    if nvda.index.min() <= pd.Timestamp(p.RAW_START) + pd.Timedelta(days=7):
        pytest.skip("Cache already has full history")
    absent = [name for name in OPTIONAL if not (REPO / "data" / name).exists()]
    with pytest.raises(ValueError, match="required start"):
        create_snapshot(tmp_path / "snapshot", REPO / "data", download=False, allow_missing=absent)
    assert not (tmp_path / "snapshot").exists()
    assert not list(tmp_path.glob("snapshot.partial-*"))


def test_no_default_execution(capsys):
    assert runner.main([]) == 0
    assert "No default execution" in capsys.readouterr().out


def test_code_fingerprint_changes_with_source(tmp_path):
    for relative in ("quantimate/a.py", "experiments/runners/run_experiments.py",
                     "experiments/generators/nb_walkforward.py",
                     "experiments/generators/build_notebook_v8.py",
                     "experiments/generators/build_notebook_v8_5.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture")
    for name in runner.NOTEBOOKS.values():
        p.atomic_json(tmp_path / name, {"cells": []})
    before = runner.code_fingerprint(tmp_path)
    (tmp_path / "quantimate/a.py").write_text("# changed fixture")
    assert runner.code_fingerprint(tmp_path) != before


def _selection_fixture(cached_frame, monkeypatch, predict, real_constant=True):
    """Real cached prices; only the model fit and its predictions are replaced."""
    cutoff = "2025-11-11"
    d = engine.make_design(cached_frame.loc[:cutoff], cutoff)
    splits = make_purged_splits(len(d.labeled), window=1)
    train_dates, val_dates = d.labeled[splits.train], d.labeled[splits.val]
    calls = []
    monkeypatch.setattr(engine, "build_arms", lambda *args: ({"baseline": d.x}, []))
    def fit(x, y, hp, seed, iterations, quantiles=p.QUANTILES):
        # Stage 1 fits on the whole training window, stage 2 on purged folds of it;
        # validation and test rows never reach a fit.
        assert x.index.isin(train_dates).all() and x.index.equals(y.index)
        assert seed == 42
        calls.append("fit" if x.index.equals(train_dates) else "fold-fit")
        return object()
    def predict_returns(models, x, design, dates, iterations):
        on_validation = dates.equals(val_dates)
        assert on_validation or dates.isin(train_dates).all()
        calls.append("validation" if on_validation else "fold-predict")
        return predict(design, dates, iterations, on_validation)
    monkeypatch.setattr(engine, "fit_quantiles", fit)
    monkeypatch.setattr(engine, "predict_log_returns", predict_returns)
    if not real_constant:
        # Keeps the outcome independent of how strongly the cached window happens to trend.
        monkeypatch.setattr(engine, "constant_log_returns", lambda design, fit_dates, dates: np.zeros(len(dates)))
    return cutoff, d, splits, calls


def _true_log_returns(design, dates):
    return design.y.loc[dates].to_numpy() * design.sigma.loc[dates].to_numpy()


def test_selection_only_sees_train_and_validation(cached_frame, monkeypatch):
    cutoff, d, splits, calls = _selection_fixture(
        cached_frame, monkeypatch, lambda design, dates, iterations, on_validation: np.zeros(len(dates)),
        real_constant=False)
    result = engine.select(None, cutoff, "V8", d, splits)
    assert calls == (["fit"] + ["validation"] * len(p.ITERATIONS)) * len(p.BASE_CONFIGS)
    # One constant-median candidate plus every arm x config x iteration, each at every shrinkage.
    expected = len(p.SHRINKAGE) * (1 + len(p.BASE_CONFIGS) * len(p.ITERATIONS))
    assert result["decision_count"] == len(result["decision_log"]) == expected
    assert result["gate"]["candidates"] == expected
    assert {row["arm"] for row in result["decision_log"]} == {"constant", "baseline"}
    assert "hpo" not in result  # V8 never searches hyper-parameters.


def test_gate_keeps_naive_benchmark_without_evidence(cached_frame, monkeypatch):
    """Models that merely reproduce the benchmark must not displace it."""
    cutoff, d, splits, _ = _selection_fixture(
        cached_frame, monkeypatch, lambda design, dates, iterations, on_validation: np.zeros(len(dates)),
        real_constant=False)
    result = engine.select(None, cutoff, "V8", d, splits)
    assert result["gate"]["passed"] is False and result["gate"]["pvalue"] >= p.GATE_ALPHA
    assert result["gate"]["replication"] is None      # stage 2 is not even run
    chosen = result["selected"]
    assert (chosen["arm"], chosen["iterations"], chosen["shrinkage"], chosen["features"]) == ("naive", 0, 0.0, [])
    assert chosen["validation_mape_pct"] == result["gate"]["naive_validation_mape_pct"]
    # Degenerate comparison (identical losses): the pairwise statistic is undefined,
    # and the result must still be strict JSON for the atomic, checksummed save.
    assert result["gate"]["best_vs_naive_dm"]["statistic"] is None
    p.digest(result)


def test_gate_admits_a_genuinely_superior_candidate(cached_frame, monkeypatch):
    """Power check on real prices: a forecast carrying half of the realised return
    (an oracle, used only to exercise the gate) must pass and be selected unshrunk."""
    cutoff, d, splits, calls = _selection_fixture(
        cached_frame, monkeypatch,
        lambda design, dates, iterations, on_validation: 0.5 * _true_log_returns(design, dates))
    result = engine.select(None, cutoff, "V8", d, splits)
    gate = result["gate"]
    assert gate["passed"] is True and gate["pvalue"] < p.GATE_ALPHA
    replication = gate["replication"]
    assert replication["replicated"] is True and replication["pvalue"] < p.GATE_ALPHA
    assert replication["search_adjusted"] is False and len(replication["fold_gain_pct"]) == p.REPLICATION_FOLDS
    assert all(gain > 0 for gain in replication["fold_gain_pct"])
    assert calls.count("fold-fit") == calls.count("fold-predict") == p.REPLICATION_FOLDS
    chosen = result["selected"]
    assert chosen["arm"] == "baseline" and chosen["shrinkage"] == 1.0
    assert chosen["features"] == list(d.x.columns)
    assert chosen["validation_mape_pct"] < gate["naive_validation_mape_pct"]
    assert all("features" not in row for row in result["decision_log"])
    p.digest(result)


def test_gate_rejects_an_edge_that_does_not_replicate(cached_frame, monkeypatch):
    """Skill that exists only in the validation period (a "pocket") is significant
    there, but absent out-of-fold in the training window: the benchmark must stay."""
    def pocket(design, dates, iterations, on_validation):
        return 0.5 * _true_log_returns(design, dates) if on_validation else np.zeros(len(dates))
    cutoff, d, splits, _ = _selection_fixture(cached_frame, monkeypatch, pocket)
    result = engine.select(None, cutoff, "V8", d, splits)
    gate = result["gate"]
    assert gate["validation_passed"] is True and gate["pvalue"] < p.GATE_ALPHA
    assert gate["replication"]["replicated"] is False
    assert gate["passed"] is False
    assert result["selected"]["arm"] == "naive"


def test_nested_search_never_reads_validation_or_test(cached_frame, monkeypatch):
    cutoff = "2025-11-11"
    d = engine.make_design(cached_frame.loc[:cutoff], cutoff)
    splits = make_purged_splits(len(d.labeled), window=1)
    train_dates = d.labeled[splits.train]
    fits = []
    class Median:
        def predict(self, x, num_iteration=None):
            assert x.index.isin(train_dates).all()
            return np.zeros(len(x))
    def fit(x, y, hp, seed, iterations, quantiles=p.QUANTILES):
        assert x.index.isin(train_dates).all() and x.index.equals(y.index)
        assert quantiles == (0.5,) and seed == 42 and iterations == max(p.ITERATIONS)
        assert not set(hp) & {"subsample", "colsample_bytree", "subsample_freq", "n_estimators"}
        fits.append(x.index)
        return [Median()]
    monkeypatch.setattr(engine, "fit_quantiles", fit)
    result = engine.nested_hpo(d.x, d, train_dates)
    assert len(fits) == p.HPO_TRIALS * p.HPO_FOLDS
    # An interior fold leaves its held-out block plus a purge+embargo band on BOTH sides.
    block, band = len(train_dates) // p.HPO_FOLDS, 21 + 5
    positions = train_dates.get_indexer(fits[1])
    assert np.diff(positions).max() >= block + 2 * band
    assert len(fits[1]) <= len(train_dates) - block - 2 * band + 1
    # Identical out-of-fold losses: the default (trial 0) stays, nothing is admitted.
    assert result["best_trial"] == 0 and result["accepted"] is False
    assert result["scope"] == "training window only"
    assert result["oof_spa_pvalue"] >= p.GATE_ALPHA   # zero forecasts cannot beat the zero benchmark


def test_repeat_refits_from_scratch_and_reads_external_actual_last(cached_frame, monkeypatch):
    cutoff, forecast = "2025-11-11", "2025-11-12"
    d = engine.make_design(cached_frame.loc[:cutoff], cutoff)
    splits = make_purged_splits(len(d.labeled), window=1)
    events, models = [], []
    monkeypatch.setattr(engine, "build_arms", lambda *args: ({"generic_nvda": d.x}, []))
    def fit(x, y, hp, seed, iterations, quantiles=p.QUANTILES):
        assert seed == 7 and iterations == 100
        expected = d.labeled[splits.train] if not models else d.labeled
        assert x.index.equals(expected) and y.index.equals(expected)
        assert pd.Timestamp(cutoff) not in y.index
        model = object()
        models.append(model)
        events.append("fit")
        return model
    def predict(model, x, design, dates, iterations):
        assert model is models[-1] and iterations == 100
        if len(models) == 1:
            assert dates.equals(d.labeled[splits.test])
        else:
            assert dates.equals(pd.DatetimeIndex([cutoff]))
        events.append("predict")
        return np.full(len(dates), 0.02)
    class ReadGuard:
        def daily(self, ticker):
            assert events == ["fit", "predict", "fit", "predict"]
            events.append("read_external_actual")
            return cached_frame[["NVDA_Close"]]
    monkeypatch.setattr(engine, "fit_quantiles", fit)
    monkeypatch.setattr(engine, "predict_log_returns", predict)
    chosen = dict(selected=dict(arm="generic_nvda", config="default", iterations=100, shrinkage=0.5,
                                combine="none", hp={}, features=list(d.x)), gate=dict(pvalue=0.01))
    r = engine.repeat(ReadGuard(), cutoff, forecast, "V8.5", 7, d, splits, chosen)
    assert len(models) == 2 and models[0] is not models[1]
    assert events[-1] == "read_external_actual"
    assert r["forecast"]["actual_adj_close"] == cached_frame.loc[forecast, "NVDA_Close"]
    # Shrinkage 0.5 halves the predicted log return on the way to the price.
    assert r["forecast"]["predicted_adj_close"] == pytest.approx(d.close.loc[cutoff] * np.exp(0.01), rel=1e-12)
    assert len(r["historical_test"]) == len(splits.test)
    assert (r["config"], r["shrinkage"], r["gate_pvalue"]) == ("default", 0.5, 0.01)
    assert (r["target"], r["horizon"], r["combine"]) == ("price", 1, "none")


class _ActualOnly:
    def __init__(self, frame):
        self.frame = frame
    def daily(self, ticker):
        return self.frame[["NVDA_Close"]]


def _forbid_models(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No model may be built or fitted for this selection")
    for name in ("fit_quantiles", "build_arms", "add_hmm"):
        monkeypatch.setattr(engine, name, forbidden)


def test_repeat_with_naive_selection_is_exactly_the_benchmark(cached_frame, monkeypatch):
    cutoff, forecast = "2025-11-11", "2025-11-12"
    d = engine.make_design(cached_frame.loc[:cutoff], cutoff)
    splits = make_purged_splits(len(d.labeled), window=1)
    _forbid_models(monkeypatch)
    selection = dict(selected=dict(engine.NAIVE), gate=dict(pvalue=0.7))
    r = engine.repeat(_ActualOnly(cached_frame), cutoff, forecast, "V8", 3, d, splits, selection)
    assert r["arm"] == "naive" and r["refit_rows"] == {}
    assert r["forecast"]["predicted_adj_close"] == r["forecast"]["origin_adj_close"]
    assert r["forecast"]["ties_baseline"] == 1 and r["forecast"]["beats_baseline"] == 0
    assert all(row["ties_baseline"] == 1 for row in r["historical_test"])
    assert r["test_summary"]["test_mape_pct"] == pytest.approx(
        np.mean([row["baseline_ape_pct"] for row in r["historical_test"]]), rel=1e-12)
    assert r["test_summary"]["test_loss"] == r["test_summary"]["test_reference_loss"]
    # The benchmark still carries a calibrated band, built without any model.
    point, summary = r["forecast"], r["test_summary"]
    assert point["lower_90"] < point["lower_50"] < point["origin_adj_close"] < point["upper_50"] < point["upper_90"]
    assert point["inside_90"] in (0, 1) and point["raw_lower_90"] < point["raw_upper_90"]
    for level in (90, 50):
        assert 0 < summary[f"test_coverage_{level}"] <= 1 and 0 < summary[f"test_raw_coverage_{level}"] <= 1
    assert summary["test_width_90_pct"] > 0 and summary["test_interval_score_90_pct"] >= summary["test_width_90_pct"]
    p.digest(r)


def test_repeat_with_constant_selection_uses_only_matured_labels(cached_frame, monkeypatch):
    cutoff, forecast = "2025-11-11", "2025-11-12"
    d = engine.make_design(cached_frame.loc[:cutoff], cutoff)
    splits = make_purged_splits(len(d.labeled), window=1)
    _forbid_models(monkeypatch)
    selection = dict(selected=dict(arm="constant", config="none", hp={}, iterations=0,
                                   shrinkage=0.5, features=[]), gate=dict(pvalue=0.01))
    r = engine.repeat(_ActualOnly(cached_frame), cutoff, forecast, "V8", 3, d, splits, selection)
    train, test = d.labeled[splits.train], d.labeled[splits.test]
    # Historical test: the median of TRAIN labels only; external forecast: all matured labels.
    first = r["historical_test"][0]
    expected = d.close.loc[test[0]] * np.exp(0.5 * np.median(d.y.loc[train]) * d.sigma.loc[test[0]])
    assert first["predicted_adj_close"] == pytest.approx(expected, rel=1e-12)
    origin = pd.Timestamp(cutoff)
    expected = d.close.loc[origin] * np.exp(0.5 * np.median(d.y.loc[d.labeled]) * d.sigma.loc[origin])
    assert r["forecast"]["predicted_adj_close"] == pytest.approx(expected, rel=1e-12)
    assert r["refit_rows"] == {"NVDA": len(d.labeled)} and origin not in d.labeled


def test_shrinkage_scales_the_predicted_return_toward_the_benchmark(cached_frame):
    cutoff = "2025-11-11"
    d = engine.make_design(cached_frame.loc[:cutoff], cutoff)
    dates = d.labeled[-5:]
    close = d.close.loc[dates].to_numpy()
    returns = np.array([0.02, -0.01, 0.0, 0.03, -0.04])
    np.testing.assert_array_equal(engine.prices_from_returns(returns, d, dates, 0.0), close)
    np.testing.assert_allclose(engine.prices_from_returns(returns, d, dates, 1.0), close * np.exp(returns), rtol=1e-15)
    half = engine.prices_from_returns(returns, d, dates, 0.5)
    assert (np.abs(half - close) <= np.abs(close * np.exp(returns) - close)).all()


def test_schema3_protocol_is_fixed_and_serialisable():
    assert p.PROTOCOL["schema"] == 3
    assert p.PROTOCOL["targets"] == p.TARGETS and p.PROTOCOL["horizons"] == p.HORIZONS
    assert p.spa_block(1) == p.SPA_AVG_BLOCK and p.spa_block(21) > 21     # longer than the label overlap
    assert 0 < p.ACI_GAMMA < 0.1 and [level for level, _, _ in p.BAND_LEVELS] == [90, 50]
    assert set(p.CONFORMAL_HORIZONS) <= set(p.HORIZONS) and p.VOLATILITY_FALLBACK == "har"
    for column in ("target", "horizon", "loss", "reference_loss", "lower_90", "upper_90",
                   "test_coverage_90", "test_raw_coverage_90", "test_har_loss"):
        assert column in p.CSV_FIELDS
    assert len(p.CSV_FIELDS) == len(set(p.CSV_FIELDS))
    assert min(p.ITERATIONS) <= 5 and max(p.ITERATIONS) == 400      # the old lower edge (50) is now interior
    # Volatility tracks search further up: 400 was the winning edge of the first grid.
    assert p.VOLATILITY_ITERATIONS[:len(p.ITERATIONS)] == p.ITERATIONS and max(p.VOLATILITY_ITERATIONS) == 1600
    assert p.ITERATION_GRID == {"price": p.ITERATIONS, "volatility": p.VOLATILITY_ITERATIONS}
    assert set(p.ITERATION_GRID) == set(p.TARGETS)
    assert p.SHRINKAGE[-1] == 1.0 and all(0 < v <= 1 for v in p.SHRINKAGE)
    assert p.BASE_CONFIGS[0] == ("default", {})
    regularized = dict(p.BASE_CONFIGS)["regularized"]
    assert regularized["num_leaves"] <= 2 ** regularized["max_depth"]
    assert p.GATE_ALPHA == 0.10 and p.PBO_LIMIT == 0.5 and p.REPLICATION_FOLDS >= 2
    p.digest(p.PROTOCOL)                                               # canonical JSON, no NaN
    for column in ("selected_config", "selected_shrinkage", "gate_pvalue"):
        assert column in p.CSV_FIELDS


def test_sampling_ratios_and_seed_are_forced_without_fitting(monkeypatch):
    import lightgbm
    captured = []
    class StopBeforeFit:
        def __init__(self, **kwargs):
            captured.append(kwargs)
        def fit(self, *args):
            raise RuntimeError("stop before training")
    monkeypatch.setattr(lightgbm, "LGBMRegressor", StopBeforeFit)
    with pytest.raises(RuntimeError, match="stop before training"):
        engine.fit_quantiles(None, None, {"subsample": 1.0, "colsample_bytree": 1.0}, 17, 100)
    hp = captured[0]
    assert hp["subsample"] == hp["colsample_bytree"] == 0.8
    assert hp["subsample_freq"] == 1
    assert hp["n_estimators"] == 100
    assert hp["deterministic"] and hp["force_row_wise"] and hp["n_jobs"] == 1
    assert all(hp[k] == 17 for k in ("random_state", "bagging_seed", "feature_fraction_seed", "data_random_seed"))


@pytest.mark.parametrize("version", ["V8", "V8.5"])
def test_optional_feature_adapters_on_real_cache(cached_frame, version):
    if not all((REPO / "data" / name).exists() for name in
               ("nvda_earnings_verified.csv", "edgar_cache/form4_nvda.parquet")):
        pytest.skip("Real optional caches unavailable")
    class CacheReader:
        path = REPO / "data"
        manifest = {"files": {p.relative_to(REPO / "data").as_posix(): "unused"
                              for p in (REPO / "data").rglob("*") if p.is_file()},
                    "missing_optional": []}
        optional = Snapshot.optional
    index = cached_frame.loc[:"2025-11-11"].index
    blocks, notes = engine.optional_features(CacheReader(), index, version)
    assert "earnings" in blocks and "insider" in blocks
    for name, block in blocks.items():
        assert block.index.equals(index), name
        assert not np.isinf(block.to_numpy(dtype=float)).any(), name
    cutoff = pd.Timestamp("2021-01-29")
    prefix_blocks, _ = engine.optional_features(CacheReader(), index[index <= cutoff], version)
    for name in blocks:
        pd.testing.assert_frame_equal(blocks[name].loc[:cutoff], prefix_blocks[name],
                                      check_freq=False, check_exact=True)


# ── Schema 3: tracks, volatility target, calibrated price bands ───────────────────────────────

def test_track_job_keys_and_unknown_track():
    assert p.job_key("V8", 1) == "V8/price_h01/pair_01/selection"
    assert p.job_key("V8.5", 3, 7, "volatility", 21) == "V8_5/volatility_h21/pair_03/seed_07"
    assert len({p.job_key(v, 1, 1, t, h) for v in p.VERSIONS for t, h in p.TRACKS}) == 12
    with pytest.raises(ValueError, match="Unknown track"):
        p.job_key("V8", 1, 1, "price", 2)
    assert p.external_phase(1) == "external_next_day" and p.external_phase(21) == "external_21_sessions"


def test_variance_metrics_and_qlike():
    m = p.variance_metrics(4e-4, 4e-4, 2e-4)
    assert m["predicted_value"] == pytest.approx(2.0) and m["value_unit"] == "vol_pct"
    # QLIKE is minimised by the true variance: a perfect forecast beats any other.
    assert m["beats_reference"] == 1 and m["loss"] < m["reference_loss"]
    assert p.variance_metrics(3e-4, 4e-4, 3e-4)["ties_reference"] == 1
    for bad in (0.0, -1e-4, float("nan")):
        with pytest.raises(ValueError):
            p.variance_metrics(bad, 4e-4, 3e-4)


def test_session_after_and_optional_check(tmp_path):
    index = pd.DatetimeIndex(["2021-01-28", "2021-01-29", "2021-02-01", "2021-02-02"])
    assert session_after(index, "2021-01-29", 1) == "2021-02-01"
    assert session_after(index, "2021-01-29", 2) == "2021-02-02"
    with pytest.raises(ValueError, match="No observed 3-session target"):
        session_after(index, "2021-01-29", 3)
    # An absent data layer is an error unless it is accepted by name.
    with pytest.raises(FileNotFoundError, match="--allow-missing"):
        check_optional(tmp_path)
    assert check_optional(tmp_path, allow_missing=OPTIONAL) == list(OPTIONAL)
    with pytest.raises(ValueError, match="Unknown optional"):
        check_optional(tmp_path, allow_missing=["not_a_source.parquet"])


@pytest.mark.parametrize("horizon", [5, 21])
def test_multi_horizon_price_labels_real_cache(cached_frame, horizon):
    cutoff = "2025-11-11"
    d = engine.make_design(cached_frame.loc[:cutoff], cutoff, horizon=horizon)
    calendar = cached_frame.loc[:cutoff].index
    positions = calendar.get_indexer(d.labeled)
    assert pd.DatetimeIndex(d.target_dates.loc[d.labeled]).equals(calendar[positions + horizon])
    assert (d.target_dates.loc[d.labeled] <= pd.Timestamp(cutoff)).all()
    # The last h origins have no matured label and must stay unlabeled.
    assert not set(calendar[-horizon:]) & set(d.labeled)
    reconstructed = d.close.loc[d.labeled].to_numpy() * np.exp(
        d.y.loc[d.labeled].to_numpy() * d.sigma.loc[d.labeled].to_numpy() * np.sqrt(horizon))
    np.testing.assert_allclose(reconstructed, engine.actual_prices(d, d.labeled), rtol=1e-12)


def test_daily_range_variance_is_causal_and_positive(cached_frame):
    cutoff = pd.Timestamp("2021-01-29")
    whole = daily_range_variance(cached_frame, "NVDA")
    prefix = daily_range_variance(cached_frame.loc[:cutoff], "NVDA")
    pd.testing.assert_series_equal(whole.loc[:cutoff], prefix, check_freq=False, check_exact=True)
    assert np.isnan(whole.iloc[0]) and (whole.iloc[1:] > 0).all()


@pytest.mark.parametrize("horizon", [1, 5, 21])
def test_volatility_design_real_cache(cached_frame, horizon):
    cutoff, earlier = "2025-11-11", "2025-06-10"
    frame = cached_frame.loc[:cutoff]
    d = engine.make_design(frame, cutoff, target="volatility", horizon=horizon)
    assert pd.Timestamp(cutoff) in d.x.index and pd.Timestamp(cutoff) not in d.labeled
    assert (d.target_dates.loc[d.labeled] <= pd.Timestamp(cutoff)).all()
    # Target = sum of the NEXT h daily variances / (h * EWMA variance known at the origin).
    daily = daily_range_variance(frame, "NVDA")
    calendar = frame.index
    for date in d.labeled[[0, len(d.labeled) // 2, -1]]:
        position = calendar.get_loc(date)
        future = daily.iloc[position + 1: position + 1 + horizon].sum()
        assert d.realized.loc[date] == pytest.approx(future, rel=1e-12)
        assert d.y.loc[date] == pytest.approx(future / (horizon * d.sigma.loc[date] ** 2), rel=1e-12)
    assert (d.y.loc[d.labeled] > 0).all()
    # Features (HAR terms included) never look ahead: an earlier cutoff gives the same rows.
    assert {f"har_ratio_{w}" for w in p.HAR_WINDOWS} <= set(d.x.columns)
    short = engine.make_design(cached_frame.loc[:earlier], earlier, target="volatility", horizon=horizon)
    common = short.x.index.intersection(d.x.index)
    assert len(common) > 1000
    pd.testing.assert_frame_equal(short.x.loc[common], d.x.loc[common], check_freq=False, check_exact=True)
    # The price track keeps its schema-2 feature set.
    assert not any(c.startswith("har_ratio_") for c in engine.make_design(frame, cutoff).x.columns)


def _volatility_fixture(cached_frame, horizon=5):
    cutoff = "2025-11-11"
    d = engine.make_design(cached_frame.loc[:cutoff], cutoff, target="volatility", horizon=horizon)
    splits = make_purged_splits(len(d.labeled), window=1)
    return cutoff, d, splits


def test_volatility_benchmark_har_and_shrinkage_math(cached_frame):
    import dataclasses
    cutoff, d, splits = _volatility_fixture(cached_frame)
    train, val = d.labeled[splits.train], d.labeled[splits.val]
    level = engine.reference_level(d, train)
    assert level == pytest.approx(d.y.loc[train].mean(), rel=1e-14)
    bench = engine.benchmark_values(d, level, val)
    np.testing.assert_allclose(bench, level * 5 * d.sigma.loc[val].to_numpy() ** 2, rtol=1e-14)
    gain = engine.har_gain(d, train, val, level)
    np.testing.assert_array_equal(engine.values_from_gain(d, level, val, gain, 0.0), bench)
    full = engine.values_from_gain(d, level, val, gain, 1.0)
    np.testing.assert_allclose(full, engine.har_variance(d, train, val), rtol=1e-12)
    half = engine.values_from_gain(d, level, val, gain, 0.5)
    np.testing.assert_allclose(half, np.sqrt(bench * full), rtol=1e-12)     # geometric shrinkage
    # HAR is fitted on the training window only: later outcomes cannot move its coefficients.
    changed = d.realized.copy()
    changed.loc[val] *= 3.0
    beta, smear = engine.fit_har(d, train)
    beta2, smear2 = engine.fit_har(dataclasses.replace(d, realized=changed), train)
    np.testing.assert_array_equal(beta, beta2)
    assert smear == smear2 and smear > 0
    # QLIKE prefers the realized variance itself to any other forecast.
    actual = engine.actual_values(d, val)
    assert engine.loss_values(d, actual, actual).mean() < engine.loss_values(d, bench, actual).mean()


def _volatility_selection(cached_frame, monkeypatch, ratio, har):
    """Real cached prices; only the tree model and (optionally) HAR are replaced."""
    cutoff, d, splits = _volatility_fixture(cached_frame)
    train_dates = d.labeled[splits.train]
    monkeypatch.setattr(engine, "build_arms", lambda *args: ({"baseline": d.x}, []))
    class Model:
        def __init__(self, level):
            self.level = level
        def predict(self, x, num_iteration=None):
            return ratio(d, x.index, self.level)
    def fit(x, y, hp, seed, iterations):
        assert x.index.isin(train_dates).all() and x.index.equals(y.index) and seed == 42
        assert (y > 0).all()
        return [Model(float(y.mean()))]
    monkeypatch.setattr(engine, "fit_variance", fit)
    monkeypatch.setattr(engine, "fit_quantiles", lambda *a, **k: pytest.fail("quantile fit on a volatility track"))
    if har is not None:
        monkeypatch.setattr(engine, "har_gain", har)
    return engine.select(None, cutoff, "V8", d, splits), d


def test_volatility_gate_keeps_benchmark_without_evidence(cached_frame, monkeypatch):
    result, d = _volatility_selection(
        cached_frame, monkeypatch, ratio=lambda d, index, level: np.full(len(index), level),
        har=lambda design, fit_dates, dates, level: np.zeros(len(dates)))
    gate = result["gate"]
    assert gate["passed"] is False and gate["replication"] is None and gate["loss_metric"] == "qlike"
    assert gate["fallback"] is None      # stage 1 did not pass: no second look of any kind
    assert gate["average_block"] == p.spa_block(5)
    chosen = result["selected"]
    assert (chosen["arm"], chosen["iterations"], chosen["shrinkage"], chosen["combine"]) == ("ewma", 0, 0.0, "none")
    assert chosen["validation_qlike"] == gate["benchmark_validation_loss"]
    # HAR, plus every arm x config x iteration both alone and blended with HAR, at every shrinkage.
    expected = len(p.SHRINKAGE) * (1 + len(p.BASE_CONFIGS) * len(p.VOLATILITY_ITERATIONS)
                                   * len(p.VOLATILITY_COMBINE))
    assert result["decision_count"] == gate["candidates"] == expected
    assert {row["arm"] for row in result["decision_log"]} == {"har", "baseline"}
    p.digest(result)


def test_volatility_gate_admits_a_genuinely_superior_candidate(cached_frame, monkeypatch):
    """A forecast half-way (in logs) to the realized ratio is an oracle used only to
    exercise the gate: it must pass both stages on real prices."""
    result, d = _volatility_selection(
        cached_frame, monkeypatch, ratio=lambda d, index, level: np.sqrt(d.y.loc[index].to_numpy() * level),
        har=lambda design, fit_dates, dates, level: np.zeros(len(dates)))
    gate = result["gate"]
    assert gate["passed"] is True and gate["pvalue"] < p.GATE_ALPHA
    assert gate["replication"]["replicated"] is True and all(g > 0 for g in gate["replication"]["fold_gain"])
    assert "gain_pct" not in gate["replication"]            # QLIKE gains are not percentage points
    chosen = result["selected"]
    assert (chosen["arm"], chosen["combine"], chosen["shrinkage"]) == ("baseline", "none", 1.0)
    assert chosen["validation_qlike"] < gate["benchmark_validation_loss"]
    p.digest(result)


class _Ohlc:
    def __init__(self, frame):
        self.frame = frame
    def daily(self, ticker):
        return self.frame[["NVDA_Open", "NVDA_High", "NVDA_Low", "NVDA_Close"]]


@pytest.mark.parametrize("arm", ["ewma", "har"])
def test_repeat_volatility_without_tree_model(cached_frame, monkeypatch, arm):
    cutoff, d, splits = _volatility_fixture(cached_frame)
    forecast = session_after(cached_frame.index, cutoff, 5)
    _forbid_models(monkeypatch)
    monkeypatch.setattr(engine, "fit_variance", lambda *a, **k: pytest.fail("no tree model for this selection"))
    selected = dict(engine.benchmark_choice("volatility")) if arm == "ewma" else dict(
        arm="har", config="none", hp={}, iterations=0, shrinkage=1.0, combine="none", features=[])
    r = engine.repeat(_Ohlc(cached_frame), cutoff, forecast, "V8", 3, d, splits,
                      dict(selected=selected, gate=dict(pvalue=0.5)))
    origin, point = pd.Timestamp(cutoff), r["forecast"]
    # External forecast: the benchmark level comes from ALL matured labels, never from the cutoff row.
    level = d.y.loc[d.labeled].mean()
    assert origin not in d.labeled
    assert point["reference_variance"] == pytest.approx(level * 5 * d.sigma.loc[origin] ** 2, rel=1e-12)
    daily = daily_range_variance(cached_frame, "NVDA")
    position = cached_frame.index.get_loc(origin)
    assert point["actual_variance"] == pytest.approx(daily.iloc[position + 1: position + 6].sum(), rel=1e-12)
    assert point["target_date"] == forecast == str(cached_frame.index[position + 5].date())
    if arm == "ewma":
        assert point["predicted_variance"] == point["reference_variance"] and point["ties_reference"] == 1
        assert r["test_summary"]["test_loss"] == r["test_summary"]["test_reference_loss"]
    else:
        expected = engine.har_variance(d, d.labeled, pd.DatetimeIndex([cutoff]))[0]
        assert point["predicted_variance"] == pytest.approx(expected, rel=1e-12)
        assert r["test_summary"]["test_loss"] == pytest.approx(r["test_summary"]["test_har_loss"], rel=1e-12)
    assert "lower_90" not in point and "test_coverage_90" not in r["test_summary"]
    assert r["test_summary"]["loss_metric"] == "qlike" and np.isfinite(r["test_summary"]["test_har_loss"])
    assert (r["target"], r["horizon"]) == ("volatility", 5)
    p.digest(r)


def test_delayed_aci_matches_immediate_aci_at_one_session(cached_frame):
    cutoff = "2025-11-11"
    d = engine.make_design(cached_frame.loc[:cutoff], cutoff)
    splits = make_purged_splits(len(d.labeled), window=1)
    train, val, test = (d.labeled[getattr(splits, k)] for k in ("train", "val", "test"))
    low, high = np.quantile(d.y.loc[train], [0.05, 0.95])
    stream = val.append(test)
    y = d.y.loc[stream].to_numpy()
    lo, hi = np.full(len(stream), low), np.full(len(stream), high)
    scores = np.maximum(lo - y, y - hi)[:len(val)]
    expected_lo, expected_hi, _, _ = aci_bands(y[len(val):], lo[len(val):], hi[len(val):], scores, 0.10,
                                               gamma=p.ACI_GAMMA, alpha_clip=(1e-12, 1 - 1e-12))
    got_lo, got_hi, alphas = aci_bands_delayed(y, lo, hi, np.arange(len(stream)), len(val), 0.10, 1,
                                               gamma=p.ACI_GAMMA)
    np.testing.assert_allclose(got_lo[len(val):], expected_lo, rtol=1e-12)
    np.testing.assert_allclose(got_hi[len(val):], expected_hi, rtol=1e-12)
    assert np.isnan(got_lo[:len(val)]).all() and alphas[len(val)] == 0.10


@pytest.mark.parametrize("horizon", [1, 5, 21])
def test_price_bands_use_only_matured_outcomes(cached_frame, horizon):
    """Changing every outcome from one origin onward must not move any band whose
    origin lies less than h sessions after it: those outcomes were not yet observed.
    Where the protocol applies no calibration, the band must be the reference band."""
    import dataclasses
    cutoff = "2025-11-11"
    d = engine.make_design(cached_frame.loc[:cutoff], cutoff, horizon=horizon)
    splits = make_purged_splits(len(d.labeled), window=1)
    train, val, test = (d.labeled[getattr(splits, k)] for k in ("train", "val", "test"))
    dates = test.append(pd.DatetimeIndex([cutoff]))
    bands = engine.price_bands(d, train, val, dates)
    first = 40
    changed = d.y.copy()
    changed.loc[test[first:]] += 25.0                       # far outside any band
    moved = engine.price_bands(dataclasses.replace(d, y=changed), train, val, dates)
    unaffected = first + horizon
    for level in (90, 50):
        for side in ("lower", "upper"):
            np.testing.assert_array_equal(bands[level][side][:unaffected], moved[level][side][:unaffected])
            np.testing.assert_array_equal(bands[level]["raw_" + side], moved[level]["raw_" + side])
        if horizon in p.CONFORMAL_HORIZONS:
            assert not np.array_equal(bands[level]["upper"][unaffected:], moved[level]["upper"][unaffected:])
            assert not np.array_equal(bands[level]["upper"], bands[level]["raw_upper"])
        else:
            for side in ("lower", "upper"):
                np.testing.assert_array_equal(bands[level][side], bands[level]["raw_" + side])
                np.testing.assert_array_equal(moved[level][side], bands[level][side])
        assert (bands[level]["lower"] < bands[level]["upper"]).all()
        assert np.isfinite(bands[level]["upper"]).all()
    # The band of the cutoff origin exists although its outcome is not known yet.
    assert len(bands[90]["lower"]) == len(test) + 1 and pd.Timestamp(cutoff) not in d.labeled


@pytest.mark.parametrize("horizon", [5, 21])
def test_delayed_aci_never_uses_unobserved_outcomes(cached_frame, horizon):
    """The calibration routine itself, at the longer horizons (real returns)."""
    cutoff = "2025-11-11"
    d = engine.make_design(cached_frame.loc[:cutoff], cutoff, horizon=horizon)
    splits = make_purged_splits(len(d.labeled), window=1)
    train, val, test = (d.labeled[getattr(splits, k)] for k in ("train", "val", "test"))
    low, high = np.quantile(d.y.loc[train], [0.05, 0.95])
    stream = val.append(test)
    positions = d.close.index.get_indexer(stream)
    y = d.y.loc[stream].to_numpy()
    lo, hi = np.full(len(stream), low), np.full(len(stream), high)
    base = aci_bands_delayed(y, lo, hi, positions, len(val), 0.10, horizon, gamma=0.02)
    first = len(val) + 40
    changed = y.copy()
    changed[first:] += 25.0
    moved = aci_bands_delayed(changed, lo, hi, positions, len(val), 0.10, horizon, gamma=0.02)
    for a, b in zip(base, moved):
        np.testing.assert_array_equal(a[:first + horizon], b[:first + horizon])
    assert not np.array_equal(base[1][first + horizon:], moved[1][first + horizon:])
    # The purge gap makes every validation outcome observed before the first test origin.
    assert positions[len(val)] - positions[len(val) - 1] > horizon
    with pytest.raises(ValueError, match="növő"):
        aci_bands_delayed(y, lo, hi, positions[::-1], len(val), 0.10, horizon)


def _true_log_ratio(design, dates, level):
    return np.log(design.y.loc[dates].to_numpy() / level)


def test_volatility_fallback_selects_har_when_the_winner_does_not_replicate(cached_frame, monkeypatch):
    """The tree model is excellent on validation only (a "pocket"); HAR carries a
    modest edge everywhere. The winner fails stage 2, the pre-registered HAR
    fallback replicates and is selected at full weight."""
    cutoff, d, splits = _volatility_fixture(cached_frame)
    val = d.labeled[splits.val]
    def ratio(d, index, level):
        if index.isin(val).all():
            return level * np.exp(0.9 * _true_log_ratio(d, index, level))
        return np.full(len(index), level)
    result, _ = _volatility_selection(
        cached_frame, monkeypatch, ratio=ratio,
        har=lambda design, fit_dates, dates, level: 0.3 * _true_log_ratio(design, dates, level))
    gate = result["gate"]
    assert gate["stage_two_alpha"] == p.GATE_ALPHA / 2
    assert gate["best_candidate"]["arm"] == "baseline" and gate["validation_passed"] is True
    assert gate["replication"]["replicated"] is False
    assert gate["fallback"]["candidate"]["arm"] == "har" and gate["fallback"]["candidate"]["shrinkage"] == 1.0
    assert gate["fallback"]["replication"]["replicated"] is True
    assert gate["fallback"]["replication"]["alpha"] == p.GATE_ALPHA / 2
    assert gate["passed"] is True
    chosen = result["selected"]
    assert (chosen["arm"], chosen["shrinkage"], chosen["iterations"], chosen["features"]) == ("har", 1.0, 0, [])
    p.digest(result)


def test_volatility_fallback_keeps_benchmark_when_har_does_not_replicate_either(cached_frame, monkeypatch):
    cutoff, d, splits = _volatility_fixture(cached_frame)
    val = d.labeled[splits.val]
    def pocket(scale):
        def gain(design, index, level):
            return scale * _true_log_ratio(design, index, level) if index.isin(val).all() else np.zeros(len(index))
        return gain
    model = pocket(0.9)
    result, _ = _volatility_selection(
        cached_frame, monkeypatch, ratio=lambda d, index, level: level * np.exp(model(d, index, level)),
        har=lambda design, fit_dates, dates, level: pocket(0.3)(design, dates, level))
    gate = result["gate"]
    assert gate["validation_passed"] is True and gate["replication"]["replicated"] is False
    assert gate["fallback"]["replication"]["replicated"] is False
    assert gate["passed"] is False and result["selected"]["arm"] == "ewma"


def test_price_gate_has_no_fallback_and_keeps_its_level(cached_frame, monkeypatch):
    def pocket(design, dates, iterations, on_validation):
        return 0.5 * _true_log_returns(design, dates) if on_validation else np.zeros(len(dates))
    cutoff, d, splits, _ = _selection_fixture(cached_frame, monkeypatch, pocket)
    gate = engine.select(None, cutoff, "V8", d, splits)["gate"]
    assert gate["stage_two_alpha"] == p.GATE_ALPHA and gate["replication"]["alpha"] == p.GATE_ALPHA
    assert gate["fallback"] is None and gate["passed"] is False
