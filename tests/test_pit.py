"""V8-10 PIT-registry tesztek: (a) 100% oszlop-lefedettség a pipeline-kimeneten,
(b) as-of szimuláció — a t-napi jellemző a csak t-ig elérhető nyersadatból
újraszámolva bitre azonos (lag=0 oszlopok), (c) registry-integritás.

A lag_days=-1 (előre ismert: naptár, ütemezett earnings) oszlopok az as-of
prefix-összevetés alól FELMENTETTEK — az as-of szabályuk épp az, hogy jövőbeli
naptári tényt jogosan kódolnak; a lag_days>0 oszlopok lag-fegyelmét a
forrás-moduljuk saját tesztje bizonyítja (pl. test_macro release-lag).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from quantimate.data import pit
from quantimate.config import DEFAULT_CONFIG
from quantimate.data import load_universe_from_dir
from quantimate.features import build_features, generate_earnings_candidates
from quantimate.pipeline import EARNINGS_SIGMA, build_pipeline

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
REF_FILE = GOLDEN_DIR / "reference_hashes.json"

pytestmark = pytest.mark.skipif(
    not REF_FILE.exists(),
    reason="nincs golden-referencia — futtasd: scripts/generate_golden.py")


@pytest.fixture(scope="module")
def df():
    # A fagyasztott golden-adat jogi okból nincs a repóban, csak helyben.
    if not (GOLDEN_DIR / "data" / "NVDA.parquet").exists():
        pytest.skip("nincs fagyasztott golden-adat (tests/golden/data) ezen a gépen")
    ref = json.loads(REF_FILE.read_text(encoding="utf-8"))
    return load_universe_from_dir(GOLDEN_DIR / "data",
                                  DEFAULT_CONFIG.ohlcv_tickers,
                                  DEFAULT_CONFIG.light_tickers,
                                  end_date=ref["golden_end"])


def test_registry_sanity():
    """lookup/missing alapműködés + minden regisztrált lag értelmezett."""
    assert pit.lookup("NVDA_log_ret") is not None
    assert pit.lookup("regime_p1") is not None
    assert pit.lookup("nem_letezo_oszlop_xyz") is None
    assert pit.missing(["NVDA_rsi14", "fantom_jel"]) == ["fantom_jel"]
    for row in pit.registry_snapshot():
        assert row["lag_days"] >= -1, row
        assert row["source"] and row["asof_rule"], row


def test_unregistered_column_fails():
    with pytest.raises(AssertionError, match="fantom_jel"):
        pit.assert_all_registered(["NVDA_log_ret", "fantom_jel"])


def test_pipeline_columns_100pct_registered(df):
    """V8-10: a DEFAULT_CONFIG pipeline-kimenet MINDEN oszlopa regisztrált."""
    art = build_pipeline(df, DEFAULT_CONFIG, verbose=False)
    pit.assert_all_registered(art.feature_cols)


def test_optional_v7_columns_registered():
    """A flag mögötti V7-modulok oszlopnevei is le vannak fedve (use_macro,
    use_fracdiff, use_har_feature) — statikus névlista a hálózat elkerülésére."""
    pit.assert_all_registered([
        "vixterm_slope", "vixterm_contango", "vixterm_slope_long",
        "epu_level", "epu_chg5", "epu_dist_sma20",
        "NVDA_Close_log_ffd", "NVDA_rsi14_ffd", "har_sigma",
    ])


def test_asof_simulation_prefix_invariance(df):
    """(b) Véletlen napokon (fix seed) a jellemző-mátrix t-edik sora bitre
    azonos a t-ig CSONKOLT nyersadatból számolttal — minden lag=0 oszlopra.

    Az earnings-naptár rögzített INPUT (előre ismert ütemezés, lag=-1), nem a
    prefixből generáljuk újra; a lag=-1 oszlopok a bitre-összevetés alól a
    registry alapján felmentettek.
    """
    earn_dates = generate_earnings_candidates(
        df[f"{DEFAULT_CONFIG.target}_Close"], threshold_sigma=EARNINGS_SIGMA)
    feats_full = build_features(df, DEFAULT_CONFIG.target,
                                DEFAULT_CONFIG.ohlcv_tickers,
                                DEFAULT_CONFIG.light_tickers, earn_dates)
    pit.assert_all_registered(feats_full.columns)

    lag0_cols = [c for c in feats_full.columns if pit.lookup(c).lag_days == 0]
    assert len(lag0_cols) >= 40   # a jellemzők zöme ár-derivált

    rng = np.random.default_rng(42)
    positions = rng.integers(len(df) // 2, len(df) - 1, size=5)
    for p in positions:
        prefix = df.iloc[:p + 1]
        feats_pre = build_features(prefix, DEFAULT_CONFIG.target,
                                   DEFAULT_CONFIG.ohlcv_tickers,
                                   DEFAULT_CONFIG.light_tickers, earn_dates)
        a = feats_full[lag0_cols].iloc[p].to_numpy(dtype=float)
        b = feats_pre[lag0_cols].iloc[-1].to_numpy(dtype=float)
        bad = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        assert not bad.any(), (
            f"as-of sértés a(z) {p}. pozíción: "
            f"{[lag0_cols[i] for i in np.where(bad)[0]]}")
