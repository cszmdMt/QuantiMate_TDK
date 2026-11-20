"""Golden-run teszt: a V6-os flag-kombináció bitre reprodukálja a V6
artefaktumait — minden fázis-zárás előfeltétele.

A teszt FAGYASZTOTT adatból dolgozik (tests/golden/data/*.parquet, fix
záródátum), így hálózat- és cache-független. Ha a hash eltér, annak két oka
lehet: (1) kód-regresszió — javítandó; (2) a golden-adat tudatos újragenerálása
forrás-revízió után — ehhez tudatos döntés kell, a referencia nem
generálható újra automatikusan.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from quantimate.config import DATA_DIR, DEFAULT_CONFIG
from quantimate.data import load_universe_from_dir
from quantimate.pipeline import build_pipeline

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
REF_FILE = GOLDEN_DIR / "reference_hashes.json"

GOLDEN_DATA = GOLDEN_DIR / "data"

# A fagyasztott golden-adat (Yahoo-árfolyamok) jogi okból nincs a repóban, csak helyben;
# nélküle a teszt kimarad (frissen letöltött adatból a fagyasztott hash nem reprodukálható).
pytestmark = pytest.mark.skipif(
    not REF_FILE.exists() or not (GOLDEN_DATA / "NVDA.parquet").exists(),
    reason="nincs golden-referencia vagy fagyasztott golden-adat (tests/golden/data) ezen a gépen")


def sha(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@pytest.fixture(scope="module")
def ref():
    return json.loads(REF_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def art(ref):
    df = load_universe_from_dir(GOLDEN_DATA,
                                DEFAULT_CONFIG.ohlcv_tickers,
                                DEFAULT_CONFIG.light_tickers,
                                end_date=ref["golden_end"])
    return build_pipeline(df, DEFAULT_CONFIG, verbose=False)


MSG = ("Golden-eltérés! Ha a quantimate/ kódját módosítottad, ez REGRESSZIÓ a "
       "V6-ekvivalens úton. Ha a tests/golden/data tartalma "
       "változott (forrás-adat revízió), emberi döntés kell az újrageneráláshoz.")


def test_pipeline_shapes_and_splits(art, ref):
    assert list(art.feats.shape) == ref["shape_feats"], MSG
    assert art.n_labeled == ref["n_labeled"], MSG
    for k in ("train", "val", "test"):
        blk = getattr(art.S, k)
        assert [int(blk.min()), int(blk.max()), len(blk)] == ref["splits"][k], MSG


def test_pipeline_artifact_hashes(art, ref):
    """Nem-rezsim oszlopok: EXAKT hash. Rezsim-oszlopok: toleranciás
    npy-referencia — a HMM-EM folyamatok közti BLAS-zaja ~1e-12,
    ami hash-t tör, de numerikusan irreleváns."""
    n_reg = ref["n_regime_cols"]
    feats_mat = art.feats.to_numpy(dtype=float)
    assert sha(feats_mat[:, :-n_reg]) == ref["sha_feats_nonregime"], MSG
    assert sha(art.X_scaled[:, :-n_reg]) == ref["sha_X_scaled_nonregime"], MSG
    assert sha(np.nan_to_num(art.Y_all, nan=-9e9)) == ref["sha_Y_all"], MSG
    assert sha(np.concatenate([art.S.train, art.S.val, art.S.test])) == \
        ref["sha_splits"], MSG
    assert art.regime_names == ref["regime_names"], MSG

    ref_filtered = np.load(GOLDEN_DIR / "filtered_o.npy")
    np.testing.assert_allclose(art.filtered_o, ref_filtered, atol=1e-9,
                               err_msg=MSG + " (rezsim-valószínűségek 1e-9 felett térnek el)")


def test_frozen_v6_winner_val_pinball(art, ref):
    """(b) ág: a fagyasztott V6-győztes visszatöltve a tárolt val-pinballt adja."""
    if ref.get("winner_val_pinball") is None:
        pytest.skip("nincs tárolt győztes-referencia")
    # VÉDETT könyvtár: a data/models_v6-ot a notebook-futások felülírják
    # (a SMOKE is!) — a fagyasztott győztes a models_v6_frozen-ben él, amit
    # semmilyen futás nem ír.
    winner_w = DATA_DIR / "models_v6_frozen" / "LSTM_stacked_s42_r0.weights.h5"
    if not winner_w.exists():
        pytest.skip("nincs fagyasztott V6-győztes ezen a gépen (data/ gitignored)")

    import tensorflow as tf
    from quantimate.models import build_lstm, pinball_loss_np, predict_quantiles
    from quantimate.validation.splits import make_sequences

    tf.keras.backend.clear_session()
    H, Q = DEFAULT_CONFIG.horizons, DEFAULT_CONFIG.quantiles
    model = build_lstm((DEFAULT_CONFIG.window, art.X_scaled.shape[1]),
                       len(H) * len(Q), use_revin=False)
    model.load_weights(winner_w)
    X_va = make_sequences(art.X_scaled, art.S.val, DEFAULT_CONFIG.window)
    q_va = predict_quantiles(model, X_va, n_horizons=len(H), n_quantiles=len(Q))
    val_pin = float(pinball_loss_np(art.Y_all[art.S.val], q_va, quantiles=Q))
    # 1e-7: a rezsim-oszlopok ~1e-12-es HMM-zaja a hálón átfolyva ennél
    # kisebb pinball-ingadozást ad; e felett valódi regresszió.
    assert val_pin == pytest.approx(ref["winner_val_pinball"], abs=1e-7), MSG


def test_default_config_is_v6_equivalent():
    """Őr: az alapértelmezett flagek V6-ekvivalens állapotot írnak le —
    aki V7-blokkot alapból bekapcsol, annak ez a teszt törik el először."""
    c = DEFAULT_CONFIG
    assert c.sigma_mode == "ewma"
    assert c.use_gbdt is False and c.use_ensemble is False
    assert c.mondrian is False and c.conformal_method == "aci"
    assert c.use_fracdiff is False and c.use_macro is False and c.xai is False
