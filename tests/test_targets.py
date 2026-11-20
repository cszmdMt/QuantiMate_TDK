"""Célváltozó, elfogadási kritériumok: target-igazítás bitpontosan, off-by-one mindkét
irányban, σ-normalizálás visszaalakítása."""
import numpy as np
import pandas as pd
import pytest

from quantimate.features import ewma_volatility
from quantimate.features.targets import (build_targets, denormalize, forward_cum_return,
                                reconstruct_price_path)


@pytest.fixture()
def hand_series():
    """Kézzel konstruált 30 elemű ársor determinisztikus, egyedi hozamokkal."""
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.02, 30)
    r[0] = 0.0
    idx = pd.bdate_range("2024-01-01", periods=30)
    log_ret = pd.Series(r, index=idx)
    return log_ret


@pytest.mark.parametrize("h", [1, 5, 21])
def test_forward_cum_return_exact(hand_series, h):
    """y_{t,h} nevezője előtti összeg bitpontosan Σ_{i=1..h} r_{t+i}."""
    fwd = forward_cum_return(hand_series, h)
    r = hand_series.values
    for t in range(len(r) - h):
        expected = r[t + 1: t + h + 1].sum()
        assert fwd.iloc[t] == pytest.approx(expected, abs=1e-15), f"t={t}, h={h}"
    # Az utolsó h napra nincs teljes jövő → NaN
    assert fwd.iloc[-h:].isna().all()


def test_off_by_one_both_directions(hand_series):
    """Off-by-one bizonyíték mindkét irányban: az r_t-t TARTALMAZÓ vagy az
    r_{t+h+1}-ig CSÚSZÓ összeg nem egyezik a helyes targettel."""
    h = 5
    fwd = forward_cum_return(hand_series, h).values
    r = hand_series.values
    t = 10
    correct = r[t + 1: t + h + 1].sum()
    too_early = r[t: t + h].sum()        # r_t-t tartalmazza (múltba csúszott)
    too_late = r[t + 2: t + h + 2].sum() # r_{t+1}-et kihagyja (jövőbe csúszott)
    assert fwd[t] == pytest.approx(correct, abs=1e-15)
    assert abs(fwd[t] - too_early) > 1e-12
    assert abs(fwd[t] - too_late) > 1e-12


def test_sigma_normalisation_roundtrip(hand_series):
    """A normált target σ_t·√h-val visszaszorozva bitre a nyers hozamösszeg."""
    sigma = ewma_volatility(hand_series)
    tg = build_targets(hand_series, sigma, horizons=(1, 5, 21))
    for h in (1, 5, 21):
        y = tg[f"y_h{h}"].values
        raw = tg[f"fwd_ret_h{h}"].values
        back = denormalize(y, sigma.clip(lower=1e-8).values, h)
        mask = ~np.isnan(raw)
        np.testing.assert_allclose(back[mask], raw[mask], rtol=1e-12)


def test_direction_label_matches_h1_sign(hand_series):
    sigma = ewma_volatility(hand_series)
    tg = build_targets(hand_series, sigma, horizons=(1, 5, 21))
    mask = tg["fwd_ret_h1"].notna()
    assert ((tg.loc[mask, "fwd_ret_h1"] > 0).astype(int)
            == tg.loc[mask, "direction"]).all()


def test_reconstruction_follows_predicted_sign():
    """A rekonstruált pálya a jósolt hozamok előjelét követi — esésre esik."""
    med = np.array([-0.01, -0.02, 0.005, -0.015])
    path = reconstruct_price_path(100.0, med)
    assert path[0] < 100.0            # első nap: negatív jóslat → esik
    assert path[1] < path[0]
    assert path[2] > path[1]          # pozitív jóslat → emelkedik
    assert path[3] < path[2]
    np.testing.assert_allclose(path[-1], 100.0 * np.exp(med.sum()), rtol=1e-12)
