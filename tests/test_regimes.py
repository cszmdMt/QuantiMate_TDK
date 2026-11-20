"""Rezsim-modell, elfogadási kritériumok: a szűrt poszterior kauzális és mérhetően eltér
a simítotttól; a címkézés a vol+hozam szabályt követi."""
import numpy as np
import pytest

from quantimate.models.regimes import fit_hmm, hmm_filtered_proba, label_regimes


@pytest.fixture(scope="module")
def synthetic_hmm_data():
    """Szintetikus 2-állapotú rezsimváltós sor: nyugodt (+) és turbulens (−)."""
    rng = np.random.default_rng(42)
    T = 600
    states = np.zeros(T, dtype=int)
    for t in range(1, T):
        p_stay = 0.97 if states[t - 1] == 0 else 0.95
        states[t] = states[t - 1] if rng.random() < p_stay else 1 - states[t - 1]
    mu = np.array([0.001, -0.002])
    sig = np.array([0.01, 0.04])
    r = mu[states] + sig[states] * rng.normal(size=T)
    vol = np.sqrt(np.convolve(r ** 2, np.ones(10) / 10, mode="same"))
    X = np.column_stack([r, vol])
    return X


@pytest.fixture(scope="module")
def fitted(synthetic_hmm_data):
    X = synthetic_hmm_data
    model = fit_hmm(X[:400], n_components=2, seed=7)
    return model, X


def test_filtered_differs_from_smoothed(fitted):
    """(i) A szűrt és a hmmlearn-féle simított poszterior mérhetően eltér
    (max |Δ| > 0.05) — bizonyíték, hogy a predict_proba tényleg mást ad."""
    model, X = fitted
    filtered = hmm_filtered_proba(model, X)
    smoothed = model.predict_proba(X)
    assert filtered.shape == smoothed.shape
    max_diff = np.abs(filtered - smoothed).max()
    assert max_diff > 0.05, f"A szűrt és simított poszterior gyanúsan azonos (max|Δ|={max_diff:.4f})"


def test_filtered_is_causal_prefix_identical(fitted):
    """(ii) A t-ig tartó prefixen számolt szűrt poszterior BITRE azonos a
    teljes sorozaton számolt t-edik értékkel → a jövő nem folyik vissza."""
    model, X = fitted
    full = hmm_filtered_proba(model, X)
    for t in (1, 5, 50, 200, len(X) - 1):
        prefix = hmm_filtered_proba(model, X[: t + 1])
        np.testing.assert_array_equal(
            prefix[-1], full[t],
            err_msg=f"t={t}: a szűrt poszterior nem kauzális!")


def test_smoothed_is_not_causal_control(fitted):
    """Kontroll: a hmmlearn predict_proba (simított) NEM kauzális — a prefixen
    számolt érték eltér a teljes sorozatostól. Ez igazolja, hogy a V5 hibája
    valós volt, és hogy a tesztünk ténylegesen kauzalitást mér."""
    model, X = fitted
    full_smoothed = model.predict_proba(X)
    diffs = []
    for t in (50, 200, 399):
        prefix_smoothed = model.predict_proba(X[: t + 1])
        diffs.append(np.abs(prefix_smoothed[-1] - full_smoothed[t]).max())
    assert max(diffs) > 1e-6, "A simított poszterior kauzálisnak tűnik — a kontroll-teszt érvénytelen"


def test_rows_sum_to_one(fitted):
    model, X = fitted
    filtered = hmm_filtered_proba(model, X)
    np.testing.assert_allclose(filtered.sum(axis=1), 1.0, atol=1e-10)


def test_label_regimes_vol_and_return_rule(fitted):
    """A legmagasabb vol-ú állapot 'Turbulens'; K=2-nél a maradék hozam-címkét kap."""
    model, X = fitted
    order, names = label_regimes(model)
    assert names[-1] == "Turbulens"
    vols = np.sqrt(np.array([model.covars_[k][0, 0] for k in range(model.n_components)]))
    assert order[-1] == int(np.argmax(vols))
    assert len(order) == model.n_components
    assert len(set(order.tolist())) == model.n_components


# ── fit_hmm_scaled: a V8/V8.5 kísérleti út numerikusan stabil illesztése ─────

def _scaled_proba(X_fit, X_all):
    from quantimate.models.regimes import fit_hmm_scaled
    model, center, scale = fit_hmm_scaled(X_fit, n_components=2, seed=7)
    order, _ = label_regimes(model)
    return hmm_filtered_proba(model, (X_all - center) / scale)[:, order]


def test_scaled_fit_is_invariant_to_observation_units(synthetic_hmm_data):
    """A standardizálás lényege: a rezsim-valószínűség nem függhet a
    megfigyelések mértékegységétől (a nyers illesztésnél a hmmlearn 1e-3-as
    kovariancia-padlója miatt függ — ez okozta a 6. dátumpár hibáját)."""
    X = synthetic_hmm_data
    reference = _scaled_proba(X[:400], X)
    rescaled = X * np.array([1e-3, 250.0]) + np.array([5.0, -2.0])
    np.testing.assert_allclose(_scaled_proba(rescaled[:400], rescaled), reference, atol=1e-6)


def test_scaled_filter_is_causal_prefix_identical(synthetic_hmm_data):
    """A center/scale csak a tanító szakaszból jön, ezért a t-ig tartó
    prefixen számolt szűrt poszterior bitre azonos a teljes soron számolttal."""
    from quantimate.models.regimes import fit_hmm_scaled
    X = synthetic_hmm_data
    model, center, scale = fit_hmm_scaled(X[:400], n_components=2, seed=7)
    full = hmm_filtered_proba(model, (X - center) / scale)
    for t in (1, 50, 399, 450, len(X) - 1):
        prefix = hmm_filtered_proba(model, (X[: t + 1] - center) / scale)
        np.testing.assert_array_equal(prefix[-1], full[t])
    # A statisztika valóban a tanító szakaszé, nem a teljes soré.
    np.testing.assert_array_equal(center, X[:400].mean(axis=0))
    np.testing.assert_array_equal(scale, X[:400].std(axis=0))


def test_scaled_fit_is_deterministic(synthetic_hmm_data):
    X = synthetic_hmm_data
    np.testing.assert_array_equal(_scaled_proba(X[:400], X), _scaled_proba(X[:400], X))


def test_scaled_fit_falls_back_to_diagonal_then_fails_loudly(synthetic_hmm_data, monkeypatch):
    from hmmlearn.hmm import GaussianHMM
    from quantimate.models.regimes import fit_hmm_scaled
    original = GaussianHMM.fit

    def full_is_singular(self, X, lengths=None):
        if self.covariance_type == "full":
            raise ValueError("'covars' must be symmetric, positive-definite")
        return original(self, X, lengths)

    monkeypatch.setattr(GaussianHMM, "fit", full_is_singular)
    model, _, _ = fit_hmm_scaled(synthetic_hmm_data[:400], n_components=2, seed=7)
    assert model.covariance_type == "diag"
    # A szűrő a diagonális modell (K, D, D) alakú covars_ nézetével is működik.
    assert hmm_filtered_proba(model, synthetic_hmm_data).shape == (600, 2)

    def always_singular(self, X, lengths=None):
        raise ValueError("'covars' must be symmetric, positive-definite")

    monkeypatch.setattr(GaussianHMM, "fit", always_singular)
    with pytest.raises(ValueError, match="sikertelen"):
        fit_hmm_scaled(synthetic_hmm_data[:400], n_components=2, seed=7)


def test_scaled_fit_rejects_constant_column(synthetic_hmm_data):
    from quantimate.models.regimes import fit_hmm_scaled
    X = synthetic_hmm_data[:400].copy()
    X[:, 1] = 0.01
    with pytest.raises(ValueError, match="konstans"):
        fit_hmm_scaled(X, n_components=2, seed=7)
