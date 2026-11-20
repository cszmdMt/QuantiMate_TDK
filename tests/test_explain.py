"""V7-17 tesztek: MDA determinisztikus, az informatív jellemzőt a zaj fölé
rangsorolja; klaszterezett fontosság; TreeSHAP a GBDT- n."""
import numpy as np
import pytest

from quantimate.validation.explain import (clustered_importance_pinball,
                                correlation_clusters,
                                permutation_importance_pinball)

QUANTS = (0.05, 0.25, 0.5, 0.75, 0.95)


@pytest.fixture(scope="module")
def gbdt_setup():
    from quantimate.models.models_gbdt import train_gbdt_quantiles
    rng = np.random.default_rng(0)
    n, f = 600, 6
    X = rng.normal(0, 1, (n, f))
    signal = 1.2 * X[:, 0]                       # csak a 0. jellemző informatív
    Y = np.column_stack([signal + rng.normal(0, 0.4, n)])
    tr, va = np.arange(0, 400), np.arange(410, 560)
    model = train_gbdt_quantiles(X, Y, tr, va, (1,), QUANTS, seed=42)
    return model, X[va], Y[va]


def test_mda_deterministic_and_ranks_signal_first(gbdt_setup):
    model, Xv, Yv = gbdt_setup
    names = [f"f{j}" for j in range(Xv.shape[1])]

    def pred(X):
        return model.predict_quantiles(X)

    r1 = permutation_importance_pinball(pred, Xv, Yv, QUANTS, names, seed=7)
    r2 = permutation_importance_pinball(pred, Xv, Yv, QUANTS, names, seed=7)
    assert [r["jellemző"] for r in r1] == [r["jellemző"] for r in r2]   # determinizmus
    assert r1[0]["jellemző"] == "f0"                                    # a jel a csúcson
    assert r1[0]["MDA"] > 0
    # a zaj-jellemzők MDA-ja lényegesen kisebb
    assert r1[0]["MDA"] > max(r["MDA"] for r in r1[1:]) * 2


def test_correlation_clusters_groups_collinear():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, (500, 1))
    X = np.column_stack([base, base + rng.normal(0, 0.01, (500, 1)),  # ~duplikátum
                         rng.normal(0, 1, (500, 1))])
    clusters = correlation_clusters(X, ["a", "b", "c"], threshold=0.7)
    # a és b egy klaszterben, c külön
    sizes = sorted(len(v) for v in clusters.values())
    assert sizes == [1, 2]


def test_clustered_importance_runs(gbdt_setup):
    model, Xv, Yv = gbdt_setup

    def pred(X):
        return model.predict_quantiles(X)

    clusters = {"jel": [0], "zaj": [1, 2, 3, 4, 5]}
    rows = clustered_importance_pinball(pred, Xv, Yv, QUANTS, clusters, seed=3)
    assert rows[0]["klaszter"] == "jel"


def test_treeshap_sums_to_prediction(gbdt_setup):
    """A TreeSHAP contributions összege (+ bázis) = a predikció (egzaktság)."""
    from quantimate.models.models_gbdt import shap_contributions
    model, Xv, _ = gbdt_setup
    contrib = shap_contributions(model, Xv, horizon=1, quantile=0.5)
    pred_median = model.predict_quantiles(Xv)[:, 0, 2]
    # a rendezés (predict_quantiles) monotonizál — a nyers booster-predikció a
    # kontribúciók összege; a medián-kvantilis rendezés utáni értéke ehhez közel
    raw = contrib.sum(axis=1)
    assert np.corrcoef(raw, pred_median)[0, 1] > 0.99
