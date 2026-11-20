"""QuantiMate — moduláris kvantitatív elemző és előrejelző csomag.

Alcsomagok:
- quantimate.data: Adatgyűjtés, Point-in-Time (PIT) regiszter, fundamentumok, earnings, intraday
- quantimate.features: Feature engineering, fracdiff, makro, geo, volatilitás, célváltozók, szentiment
- quantimate.models: Neurális modellek, GBDT, rezsimmodellek (HMM/Jump), ensemble, HPO, Bayes
- quantimate.validation: Konformális előrejelzés, CPCV, kalibráció, szivárgásmentes felosztások, XAI
- quantimate.backtest: Portfólió-szimuláció, végrehajtási költség, élő hash-lánc napló
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys

__version__ = "8.5.0"

_SUBMODULE_ALIASES: dict[str, str] = {
    # data
    "collect": "quantimate.data.collect",
    "pit": "quantimate.data.pit",
    "fundamentals": "quantimate.data.fundamentals",
    "earnings": "quantimate.data.earnings",
    "intraday": "quantimate.data.intraday",
    # features
    "fracdiff": "quantimate.features.fracdiff",
    "macro": "quantimate.features.macro",
    "geo": "quantimate.features.geo",
    "volatility": "quantimate.features.volatility",
    "targets": "quantimate.features.targets",
    "pooled": "quantimate.features.pooled",
    "news_history": "quantimate.features.news_history",
    "sentiment_engine": "quantimate.features.sentiment_engine",
    # models
    "models_gbdt": "quantimate.models.models_gbdt",
    "models_zeroshot": "quantimate.models.models_zeroshot",
    "regimes": "quantimate.models.regimes",
    "regimes_jump": "quantimate.models.regimes_jump",
    "bayes": "quantimate.models.bayes",
    "ensemble": "quantimate.models.ensemble",
    "metalabel": "quantimate.models.metalabel",
    "hpo": "quantimate.models.hpo",
    # validation
    "conformal": "quantimate.validation.conformal",
    "validation_cpcv": "quantimate.validation.validation_cpcv",
    "calibration": "quantimate.validation.calibration",
    "splits": "quantimate.validation.splits",
    "evaluation": "quantimate.validation.evaluation",
    "explain": "quantimate.validation.explain",
    # backtest
    "execution": "quantimate.backtest.execution",
    "live_ledger": "quantimate.backtest.live_ledger",
}


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, alias_fullname: str, canonical_fullname: str) -> None:
        self.alias_fullname = alias_fullname
        self.canonical_fullname = canonical_fullname
        self._orig_name: str | None = None
        self._orig_spec = None

    def create_module(self, spec):
        mod = importlib.import_module(self.canonical_fullname)
        self._orig_name = mod.__name__
        self._orig_spec = mod.__spec__
        return mod

    def exec_module(self, module):
        if self._orig_name is not None:
            module.__name__ = self._orig_name
        if self._orig_spec is not None:
            module.__spec__ = self._orig_spec
        sys.modules[self.canonical_fullname] = module
        sys.modules[self.alias_fullname] = module
        short_name = self.alias_fullname.rsplit(".", 1)[-1]
        pkg = sys.modules.get("quantimate")
        if pkg is not None:
            setattr(pkg, short_name, module)


class _SubmoduleAliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):
        if not fullname.startswith("quantimate."):
            return None
        rest = fullname[len("quantimate."):]
        if "." in rest or rest not in _SUBMODULE_ALIASES:
            return None
        canonical = _SUBMODULE_ALIASES[rest]
        loader = _AliasLoader(fullname, canonical)
        return importlib.machinery.ModuleSpec(fullname, loader)


if not any(isinstance(f, _SubmoduleAliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _SubmoduleAliasFinder())


def __getattr__(name: str):
    if name in _SUBMODULE_ALIASES:
        mod = importlib.import_module(_SUBMODULE_ALIASES[name])
        sys.modules[f"quantimate.{name}"] = mod
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'quantimate' has no attribute {name!r}")
