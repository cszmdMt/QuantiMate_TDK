"""PIT / release-lag registry (V8-10) — a look-ahead elleni védelem formalizálása.

Minden jellemző-oszlop, ami a pipeline kimenetébe (feats/feature_cols) kerülhet,
KÖTELEZŐEN regisztrált itt: forrás, elérhetőségi lag (kereskedési napban) és
szöveges as-of szabály. A tests/test_pit.py két irányból ellenőrzi:
  (a) a pipeline-kimenet MINDEN oszlopa regisztrált (lefedettség 100%);
  (b) as-of szimuláció: a t-napi sorban álló jellemző bitre azonos a csak
      t-ig elérhető nyersadatból újraszámolt értékkel (véletlen napok, fix seed).

Lag-konvenció:
  lag_days = 0  → a t-napi ZÁRÁSKOR ismert (piaci ár-derivált jellemzők);
  lag_days = k  → az érték csak k kereskedési nappal a vonatkozási nap után
                  ismert — a jellemzőt építő modul felelőssége a pozitív shift
                  (lásd macro.apply_release_lag), a registry ezt DOKUMENTÁLJA
                  és a tesztek számon kérik;
  lag_days = -1 → előre ismert, determinisztikus tény (naptár, ütemezett
                  earnings-dátum) — nem szivároghat.

Új V8-modul (intraday/fundamentals/geo) minden oszlopa ide regisztrál a modul
importjakor — regisztráció nélkül a pipeline-teszt bukik.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PITMeta:
    """Egy jellemző-oszlop(család) point-in-time metaadata."""
    source: str      # honnan jön a nyersadat (pl. "yahoo_daily")
    lag_days: int    # elérhetőségi lag kereskedési napban (lásd modul-docstring)
    asof_rule: str   # hogyan áll elő a t-napi érték csak t-ig elérhető adatból
    notes: str = ""


# (minta, meta) párok — az ELSŐ találat nyer, a minta a teljes oszlopnévre illik.
_REGISTRY: list[tuple[re.Pattern, PITMeta]] = []


def register(pattern: str, source: str, lag_days: int, asof_rule: str,
             notes: str = "") -> None:
    """Oszlopnév-minta regisztrálása. A minta implicit teljes illesztésű."""
    _REGISTRY.append((re.compile(pattern),
                      PITMeta(source, lag_days, asof_rule, notes)))


def lookup(column: str) -> PITMeta | None:
    """Az első illeszkedő regisztráció metaadata, vagy None."""
    for pat, meta in _REGISTRY:
        if pat.fullmatch(column):
            return meta
    return None


def missing(columns) -> list[str]:
    """A nem regisztrált oszlopok listája (sorrendtartó)."""
    return [c for c in columns if lookup(c) is None]


def assert_all_registered(columns) -> None:
    """A pipeline-teszt őre: minden oszlop regisztrált, különben hibaüzenet
    a hiányzók listájával (V8-10: 100% lefedettség)."""
    miss = missing(columns)
    assert not miss, (
        f"PIT-registry: {len(miss)} nem regisztrált jellemző-oszlop: {miss}. "
        f"Regisztráld a quantimate/pit.py-ban (forrás, lag, as-of szabály).")


def registry_snapshot() -> list[dict]:
    """A teljes registry riport-barát formában (forrás-leltárhoz)."""
    return [{"pattern": pat.pattern, **meta.__dict__} for pat, meta in _REGISTRY]


# ─────────────────────────────────────────────────────────────────────────────
# V7-örökség: a meglévő pipeline-oszlopok visszamenőleges regisztrálása.
# Minden ár-derivált jellemző kauzális rolling/EWM ablakból számolódik —
# a t-napi érték a t-napi zárásig elérhető adatból áll elő (lag=0), ezt a
# prefix-invariancia (as-of szimuláció) teszt bizonyítja.
# ─────────────────────────────────────────────────────────────────────────────

_TICK = r"(NVDA|SPY|QQQ|AMD|SMH)"
_IDX = r"(VIX|VXN|TNX)"
_CAUSAL = "kauzális rolling/EWM a t-napi zárásig — prefix-invariáns"

# OHLCV-tickerek indikátorai (NVDA teljes + társak könnyű készlete)
register(_TICK + r"_(log_ret|ewma_vol|dist_sma20|dist_sma50|rsi14|macd_hist"
         r"|boll_pctB|boll_bw|atr_norm|adx14|di_spread|stoch_k|stoch_d"
         r"|roc10|obv_z|vol_z)",
         source="yahoo_daily", lag_days=0, asof_rule=_CAUSAL,
         notes="features.compute_indicators; osztalék-korrigált OHLCV")

# Volume nélküli indexek (VIX/VXN/TNX) könnyű készlete
register(_IDX + r"_(level|chg5|dist_sma20)",
         source="yahoo_daily", lag_days=0, asof_rule=_CAUSAL,
         notes="features.compute_index_features; záró szint aznap ismert")

# Overnight/intraday felbontás (B3e)
register(_TICK + r"_r_(overnight|intraday)",
         source="yahoo_daily", lag_days=0,
         asof_rule="r_on=log(O_t/C_{t-1}), r_id=log(C_t/O_t) — t-napon ismert")

# Naptári jellemzők — determinisztikus, előre ismert tények
register(r"cal_(dow_sin|dow_cos|month_end|opex_friday)",
         source="naptár", lag_days=-1,
         asof_rule="determinisztikus naptári tény, előre ismert — nem szivárog")

# Earnings-jellemzők — az ütemezett dátum ELŐRE nyilvános (lag=-1);
# a historikus árfolyamból GENERÁLT jelöltnaptár proxy-korlátja dokumentált,
# a V8-4 hitelesített naptár váltja ki.
register(r"earn_(days_to_next|days_since_prev|week)",
         source="earnings-naptár (V6: generált jelöltek; V8-4: hitelesített CSV)",
         lag_days=-1,
         asof_rule="ütemezett bejelentési dátum hetekkel előre nyilvános",
         notes="generált jelöltnél a proxy a jelentést KÖVETŐ mozgás napja — "
               "a V8-4 hitelesítés zárja ezt a rést")

# Rezsim-valószínűségek — HMM fit CSAK train-en, kauzális (filtered) szűrés
register(r"regime_p[0-2]",
         source="HMM(log_ret, ewma_vol)", lag_days=0,
         asof_rule="train-only fit + kauzális forward-szűrés — prefix-invariáns "
                   "(test_regimes)",
         notes="a jump-modell A/B ugyanezeket az oszlopokat cseréli (V7-8)")

# HAR-σ jellemző (flag: use_har_feature) — train-only illesztés
register(r"har_sigma",
         source="Yang–Zhang σ + HAR-regresszió", lag_days=0,
         asof_rule="HAR-fit a train-szakaszon, előrejelzés kauzális lagekből")

# Frakcionális differenciálás (flag: use_fracdiff)
register(r".*_ffd",
         source="származtatott (FFD)", lag_days=0,
         asof_rule="rögzített súlyú, csak múltra néző FFD-konvolúció; d* train-only")

# Makro-jellemzők (flag: use_macro) — release-lag a macro.py-ban
register(r"vixterm_(slope|contango|slope_long)",
         source="yahoo_daily (VIX-terminstruktúra)", lag_days=0,
         asof_rule="piaci záró adat, aznap ismert; ffill(limit=3)")
register(r"epu_(level|chg5|dist_sma20)",
         source="FRED USEPUINDXD", lag_days=1,
         asof_rule="apply_release_lag(+1 kereskedési nap) a macro.py-ban; "
                   "a napi EPU materiálisan nem revideálódik")
