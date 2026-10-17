"""A2: historical NFHS-4 state / facility-sector cesarean context for NFHS-5 births.

The mapping is fitted on NFHS-4 labelled births ONLY and then frozen. Its
transform reads exactly two target columns (source_unit, sector), so NFHS-5
outcomes cannot reach feature construction. Rates are descriptive historical
source context, not causal sector effects or prenatal predictors.

Smoothing (pseudo-count alpha, predeclared; no tuning on target labels):
  global  g = events / n
  sector  r_k = (events_k + alpha*g) / (n_k + alpha)
  state   r_s = (events_s + alpha*g) / (n_s + alpha)
  cell    r_sk = (events_sk + alpha*prior_sk) / (n_sk + alpha)
          prior_sk = r_s if state supported, else r_k if sector supported, else g
A level is "supported" when its support (births, or Kish effective n when
weighted) is >= n_min. Transform fallback order (fixed in code):
  state feature: supported state -> global
  cell feature:  supported cell -> supported state -> supported sector -> global
Unsupported, zero-count and unseen cells therefore never yield raw tiny-cell rates.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import numpy as np
import pandas as pd

SOURCE_WAVE = "NFHS-4"
SECTORS = ("public", "private", "other")
STATE_RATE = "hist_state_cs_rate"
CELL_RATE = "hist_state_sector_cs_rate"
SUPPORT_FEATURES = ("hist_state_support_log", "hist_cell_support_log",
                    "hist_state_unsupported", "hist_cell_unsupported")
FEATURE_NAMES = (STATE_RATE, CELL_RATE, *SUPPORT_FEATURES)
DIAGNOSTIC_NAMES = ("hist_state_level", "hist_cell_level")


@dataclass(frozen=True)
class SmoothingSpec:
    alpha: float = 20.0
    n_min: float = 30.0
    weighted: bool = False

    def __post_init__(self):
        if not (np.isfinite(self.alpha) and self.alpha >= 0):
            raise ValueError("alpha must be finite and >= 0")
        if not (np.isfinite(self.n_min) and self.n_min >= 1):
            raise ValueError("n_min must be >= 1")


def _summarise(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = frame.groupby(keys, sort=True, observed=True) if keys else frame.assign(_k=0).groupby("_k")
    out = pd.DataFrame({
        "n_births": g["y"].size(), "n_events": g["y"].sum(),
        "w_sum": g["w"].sum(), "w_events": g["wy"].sum(), "w_sq": g["w2"].sum(),
    })
    out["kish_n"] = out["w_sum"] ** 2 / out["w_sq"]
    return out


class HistoricalContextMapping:
    """fit(NFHS-4 source) once; transform(target units/sectors) many times."""

    def __init__(self, spec: SmoothingSpec = SmoothingSpec()):
        self.spec = spec
        self._tables = None

    # ------------------------------------------------------------------ fit
    def fit(self, source: pd.DataFrame, *, unit_col="source_unit", sector_col="sector",
            outcome_col="m17", wave_col="wave", weight_col="sample_weight"):
        if self._tables is not None:
            raise RuntimeError("Mapping is frozen; refit requires a new instance")
        required = [wave_col, unit_col, sector_col, outcome_col] + ([weight_col] if self.spec.weighted else [])
        absent = [c for c in required if c not in source.columns]
        if absent:
            raise ValueError(f"Missing source columns: {absent}")
        if source.empty:
            raise ValueError("Empty source frame")
        waves = set(source[wave_col].astype(str))
        if waves != {SOURCE_WAVE} or source[wave_col].isna().any():
            raise ValueError(f"LEAKAGE: historical context may only be fitted on {SOURCE_WAVE} rows; got {sorted(waves)}")
        unit = source[unit_col].astype("string")
        if unit.isna().any() or unit.str.strip().eq("").any():
            raise ValueError("Missing source unit in NFHS-4 source rows")
        sector = source[sector_col].astype("string")
        if sector.isna().any() or not sector.isin(SECTORS).all():
            raise ValueError("Invalid sector in source rows")
        y = pd.to_numeric(source[outcome_col], errors="coerce")
        if y.isna().any() or not y.isin([0, 1]).all():
            raise ValueError("Source outcome must be nonmissing 0/1")
        if self.spec.weighted:
            w = pd.to_numeric(source[weight_col], errors="coerce").to_numpy(float)
            if not (np.isfinite(w).all() and (w > 0).all()):
                raise ValueError("Source weights must be finite and positive")
            w = w / w.mean()  # pseudo-counts on the same scale as alpha
        else:
            w = np.ones(len(source))
        df = pd.DataFrame({"unit": unit.to_numpy(), "sector": sector.to_numpy(),
                           "y": y.to_numpy(int), "w": w})
        df["wy"] = df["w"] * df["y"]
        df["w2"] = df["w"] ** 2
        alpha, n_min = float(self.spec.alpha), float(self.spec.n_min)

        def rate_parts(t):
            if self.spec.weighted:
                return t["w_events"], t["w_sum"], t["kish_n"]
            return t["n_events"].astype(float), t["n_births"].astype(float), t["n_births"].astype(float)

        glob = _summarise(df, [])
        ge, gn, _ = rate_parts(glob)
        g = float(ge.iloc[0] / gn.iloc[0])

        def smooth(events, n, prior):
            return (events + alpha * prior) / (n + alpha)

        sec = _summarise(df, ["sector"])
        e, n, sup = rate_parts(sec)
        sec["rate"] = smooth(e, n, g)
        sec["support"] = sup
        sec["supported"] = sup >= n_min

        st = _summarise(df, ["unit"])
        e, n, sup = rate_parts(st)
        st["rate"] = smooth(e, n, g)
        st["support"] = sup
        st["supported"] = sup >= n_min

        cell = _summarise(df, ["unit", "sector"])
        e, n, sup = rate_parts(cell)
        units = cell.index.get_level_values("unit")
        sectors = cell.index.get_level_values("sector")
        prior = [self._prior_from(st, sec, g, u, k) for u, k in zip(units, sectors)]
        cell["prior"] = prior
        cell["rate"] = smooth(e, n, np.asarray(prior))
        cell["support"] = sup
        cell["supported"] = sup >= n_min

        def pack(t, cols):
            return {(k if isinstance(k, str) else "|".join(k)): {c: (bool(v[c]) if c == "supported" else
                    (int(v[c]) if c in ("n_births", "n_events") else float(v[c]))) for c in cols}
                    for k, v in t.iterrows()}

        cols = ["n_births", "n_events", "support", "rate", "supported"]
        self._tables = {
            "spec": {"alpha": alpha, "n_min": n_min, "weighted": bool(self.spec.weighted)},
            "source_wave": SOURCE_WAVE,
            "global": {"n_births": int(glob["n_births"].iloc[0]), "n_events": int(glob["n_events"].iloc[0]),
                       "rate": g},
            "sector": pack(sec, cols),
            "state": pack(st, cols),
            "cell": pack(cell, cols + ["prior"]),
        }
        missing_sectors = set(SECTORS) - set(self._tables["sector"])
        for k in missing_sectors:  # unseen sector in source: unsupported, falls to global
            self._tables["sector"][k] = {"n_births": 0, "n_events": 0, "support": 0.0, "rate": g,
                                         "supported": False}
        return self

    @staticmethod
    def _prior_from(st, sec, g, unit, sector):
        if unit in st.index and bool(st.at[unit, "supported"]):
            return float(st.at[unit, "rate"])
        if sector in sec.index and bool(sec.at[sector, "supported"]):
            return float(sec.at[sector, "rate"])
        return g

    # ------------------------------------------------------------ transform
    def transform(self, target: pd.DataFrame, *, unit_col="source_unit", sector_col="sector") -> pd.DataFrame:
        """Map target rows to frozen historical context. Reads ONLY unit and sector."""
        if self._tables is None:
            raise ValueError("Fit on NFHS-4 source before transform")
        for c in (unit_col, sector_col):
            if c not in target.columns:
                raise ValueError(f"Missing target column: {c}")
        keys = target[[unit_col, sector_col]]
        unit = keys[unit_col].astype("string")
        sector = keys[sector_col].astype("string")
        if sector.isna().any() or not sector.isin(SECTORS).all():
            raise ValueError("Unknown or missing target sector; sector vocabulary is fixed")
        t = self._tables
        g = t["global"]["rate"]
        st_rate, st_lvl, st_sup = [], [], []
        c_rate, c_lvl, c_sup = [], [], []
        for u, k in zip(unit.tolist(), sector.tolist()):
            s = t["state"].get(u) if isinstance(u, str) else None  # <NA> = declared no-source unit
            sk = t["sector"][k]
            st_sup.append(s["support"] if s else 0.0)
            if s and s["supported"]:
                st_rate.append(s["rate"]); st_lvl.append("state")
            else:
                st_rate.append(g); st_lvl.append("global")
            c = t["cell"].get(f"{u}|{k}") if s else None
            c_sup.append(c["support"] if c else 0.0)
            if c and c["supported"]:
                c_rate.append(c["rate"]); c_lvl.append("cell")
            elif s and s["supported"]:
                c_rate.append(s["rate"]); c_lvl.append("state")
            elif sk["supported"]:
                c_rate.append(sk["rate"]); c_lvl.append("sector")
            else:
                c_rate.append(g); c_lvl.append("global")
        out = pd.DataFrame(index=target.index)
        out[STATE_RATE] = np.asarray(st_rate, dtype=float)
        out[CELL_RATE] = np.asarray(c_rate, dtype=float)
        out["hist_state_support_log"] = np.log1p(np.asarray(st_sup, dtype=float))
        out["hist_cell_support_log"] = np.log1p(np.asarray(c_sup, dtype=float))
        out["hist_state_unsupported"] = (np.asarray(st_lvl) != "state").astype(float)
        out["hist_cell_unsupported"] = (np.asarray(c_lvl) != "cell").astype(float)
        out["hist_state_level"] = st_lvl
        out["hist_cell_level"] = c_lvl
        if not np.isfinite(out[list(FEATURE_NAMES)].to_numpy(float)).all():
            raise RuntimeError("Non-finite historical context feature")
        return out

    # ----------------------------------------------------------- provenance
    def mapping_hash(self) -> str:
        """Bitwise SHA-256 of the frozen tables (floats serialised via float.hex)."""
        if self._tables is None:
            raise ValueError("Mapping not fitted")

        def enc(x):
            if isinstance(x, dict):
                return {k: enc(v) for k, v in x.items()}
            if isinstance(x, float):
                return float(x).hex()
            return x
        payload = json.dumps(enc(self._tables), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def aggregate_tables(self) -> dict[str, pd.DataFrame]:
        """Source aggregate tables (apply small-cell suppression before export)."""
        if self._tables is None:
            raise ValueError("Mapping not fitted")
        t = self._tables
        state = pd.DataFrame.from_dict(t["state"], orient="index").rename_axis("source_unit").reset_index()
        sector = pd.DataFrame.from_dict(t["sector"], orient="index").rename_axis("sector").reset_index()
        cell = pd.DataFrame.from_dict(t["cell"], orient="index").rename_axis("key").reset_index()
        cell[["source_unit", "sector"]] = cell["key"].str.split("|", expand=True)
        cell = cell.drop(columns="key")[["source_unit", "sector", *[c for c in cell.columns
                                                                    if c not in ("key", "source_unit", "sector")]]]
        return {"global": pd.DataFrame([t["global"]]), "sector": sector, "state": state, "cell": cell}

    @property
    def spec_dict(self) -> dict:
        return {"alpha": self.spec.alpha, "n_min": self.spec.n_min, "weighted": self.spec.weighted}


def build_a2_features(source: pd.DataFrame, target: pd.DataFrame,
                      spec: SmoothingSpec = SmoothingSpec()) -> tuple[HistoricalContextMapping, pd.DataFrame]:
    """The single code path used by the runner: fit on NFHS-4, map onto target."""
    mapping = HistoricalContextMapping(spec).fit(source)
    return mapping, mapping.transform(target)
