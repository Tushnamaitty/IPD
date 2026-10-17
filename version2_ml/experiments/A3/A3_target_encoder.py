"""A3: grouped out-of-fold (OOF) state x sector target encoding on NFHS-5.

Unlike A2, A3 learns from NFHS-5 *outer-training* labels. Guarantees:
  * fit only on rows declared partition="train" (validation/test refused);
  * outer-training rows receive OOF encodings: rate tables fitted on K-1 inner
    mother-grouped folds encode the omitted fold, so no mother is in both the
    table-fitting rows and the encoded rows; results return in input row order;
  * validation/test rows are encoded by one table fitted on 100% of outer train;
  * transform reads ONLY the state and sector columns.

Smoothing at every level: TE = (y + alpha*mu) / (n + alpha), mu = overall rate
of the rows used to fit that table. Predeclared fallback when a level has
n < n_min (or is unseen):
  te_state_sector: cell -> state -> sector -> mu
  te_state:        state -> mu
  te_sector:       sector -> mu
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

TE_CELL = "te_state_sector"
TE_STATE = "te_state"
TE_SECTOR = "te_sector"
TE_FLAG = "te_cell_fallback"
TE_COLUMNS = (TE_CELL, TE_STATE, TE_SECTOR, TE_FLAG)
LEVEL = "te_cell_level"


class NotEstimable(ValueError):
    """Raised instead of silently violating mother grouping or class support."""


@dataclass(frozen=True)
class TESpec:
    alpha: float = 20.0
    n_min: int = 30

    def __post_init__(self):
        if not (np.isfinite(self.alpha) and self.alpha >= 0):
            raise ValueError("alpha must be finite and >= 0")
        if int(self.n_min) < 1:
            raise ValueError("n_min must be >= 1")


def _keys(frame, state_col, sector_col):
    for c in (state_col, sector_col):
        if c not in frame.columns:
            raise ValueError(f"Missing column: {c}")
    sub = frame[[state_col, sector_col]]
    state = sub[state_col].astype("string")
    sector = sub[sector_col].astype("string")
    if state.isna().any() or sector.isna().any():
        raise ValueError("Missing state or sector key")
    return state, sector


class RateTable:
    """Smoothed state/sector/cell rates from one set of labelled rows."""

    def __init__(self, spec: TESpec):
        self.spec = spec
        self.tables = None

    def fit(self, state: pd.Series, sector: pd.Series, y: np.ndarray):
        y = np.asarray(y, dtype=float)
        if len(y) == 0 or len(y) != len(state) or len(y) != len(sector):
            raise ValueError("Empty or misaligned rate-table inputs")
        a, n_min = float(self.spec.alpha), int(self.spec.n_min)
        mu = float(y.mean())
        df = pd.DataFrame({"s": state.to_numpy(), "k": sector.to_numpy(), "y": y})
        df["c"] = df["s"] + "|" + df["k"]

        def level(col):
            g = df.groupby(col, sort=True)["y"].agg(["size", "sum"])
            te = (g["sum"] + a * mu) / (g["size"] + a)
            return {str(i): {"n": int(r["size"]), "y": int(r["sum"]), "te": float(te[i]),
                             "supported": bool(r["size"] >= n_min)} for i, r in g.iterrows()}
        self.tables = {"mu": mu, "n": int(len(y)), "alpha": a, "n_min": n_min,
                       "state": level("s"), "sector": level("k"), "cell": level("c")}
        return self

    def transform(self, state: pd.Series, sector: pd.Series, index) -> pd.DataFrame:
        if self.tables is None:
            raise ValueError("RateTable not fitted")
        t, mu = self.tables, self.tables["mu"]

        def lookup(level, keys):
            sup = {k: v["te"] for k, v in t[level].items() if v["supported"]}
            vals = keys.map(sup)
            return vals.astype(float).to_numpy()
        s_te = lookup("state", state)
        k_te = lookup("sector", sector)
        c_te = lookup("cell", state + "|" + sector)
        cell = np.where(~np.isnan(c_te), c_te, np.where(~np.isnan(s_te), s_te, np.where(~np.isnan(k_te), k_te, mu)))
        lvl = np.where(~np.isnan(c_te), "cell", np.where(~np.isnan(s_te), "state",
                                                          np.where(~np.isnan(k_te), "sector", "global")))
        out = pd.DataFrame(index=index)
        out[TE_CELL] = cell
        out[TE_STATE] = np.where(np.isnan(s_te), mu, s_te)
        out[TE_SECTOR] = np.where(np.isnan(k_te), mu, k_te)
        out[TE_FLAG] = (lvl != "cell").astype(float)
        out[LEVEL] = lvl
        return out

    def digest(self) -> str:
        def enc(x):
            if isinstance(x, dict):
                return {k: enc(v) for k, v in x.items()}
            return float(x).hex() if isinstance(x, float) else x
        return hashlib.sha256(json.dumps(enc(self.tables), sort_keys=True).encode()).hexdigest()


class GroupedOOFTargetEncoder:
    def __init__(self, spec: TESpec = TESpec(), *, n_splits: int = 3, seed: int = 42,
                 state_col: str = "nfhs5_state", sector_col: str = "sector"):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.spec, self.n_splits, self.seed = spec, int(n_splits), int(seed)
        self.state_col, self.sector_col = state_col, sector_col
        self.full_table = None
        self.fold_report = None

    def inner_folds(self, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """Inner fold id per training row (mother-grouped, outcome-stratified)."""
        if pd.Series(groups).nunique() < self.n_splits:
            raise NotEstimable("NOT ESTIMABLE: fewer mothers than inner folds")
        if len(np.unique(y)) < 2:
            raise NotEstimable("NOT ESTIMABLE: training data has one outcome class")
        cv = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        fold = np.full(len(y), -1)
        for k, (_, enc_idx) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            fold[enc_idx] = k
        if (fold < 0).any():
            raise RuntimeError("Unassigned training row in inner folds")
        per_group = pd.Series(fold).groupby(pd.Series(groups)).nunique()
        if per_group.gt(1).any():
            raise AssertionError("LEAKAGE: a mother spans several inner folds")
        return fold

    def check_folds(self, fold, groups) -> np.ndarray:
        """Validate externally supplied inner folds (reused across alphas)."""
        fold = np.asarray(fold)
        if fold.shape != (len(groups),) or set(np.unique(fold)) != set(range(self.n_splits)):
            raise ValueError("Inner folds misaligned with training rows or wrong fold labels")
        if pd.Series(fold).groupby(pd.Series(groups)).nunique().gt(1).any():
            raise AssertionError("LEAKAGE: a mother spans several inner folds")
        return fold

    def fit_transform_train(self, frame: pd.DataFrame, y, groups, *, partition: str, folds=None) -> pd.DataFrame:
        if partition != "train":
            raise ValueError("LEAKAGE: target-encoder fit only allowed on outer training data")
        y = np.asarray(y)
        groups = np.asarray(groups)
        if len(frame) != len(y) or len(frame) != len(groups):
            raise ValueError("Frame, outcome and group arrays must be aligned")
        if not frame.index.is_unique:
            raise ValueError("Training index must be unique for order reconstruction")
        if pd.isna(groups).any() or pd.Series(groups).astype(str).str.strip().eq("").any():
            raise ValueError("Missing mother group key")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("Outcome must be 0/1")
        y = y.astype(int)
        state, sector = _keys(frame, self.state_col, self.sector_col)
        fold = self.inner_folds(y, groups) if folds is None else self.check_folds(folds, groups)
        parts, report = [], []
        for k in range(self.n_splits):
            enc = fold == k
            fit = ~enc
            if set(groups[enc]) & set(groups[fit]):
                raise AssertionError("LEAKAGE: mother in both table-fit and encoded rows")
            if len(np.unique(y[fit])) < 2 or enc.sum() == 0:
                raise NotEstimable(f"NOT ESTIMABLE: inner fold {k} lacks support")
            table = RateTable(self.spec).fit(state[fit], sector[fit], y[fit])
            parts.append(table.transform(state[enc], sector[enc], frame.index[enc]))
            report.append({"inner_fold": k, "fit_births": int(fit.sum()), "encoded_births": int(enc.sum()),
                           "fit_mothers": int(pd.Series(groups[fit]).nunique()),
                           "encoded_mothers": int(pd.Series(groups[enc]).nunique()),
                           "encoded_events": int(y[enc].sum()), "mother_overlap": 0,
                           "encoded_cell_fallback_births": int(parts[-1][TE_FLAG].sum())})
        oof = pd.concat(parts).reindex(frame.index)
        if oof[TE_CELL].isna().any() or not oof.index.equals(frame.index):
            raise RuntimeError("OOF row-order reconstruction failed")
        self.full_table = RateTable(self.spec).fit(state, sector, y)
        self.fold_report = report
        return oof

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Encode validation/test (or any) rows with the 100%-outer-train table."""
        if self.full_table is None:
            raise ValueError("Call fit_transform_train on outer training rows first")
        state, sector = _keys(frame, self.state_col, self.sector_col)
        return self.full_table.transform(state, sector, frame.index)


def oof_log_loss(y, te) -> float:
    p = np.clip(np.asarray(te, float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def select_alpha(frame: pd.DataFrame, y, groups, grid, *, n_min=30, n_splits=3, seed=42,
                 state_col="nfhs5_state", sector_col="sector", partition: str,
                 folds=None) -> tuple[float, pd.DataFrame]:
    """Choose alpha by OOF log-loss of te_state_sector on OUTER-TRAIN rows only.

    Ties (within 1e-12) go to the larger alpha (more smoothing).
    """
    if partition != "train":
        raise ValueError("LEAKAGE: alpha tuning only allowed on outer training data")
    rows = []
    for a in grid:
        enc = GroupedOOFTargetEncoder(TESpec(alpha=float(a), n_min=n_min), n_splits=n_splits, seed=seed,
                                      state_col=state_col, sector_col=sector_col)
        oof = enc.fit_transform_train(frame, y, groups, partition="train", folds=folds)
        rows.append({"alpha": float(a), "oof_log_loss_te_state_sector": oof_log_loss(y, oof[TE_CELL])})
    res = pd.DataFrame(rows)
    best = res["oof_log_loss_te_state_sector"].min()
    chosen = float(res.loc[res["oof_log_loss_te_state_sector"] <= best + 1e-12, "alpha"].max())
    res["selected"] = res["alpha"].eq(chosen)
    return chosen, res


def encode_partitions(frame: pd.DataFrame, y, groups, partition: pd.Series, spec: TESpec, *,
                      n_splits=3, seed=42, state_col="nfhs5_state", sector_col="sector", folds=None):
    """Runner path: OOF for outer train, full-train table for every other partition.

    Outcomes of non-train rows are sliced away before the encoder sees anything.
    """
    part = pd.Series(np.asarray(partition), index=frame.index)
    train = part.eq("train").to_numpy()
    y_train = np.asarray(y)[train]
    enc = GroupedOOFTargetEncoder(spec, n_splits=n_splits, seed=seed, state_col=state_col, sector_col=sector_col)
    oof = enc.fit_transform_train(frame[train], y_train, np.asarray(groups)[train], partition="train", folds=folds)
    rest = enc.transform(frame[~train])
    out = pd.concat([oof, rest]).reindex(frame.index)
    return enc, out
