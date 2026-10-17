"""A5: NFHS-4 -> NFHS-5 source-only temporal transfer on the eight common-core features.

Guarantees enforced in code:
  * fit only on rows whose wave is NFHS-4 and whose partition is "train";
  * the decision threshold is chosen on NFHS-4 validation rows only, after which
    the model is frozen and SHA-256 hashed; every predict re-checks the hash;
  * feature set is exactly the eight FT core features; the structurally
    restricted F0 pregnancy/ANC predictors are rejected;
  * categorical levels outside the fixed codebook vocabulary fail closed; levels
    inside the vocabulary but unseen in source training encode as all-zero
    (OneHotEncoder handle_unknown="ignore") and are counted, never crash.
Optional target recalibration (D) uses labels from target mothers disjoint from
the evaluation mothers and is reported as target-supervised, never zero-shot.
"""
from __future__ import annotations
import hashlib
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "version2_ml", ROOT / "version2_ml" / "experiments" / "common"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from feature_builder import CORE_NAMES, F0_NAMES  # noqa: E402
from preprocessing import FeatureManifest  # noqa: E402
from model_framework import AuthorizedClassifier  # noqa: E402
from exp_metrics import select_threshold_max_f1, _newton  # noqa: E402

SOURCE_WAVE, TARGET_WAVE = "NFHS-4", "NFHS-5"
FT_NUM = ("maternal_age_at_index_birth", "education_years", "birth_order")
FT_CAT = ("wealth_index", "facility_type", "residence", "religion", "twin_order")
FT_MANIFEST = FeatureManifest(numerical=FT_NUM, categorical=FT_CAT, tier="FT")
# F0-only predictors (incl. interview-age maternal_age and the 22 pregnancy/ANC items) are excluded from FT.
RESTRICTED_F0 = tuple(sorted(set(F0_NAMES) - set(CORE_NAMES)))
VOCAB = {"wealth_index": {"1", "2", "3", "4", "5"},
         "facility_type": {"public", "private", "other"},
         "residence": {"urban", "rural"},
         "religion": {"1", "2", "3", "4", "5", "6", "7", "8", "9", "96"},
         "twin_order": {"0", "1", "2", "3", "4", "5"}}


def assert_ft_only(feature_names) -> None:
    names = tuple(feature_names)
    if set(names) & set(RESTRICTED_F0):
        raise ValueError(f"Restricted F0-only predictors not allowed in FT: {sorted(set(names) & set(RESTRICTED_F0))}")
    if sorted(names) != sorted(FT_NUM + FT_CAT) or len(names) != 8:
        raise ValueError("FT feature set must be exactly the eight common-core features")


assert_ft_only(FT_MANIFEST.numerical + FT_MANIFEST.categorical)


def harmonize_core(frame: pd.DataFrame, wave: str) -> pd.DataFrame:
    """Return the eight FT columns with wave-independent dtypes; fail closed on codebook drift."""
    if wave not in (SOURCE_WAVE, TARGET_WAVE):
        raise ValueError("Unknown wave")
    missing = [c for c in FT_NUM + FT_CAT if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing FT columns: {missing}")
    out = pd.DataFrame(index=frame.index)
    for c in FT_NUM:
        out[c] = pd.to_numeric(frame[c], errors="coerce").astype(float)
        if out[c].isna().any():
            raise ValueError(f"{wave}: missing/non-numeric {c}")
    for c in FT_CAT:
        s = frame[c].astype("string")
        bad = sorted(set(s.dropna()) - VOCAB[c])
        if s.isna().any() or bad:
            raise ValueError(f"{wave}: {c} outside fixed codebook vocabulary: {bad}")
        out[c] = s.astype(object)
    return out


def harmonization_report(source_train: pd.DataFrame, target: pd.DataFrame) -> dict:
    """Aggregate vocabulary/unseen-level and numeric-range comparison (no rows)."""
    rep = {"categorical": {}, "numeric": {}}
    for c in FT_CAT:
        src = set(source_train[c])
        tgt_counts = target[c].value_counts()
        unseen = {k: int(v) for k, v in tgt_counts.items() if k not in src}
        rep["categorical"][c] = {"fixed_vocabulary": sorted(VOCAB[c]), "source_train_levels": sorted(src),
                                 "target_levels": sorted(tgt_counts.index.tolist()),
                                 "target_levels_unseen_in_source": sorted(unseen),
                                 "target_rows_with_unseen_level": int(sum(unseen.values())),
                                 "target_unseen_rate": float(sum(unseen.values()) / len(target)),
                                 "unseen_handling": "all-zero one-hot (handle_unknown='ignore')"}
    for c in FT_NUM:
        lo, hi = source_train[c].quantile([0.01, 0.99])
        rep["numeric"][c] = {"source_p01_p99": [float(lo), float(hi)],
                             "target_share_outside_source_p01_p99": float((~target[c].between(lo, hi)).mean())}
    return rep


def _digest(obj) -> str:
    return hashlib.sha256(pickle.dumps(obj, protocol=4)).hexdigest()


class SourceOnlyTransferModel:
    """Fit on NFHS-4 train, threshold on NFHS-4 validation, then frozen."""

    def __init__(self, algorithm: str = "logistic_regression", seed: int = 42):
        self.algorithm, self.seed = algorithm, seed
        self.clf = None
        self.threshold = None
        self.frozen_digest = None
        self.source_levels = None

    @staticmethod
    def _require_source(wave, n, what):
        w = pd.Series(np.asarray(wave)).astype("string")
        if len(w) != n or w.isna().any() or set(w) != {SOURCE_WAVE}:
            raise ValueError(f"LEAKAGE: {what} only allowed on {SOURCE_WAVE} rows")

    def fit(self, frame: pd.DataFrame, y, *, wave, partition: str):
        if self.frozen_digest is not None:
            raise RuntimeError("Model is frozen; create a new instance to refit")
        if partition != "train":
            raise ValueError("LEAKAGE: source fit only allowed on the source training partition")
        self._require_source(wave, len(frame), "fit")
        x = harmonize_core(frame, SOURCE_WAVE)
        self.source_levels = {c: sorted(set(x[c])) for c in FT_CAT}
        self.clf = AuthorizedClassifier(FT_MANIFEST, algorithm=self.algorithm, seed=self.seed)
        self.clf.fit(x, y, partition="train")
        return self

    def select_threshold(self, frame: pd.DataFrame, y, *, wave, partition: str) -> float:
        if self.clf is None:
            raise ValueError("Fit before threshold selection")
        if self.frozen_digest is not None:
            raise RuntimeError("Threshold already frozen")
        if partition != "validation":
            raise ValueError("LEAKAGE: threshold selection only on source validation")
        self._require_source(wave, len(frame), "threshold selection")
        p = self.clf.predict_proba(harmonize_core(frame, SOURCE_WAVE), partition="validation")
        self.threshold = select_threshold_max_f1(np.asarray(y), p)
        self.frozen_digest = self.digest()
        return self.threshold

    def digest(self) -> str:
        return _digest((self.clf.preprocessor.pipeline, self.clf.estimator, self.threshold))

    def predict(self, frame: pd.DataFrame, *, wave: str) -> np.ndarray:
        if self.frozen_digest is None:
            raise ValueError("Freeze (select source threshold) before scoring")
        if self.digest() != self.frozen_digest:
            raise RuntimeError("LEAKAGE: frozen source model changed after freezing")
        part = "validation"  # AuthorizedClassifier label for a non-fitting scoring call
        return self.clf.predict_proba(harmonize_core(frame, wave), partition=part)

    def fitted_categories(self) -> dict:
        ct = self.clf.preprocessor.pipeline
        enc = ct.named_transformers_["categorical"].named_steps["onehot"]
        return {c: sorted(map(str, cats)) for c, cats in zip(FT_CAT, enc.categories_)}


def fit_source_model(source: pd.DataFrame, *, algorithm: str, seed: int = 42,
                     part_col: str = "source_partition") -> SourceOnlyTransferModel:
    """Fit on NFHS-4 train rows, choose threshold on NFHS-4 validation rows, freeze."""
    tr = source[part_col].eq("train").to_numpy()
    va = source[part_col].eq("validation").to_numpy()
    model = SourceOnlyTransferModel(algorithm, seed)
    model.fit(source.loc[tr, list(FT_NUM + FT_CAT)], source.loc[tr, "m17"], wave=source.loc[tr, "wave"],
              partition="train")
    model.select_threshold(source.loc[va, list(FT_NUM + FT_CAT)], source.loc[va, "m17"],
                           wave=source.loc[va, "wave"], partition="validation")
    return model


def zero_shot_pipeline(source: pd.DataFrame, target: pd.DataFrame, *, algorithm: str, seed: int = 42,
                       part_col: str = "source_partition") -> tuple[SourceOnlyTransferModel, np.ndarray]:
    """Single code path: fit/threshold on NFHS-4 only, then score target features.

    The target frame is used only through its eight FT feature columns.
    """
    model = fit_source_model(source, algorithm=algorithm, seed=seed, part_col=part_col)
    return model, model.predict(target[list(FT_NUM + FT_CAT)], wave=TARGET_WAVE)


class AdaptationRecalibrator:
    """(D) Logistic recalibration on a target ADAPTATION sample, mother-disjoint from evaluation."""

    def __init__(self, kind: str = "intercept"):
        if kind not in ("intercept", "intercept_slope"):
            raise ValueError("kind must be 'intercept' or 'intercept_slope'")
        self.kind = kind
        self.coef = None

    def fit(self, p, y, *, adaptation_groups, evaluation_groups):
        if set(map(str, adaptation_groups)) & set(map(str, evaluation_groups)):
            raise ValueError("LEAKAGE: adaptation and evaluation mothers overlap")
        p, y = np.asarray(p, float), np.asarray(y, float)
        if p.shape != y.shape or len(np.unique(y)) < 2:
            raise ValueError("Adaptation sample misaligned or single-class")
        lp = logit(np.clip(p, 1e-12, 1 - 1e-12))
        if self.kind == "intercept":
            b = _newton(np.ones((len(y), 1)), y, np.ones_like(y), lp)
            self.coef = (float(b[0]), 1.0) if b is not None else None
        else:
            b = _newton(np.column_stack([np.ones_like(lp), lp]), y, np.ones_like(y), np.zeros_like(lp))
            self.coef = (float(b[0]), float(b[1])) if b is not None else None
        if self.coef is None:
            raise RuntimeError("Recalibration did not converge")
        return self

    def transform(self, p) -> np.ndarray:
        if self.coef is None:
            raise ValueError("Fit recalibrator first")
        lp = logit(np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12))
        return expit(self.coef[0] + self.coef[1] * lp)
