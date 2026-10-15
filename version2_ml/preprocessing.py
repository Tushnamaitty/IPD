"""Version 2 Phase 3 Step 4: safe feature mapping and fold-fitted preprocessing.

No automatic raw-to-36 feature construction: original preprocessing semantics
and F1 age/parity transformations still require data-level validation.
Caller must pass an approved *named* feature table and feature manifest.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEVEN_MAPPINGS = {
    "counselled_vaginal_bleeding": {"NFHS-4": "s419a", "NFHS-5": "s420a"},
    "counselled_convulsions": {"NFHS-4": "s419b", "NFHS-5": "s420b"},
    "counselled_prolonged_labour": {"NFHS-4": "s419c", "NFHS-5": "s420c"},
    "counselled_severe_abdominal_pain": {"NFHS-4": "s419d", "NFHS-5": "s420d"},
    "counselled_high_blood_pressure": {"NFHS-4": "s419e", "NFHS-5": "s420e"},
    "convulsions_during_pregnancy": {"NFHS-4": "s432", "NFHS-5": "s434"},
    "swelling_during_pregnancy": {"NFHS-4": "s433", "NFHS-5": "s435"},
}
BLOCKED = {
    "m17", "csection", "caseid", "respondent_id", "birth_key",
    "v005", "v001", "v021", "v022", "v023", "sample_weight",
    "primary_sampling_unit", "sample_stratum_v022", "sample_stratum_v023",
    "partition", "split",
}


def map_seven(df: pd.DataFrame, wave: str) -> pd.DataFrame:
    """Only map the seven audited raw-code pairs; preserve genuine missing/8."""
    if wave not in ("NFHS-4", "NFHS-5"):
        raise ValueError("Invalid wave")
    names = {name: mapping[wave] for name, mapping in SEVEN_MAPPINGS.items()}
    absent = [raw for raw in names.values() if raw not in df]
    if absent:
        raise ValueError(f"Missing raw mapping columns: {absent}")
    out = pd.DataFrame(index=df.index)
    for name, raw in names.items():
        values = pd.to_numeric(df[raw], errors="coerce")
        allowed = (0, 1, 8) if name in ("convulsions_during_pregnancy", "swelling_during_pregnancy") else (0, 1)
        if (values.notna() & ~values.isin(allowed)).any():
            raise ValueError(f"Invalid code for {raw}")
        # Code 8 = 'don't know'; preserve as non-negative categorical token.
        out[name] = values.map({0: "no", 1: "yes", 8: "unknown"})
    return out


@dataclass(frozen=True)
class FeatureManifest:
    """Features have to be explicitly approved and non-overlapping."""
    numerical: tuple[str, ...]
    categorical: tuple[str, ...]
    tier: str

    def validate(self, df: pd.DataFrame) -> None:
        if self.tier not in ("F0", "F1", "FT", "F2", "F3", "F4", "F5", "F6"):
            raise ValueError("Unknown feature tier")
        names = list(self.numerical) + list(self.categorical)
        if not names or len(names) != len(set(names)):
            raise ValueError("Empty/duplicated feature names")
        if set(names) & BLOCKED:
            raise ValueError("Protected target/identifier/design columns cannot be features")
        if set(names) - set(df.columns):
            raise ValueError(f"Feature columns absent: {sorted(set(names)-set(df.columns))}")
        # Planning gate: caller must separately certify timing and questionnaire eligibility.
        # Do not claim F1 is leakage reviewed by virtue of this method alone.


class TrainOnlyPreprocessor:
    """Fit once on authorized TRAIN rows. Transforms never update fitted state.

    Guardrail accepts explicit split labels and refuses fit on any other label.
    This alone does not prove that a caller hasn't mislabelled the data.
    """

    def __init__(self, manifest: FeatureManifest):
        self.manifest = manifest
        self.pipeline = None

    def fit(self, frame: pd.DataFrame, *, partition: str):
        if partition != "train":
            raise ValueError("LEAKAGE: preprocessing fit only allowed on training data")
        self.manifest.validate(frame)
        num = list(self.manifest.numerical)
        cat = list(self.manifest.categorical)
        transformers = []
        if num:
            transformers.append(("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
            ]), num))
        if cat:
            transformers.append(("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat))
        self.pipeline = ColumnTransformer(transformers=transformers, remainder="drop")
        X = self._prepare(frame)
        self.pipeline.fit(X)
        return self

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        self.manifest.validate(frame)
        df = frame[list(self.manifest.numerical) + list(self.manifest.categorical)].copy()
        for col in self.manifest.numerical:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in self.manifest.categorical:
            df[col] = df[col].astype("object").where(df[col].notna(), "__MISSING__").astype(str)
        return df

    def transform(self, frame: pd.DataFrame):
        if self.pipeline is None:
            raise ValueError("Call fit on training records before transform")
        return self.pipeline.transform(self._prepare(frame))

    def fit_transform_train(self, frame: pd.DataFrame):
        return self.fit(frame, partition="train").transform(frame)
