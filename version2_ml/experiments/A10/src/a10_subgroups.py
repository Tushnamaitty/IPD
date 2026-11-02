"""A10 subgroup membership from the A7 tier frame and A1 state labels (config.yaml: subgroups)."""
from __future__ import annotations

import numpy as np
import pandas as pd

DIMENSIONS = ("residence", "facility_sector", "birth_order", "wealth", "maternal_age", "previous_cs_history", "state")
DIMENSION_LABELS = {
    "residence": "Residence", "facility_sector": "Facility sector", "birth_order": "Birth order",
    "wealth": "Wealth quintile", "maternal_age": "Maternal age at delivery",
    "previous_cs_history": "Previous-CS history", "state": "State/UT",
}


def memberships(base: pd.DataFrame, state_labels, config: dict) -> dict:
    """{dimension: {group: boolean mask over base rows}} in config order (states sorted)."""
    spec = config["subgroups"]
    age = pd.to_numeric(base["maternal_age_at_index_birth"]).to_numpy()
    order = pd.to_numeric(base["birth_order"]).to_numpy()
    states = pd.Series(np.asarray(state_labels)).astype("string").to_numpy()
    out = {
        "residence": {g: base["residence"].astype("string").eq(g).to_numpy() for g in spec["residence"]["groups"]},
        "facility_sector": {g: base["facility_type"].astype("string").eq(g).to_numpy()
                            for g in spec["facility_sector"]["groups"]},
        "birth_order": {"first_birth": order == 1, "second_or_later": order >= 2},
        "wealth": {g: base["wealth_index"].astype("string").eq(g).to_numpy() for g in spec["wealth"]["groups"]},
        "maternal_age": {"<20": age < 20, "20-34": (age >= 20) & (age < 35), "35+": age >= 35},
        "previous_cs_history": {g: base["history_four_level"].astype("string").eq(g).to_numpy()
                                for g in spec["previous_cs_history"]["groups"]},
        "state": {g: states == g for g in sorted(set(states))},
    }
    for dimension in ("residence", "facility_sector", "birth_order", "wealth", "maternal_age"):
        total = np.sum([mask for mask in out[dimension].values()], axis=0)
        if not (total == 1).all():
            raise ValueError(f"{dimension}: groups are not an exact partition of the rows")
    return out
