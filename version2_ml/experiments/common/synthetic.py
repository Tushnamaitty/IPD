"""Deterministic synthetic NFHS-shaped birth fixtures. No real data involved.

The frame mimics the raw BR columns needed to exercise the frozen cohort rule
and the eight-core feature builder (``version2_ml/feature_builder.py``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SYNTHETIC_STATE_CODES = (1, 9, 21, 27, 33, 37)
SYNTHETIC_FACILITY_CODES = (21, 22, 24, 25, 31, 32)
SYNTHETIC_RELIGION_CODES = (1, 2, 3, 4, 96)
INTERVIEW_CMC = 1500
COLUMNS = (
    "caseid",
    "bidx",
    "b3",
    "b19",
    "v008",
    "v011",
    "m15",
    "m17",
    "v024",
    "v025",
    "v130",
    "v133",
    "v190",
    "bord",
    "b0",
)


def synthetic_birth_frame(*, n_mothers: int = 120, seed: int = 42) -> pd.DataFrame:
    """Create a deterministic synthetic birth-level frame with documented columns."""
    if n_mothers < 6:
        raise ValueError("At least six synthetic mothers are required")
    rng = np.random.RandomState(seed)
    rows = []
    for mother in range(n_mothers):
        n_births = 1 + int(rng.randint(3))
        base_months_since = 6 * n_births + int(rng.randint(4))
        mother_cmc = INTERVIEW_CMC - 12 * (18 + int(rng.randint(20)))
        parity_before = int(rng.randint(0, 4))
        state = int(rng.choice(SYNTHETIC_STATE_CODES))
        residence = 1 if rng.rand() < 0.4 else 2
        religion = int(rng.choice(SYNTHETIC_RELIGION_CODES))
        education = int(rng.randint(0, 18))
        wealth = int(rng.randint(1, 6))
        for order in range(n_births):
            twin = order > 0 and rng.rand() < 0.15
            months_since = (
                base_months_since - 6 * order if not twin
                else base_months_since - 6 * (order - 1)
            )
            months_since = max(months_since, 0)
            b3 = INTERVIEW_CMC - months_since
            rows.append(
                {
                    "caseid": f"synth_{mother:04d}",
                    "bidx": order + 1,
                    "b3": b3,
                    "b19": INTERVIEW_CMC - b3,
                    "v008": INTERVIEW_CMC,
                    "v011": mother_cmc,
                    "m15": int(rng.choice(SYNTHETIC_FACILITY_CODES)),
                    "m17": int(rng.rand() < 0.3),
                    "v024": state,
                    "v025": residence,
                    "v130": religion,
                    "v133": education,
                    "v190": wealth,
                    "bord": parity_before + order + 1,
                    "b0": 1 if twin else 0,
                }
            )
    return pd.DataFrame(rows, columns=list(COLUMNS))
