"""NFHS-4/5 v024 mappings verified against local Stata value-label metadata.

The two releases assign different numeric codes to states. Jammu and Kashmir
plus Ladakh, and Dadra and Nagar Haveli plus Daman and Diu, are combined to a
common geographic definition across rounds. Keep source codes in the audit.
"""

NFHS4 = {
    1: "Andaman and Nicobar Islands", 2: "Andhra Pradesh", 3: "Arunachal Pradesh",
    4: "Assam", 5: "Bihar", 6: "Chandigarh", 7: "Chhattisgarh",
    8: "Dadra and Nagar Haveli + Daman and Diu",
    9: "Dadra and Nagar Haveli + Daman and Diu", 10: "Goa", 11: "Gujarat",
    12: "Haryana", 13: "Himachal Pradesh", 14: "Jammu and Kashmir + Ladakh",
    15: "Jharkhand", 16: "Karnataka", 17: "Kerala", 18: "Lakshadweep",
    19: "Madhya Pradesh", 20: "Maharashtra", 21: "Manipur", 22: "Meghalaya",
    23: "Mizoram", 24: "Nagaland", 25: "Delhi", 26: "Odisha",
    27: "Puducherry", 28: "Punjab", 29: "Rajasthan", 30: "Sikkim",
    31: "Tamil Nadu", 32: "Tripura", 33: "Uttar Pradesh", 34: "Uttarakhand",
    35: "West Bengal", 36: "Telangana",
}

NFHS5 = {
    1: "Jammu and Kashmir + Ladakh", 2: "Himachal Pradesh", 3: "Punjab",
    4: "Chandigarh", 5: "Uttarakhand", 6: "Haryana", 7: "Delhi",
    8: "Rajasthan", 9: "Uttar Pradesh", 10: "Bihar", 11: "Sikkim",
    12: "Arunachal Pradesh", 13: "Nagaland", 14: "Manipur", 15: "Mizoram",
    16: "Tripura", 17: "Meghalaya", 18: "Assam", 19: "West Bengal",
    20: "Jharkhand", 21: "Odisha", 22: "Chhattisgarh",
    23: "Madhya Pradesh", 24: "Gujarat",
    25: "Dadra and Nagar Haveli + Daman and Diu", 27: "Maharashtra",
    28: "Andhra Pradesh", 29: "Karnataka", 30: "Goa", 31: "Lakshadweep",
    32: "Kerala", 33: "Tamil Nadu", 34: "Puducherry",
    35: "Andaman and Nicobar Islands", 36: "Telangana",
    37: "Jammu and Kashmir + Ladakh",
}


def harmonize_state(frame, round_name):
    """Return a copy with common state names and original v024 codes retained."""
    if round_name not in {"NFHS-4", "NFHS-5"}:
        raise ValueError(f"Unknown round: {round_name}")
    mapping = NFHS4 if round_name == "NFHS-4" else NFHS5
    out = frame.copy()
    out["state_code_original"] = out["state"]
    # Reject unexpected string labels rather than silently dropping every row.
    import pandas as pd
    codes = pd.to_numeric(out["state_code_original"], errors="coerce")
    out["state"] = codes.map(mapping)
    unmapped = out.loc[out["state"].isna(), "state_code_original"].drop_duplicates().tolist()
    if unmapped:
        raise ValueError(f"{round_name} has unmapped v024 codes: {unmapped}")
    return out


def validate_mapping():
    """Check both codebooks collapse onto identical common geographies."""
    assert len(NFHS4) == 36 and len(NFHS5) == 36
    assert set(NFHS4.values()) == set(NFHS5.values())
    assert len(set(NFHS4.values())) == 35


validate_mapping()
