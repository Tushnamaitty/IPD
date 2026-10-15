"""Synthetic tests only, not NFHS dataset validation."""
import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from feature_builder import make_core_features, feature_build_summary, validate_f0_historical, F0_NAMES


def sample():
    return pd.DataFrame({"b3":[1450,1462],"v011":[1150,1126],
                         "v133":[12,16],"v190":[1,5],"bord":[1,3],
                         "m15":[21,31],"v025":[1,2],"v130":[1,2],"b0":[0,2]})


class TestFeatureBuilder(unittest.TestCase):
    def test_core_both_waves(self):
        for wave in ("NFHS-4","NFHS-5"):
            result=make_core_features(sample(),wave=wave)
            self.assertEqual(result.shape,(2,8))
            self.assertAlmostEqual(float(result.iloc[0]["maternal_age_at_index_birth"]),25)
            self.assertEqual(result.iloc[0]["facility_type"],"public")
            self.assertEqual(result.iloc[1]["facility_type"],"private")
            self.assertEqual(feature_build_summary(result)["rows"],2)

    def test_f0_manifest_columns_only(self):
        f=pd.DataFrame({k:[0] for k in F0_NAMES})
        r=validate_f0_historical(f)
        self.assertEqual(r["feature_count"],36)
        self.assertFalse(r["historical_transformations_verified"])

    def test_missing_source_rejected(self):
        with self.assertRaises(ValueError):
            make_core_features(sample().drop(columns=["v133"]),wave="NFHS-5")

    def test_invalid_facility_rejected(self):
        f=sample();f.loc[0,"m15"]=11
        with self.assertRaises(ValueError):
            make_core_features(f,wave="NFHS-5")

    def test_missing_age_rejected(self):
        f=sample();f.loc[0,"b3"]=None
        with self.assertRaises(ValueError):
            make_core_features(f,wave="NFHS-4")

    def test_special_education_code_rejected(self):
        f=sample();f.loc[0,"v133"]=97
        with self.assertRaises(ValueError):
            make_core_features(f,wave="NFHS-5")


if __name__=="__main__":
    unittest.main()
