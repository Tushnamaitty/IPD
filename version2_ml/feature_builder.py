"""IPD V2 feature builder: conservative F1/FT core and F0 historical input guard.

Does NOT reproduce original V1 extraction. F0 requires an already constructed
36-feature table from the original notebook. No patient-level exports.
"""
import pandas as pd

F0_NUM = ("maternal_age","education_years","wealth_index","birth_order",
          "total_children_ever_born","preceding_birth_interval","age_at_first_birth",
          "bmi","first_anc_timing","anc_visits")
F0_CAT = ("facility_type","residence","religion","social_group","twin_order")
F0_BIN = ("anc_doctor","anc_nurse_midwife","anc_traditional_attendant",
          "anc_community_worker","anc_anganwadi_worker","anc_asha_worker",
          "anc_other_provider","no_anc_provider","health_insurance",
          "weight_measured_during_pregnancy","bp_measured_during_pregnancy",
          "urine_sample_during_pregnancy","blood_sample_during_pregnancy",
          "told_about_pregnancy_complications","counselled_vaginal_bleeding",
          "counselled_convulsions","counselled_prolonged_labour",
          "counselled_severe_abdominal_pain","counselled_high_blood_pressure",
          "convulsions_during_pregnancy","swelling_during_pregnancy")
F0_NAMES = F0_NUM + F0_CAT + F0_BIN
CORE_NAMES = ("maternal_age_at_index_birth","education_years","wealth_index",
              "birth_order","facility_type","residence","religion","twin_order")
CORE_RAW = {"education_years":"v133","wealth_index":"v190","birth_order":"bord",
            "facility_type":"m15","residence":"v025","religion":"v130",
            "twin_order":"b0"}
_ALLOWED_FACILITY=set(range(20,28))|set(range(30,34))|{96}
_RELIGION={1,2,3,4,5,6,7,8,9,96}


def validate_f0_historical(frame:pd.DataFrame)->dict:
    """Verify original feature names only, NOT their transformations or reproduction."""
    absent=sorted(set(F0_NAMES)-set(frame.columns))
    if absent:raise ValueError(f"Missing historical feature columns: {absent}")
    if len(F0_NAMES)!=36 or len(set(F0_NAMES))!=36:
        raise RuntimeError("Unexpected historical 36-feature manifest")
    return {"status":"F0_COLUMNS_PRESENT_NOT_REPRODUCED","n_rows":len(frame),
            "feature_count":36,"historical_transformations_verified":False}


def make_core_features(raw:pd.DataFrame, *, wave:str)->pd.DataFrame:
    """Create eight retrospective common-core candidates without imputation.

    Source data MUST be an eligible institutional-birth subset prepared locally.
    Explicitly preserves special/missing values; rejects unsupported categories.
    """
    if wave not in ("NFHS-4","NFHS-5"):
        raise ValueError("Unrecognized NFHS wave")
    required={"b3","v011",*CORE_RAW.values()}
    absent=sorted(required-set(raw.columns))
    if absent:raise ValueError(f"Missing raw feature sources: {absent}")
    source={k:pd.to_numeric(raw[v],errors="coerce") for k,v in CORE_RAW.items()}
    age_months=pd.to_numeric(raw["b3"],errors="coerce")-pd.to_numeric(raw["v011"],errors="coerce")
    if (age_months.isna() | age_months.lt(0)).any():
        raise ValueError("Invalid maternal age CMC")
    if ((age_months/12 < 10)|(age_months/12 > 55)).any():
        raise ValueError("Maternal age outside preliminary 10..55 plausibility interval")
    if (~source["education_years"].between(0,20)).any():raise ValueError("Invalid education years; inspect 97")
    if (~source["wealth_index"].isin([1,2,3,4,5])).any():raise ValueError("Invalid wealth index")
    if (~source["birth_order"].ge(1)).any():raise ValueError("Invalid birth order")
    if (~source["facility_type"].isin(_ALLOWED_FACILITY)).any():
        raise ValueError("Outside broad institutional facilities")
    if (~source["residence"].isin([1,2])).any():raise ValueError("Invalid residence")
    if (~source["religion"].isin(_RELIGION)).any():raise ValueError("Invalid religion")
    if (~source["twin_order"].isin([0,1,2,3,4,5])).any():raise ValueError("Invalid multiple-birth code")

    features=pd.DataFrame(index=raw.index)
    features["maternal_age_at_index_birth"]=age_months/12.0
    features["education_years"]=source["education_years"]
    # These are ordinal wave-relative ranks; avoid treating as absolute money.
    features["wealth_index"]=source["wealth_index"].astype("string")
    features["birth_order"]=source["birth_order"]
    facilities=source["facility_type"]
    features["facility_type"]=facilities.map(
        lambda val: "public" if val in range(20,28) else
                    ("private" if val in range(30,34) else "other"))
    features["residence"]=source["residence"].map({1:"urban",2:"rural"})
    features["religion"]=source["religion"].astype("Int64").astype("string")
    features["twin_order"]=source["twin_order"].astype("Int64").astype("string")
    if tuple(features.columns)!=CORE_NAMES:
        raise RuntimeError("Core feature ordering differs from approved candidate")
    return features


def feature_build_summary(frame:pd.DataFrame)->dict:
    if tuple(frame.columns)!=CORE_NAMES:
        raise ValueError("Expected exactly eight core columns in manifest order")
    return {"status":"EIGHT_CORE_FEATURES_BUILT_CANDIDATE",
            "rows":int(len(frame)),"columns":list(frame.columns),
            "missing_by_feature":{c:int(frame[c].isna().sum()) for c in frame},
            "feature_meanings_and_time_certified":False,
            "model_baseline_reproduced":False}
