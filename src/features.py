"""
Feature Engineering for Insurance Fraud Detection
==================================================
Handles ordinal encoding of range-string columns, binary encoding,
one-hot encoding, and interaction features.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


# --------------------------------------------------------------------------- #
#  Ordinal mappings for columns that encode ordered ranges as strings
# --------------------------------------------------------------------------- #

ORDINAL_MAPS = {
    "VehiclePrice": {
        "less than 20000": 1,
        "20000 to 29000": 2,
        "30000 to 39000": 3,
        "40000 to 59000": 4,
        "60000 to 69000": 5,
        "more than 69000": 6,
    },
    "Days_Policy_Accident": {
        "none": 0,
        "1 to 7": 1,
        "8 to 15": 2,
        "15 to 30": 3,
        "more than 30": 4,
    },
    "Days_Policy_Claim": {
        "none": 0,
        "8 to 15": 1,
        "15 to 30": 2,
        "more than 30": 3,
    },
    "PastNumberOfClaims": {
        "none": 0,
        "1": 1,
        "2 to 4": 2,
        "more than 4": 3,
    },
    "AgeOfVehicle": {
        "new": 0,
        "2 years": 1,
        "3 years": 2,
        "4 years": 3,
        "5 years": 4,
        "6 years": 5,
        "7 years": 6,
        "more than 7": 7,
    },
    "AgeOfPolicyHolder": {
        "16 to 17": 1,
        "18 to 20": 2,
        "21 to 25": 3,
        "26 to 30": 4,
        "31 to 35": 5,
        "36 to 40": 6,
        "41 to 50": 7,
        "51 to 65": 8,
        "over 65": 9,
    },
    "NumberOfSuppliments": {
        "none": 0,
        "1 to 2": 1,
        "3 to 5": 2,
        "more than 5": 3,
    },
    "AddressChange_Claim": {
        "no change": 0,
        "under 6 months": 1,
        "1 year": 2,
        "2 to 3 years": 3,
        "4 to 8 years": 4,
    },
    "NumberOfCars": {
        "1 vehicle": 1,
        "2 vehicles": 2,
        "3 to 4": 3,
        "5 to 8": 4,
        "more than 8": 5,
    },
}

# Age is a raw integer (0-80), not a string range, so it can't use
# the string-lookup approach of ORDINAL_MAPS. Instead we define bins
# and labels here and apply pd.cut() in the transformer.
# 0 = missing/unknown (Age==0 in raw data means the value wasn't recorded)
# 1 = teen (1-17), 2 = young adult (18-25), 3 = adult (26-35),
# 4 = mid-age (36-50), 5 = mature (51-65), 6 = senior (66+)
AGE_BINS   = [-1,  0, 17, 25, 35, 50, 65, 120]
AGE_LABELS = [  0,  1,  2,  3,  4,  5,  6    ]

BINARY_COLS = {
    "Sex":              {"Male": 0, "Female": 1},
    "Fault":            {"Policy Holder": 0, "Third Party": 1},
    "PoliceReportFiled":{"No": 0, "Yes": 1},
    "WitnessPresent":   {"No": 0, "Yes": 1},
    "AgentType":        {"External": 0, "Internal": 1},
    "AccidentArea":     {"Urban": 0, "Rural": 1},
}

# Fixed category lists — guarantees identical columns across CV folds
KNOWN_CATEGORIES = {
    "Month":            ['Apr', 'Aug', 'Dec', 'Feb', 'Jan', 'Jul', 'Jun', 'Mar', 'May', 'Nov', 'Oct', 'Sep'],
    "DayOfWeek":        ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday'],
    "Make":             ['Accura', 'BMW', 'Chevrolet', 'Dodge', 'Ferrari', 'Ford', 'Honda', 'Jaguar', 'Lexus', 'Mazda', 'Mecedes', 'Mercury', 'Nisson', 'Pontiac', 'Porche', 'Saab', 'Saturn', 'Toyota', 'VW'],
    "MaritalStatus":    ['Divorced', 'Married', 'Single', 'Widow'],
    "PolicyType":       ['Sedan - All Perils', 'Sedan - Collision', 'Sedan - Liability', 'Sport - All Perils', 'Sport - Collision', 'Sport - Liability', 'Utility - All Perils', 'Utility - Collision', 'Utility - Liability'],
    "VehicleCategory":  ['Sedan', 'Sport', 'Utility'],
    "BasePolicy":       ['All Perils', 'Collision', 'Liability'],
    "MonthClaimed":     ['0', 'Apr', 'Aug', 'Dec', 'Feb', 'Jan', 'Jul', 'Jun', 'Mar', 'May', 'Nov', 'Oct', 'Sep'],
    "DayOfWeekClaimed": ['0', 'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday'],
}

ONE_HOT_COLS = [
    "Month", "DayOfWeek", "Make", "MaritalStatus",
    "PolicyType", "VehicleCategory", "BasePolicy",
    "MonthClaimed", "DayOfWeekClaimed",
]

DROP_COLS = ["PolicyNumber", "Year"]  # identifiers / near-zero variance


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Stateless transformer — all mappings are fixed domain knowledge.
    Safe to use before or inside a Pipeline.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Drop identifier / low-signal columns
        df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

        # 2. Ordinal encoding — string range columns → ordered integers
        for col, mapping in ORDINAL_MAPS.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0).astype(int)

        # 2b. Age ordinal encoding — numeric column bucketed into ordered groups.
        #     Age==0 in the raw data means the value was not recorded (missing),
        #     so it maps cleanly to bucket 0 = "unknown". All other ages fall
        #     into buckets 1-6 by range. Replaces the raw 0-80 integer in place.
        if "Age" in df.columns:
            df["Age"] = pd.cut(
                df["Age"],
                bins   = AGE_BINS,
                labels = AGE_LABELS,
            ).astype(int)

        # 3. Binary encoding
        for col, mapping in BINARY_COLS.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0).astype(int)

        # 4. One-hot encoding — use fixed known categories so columns are
        #    stable across CV folds (no unseen category = missing column)
        for col in ONE_HOT_COLS:
            if col not in df.columns:
                continue
            known_cats = KNOWN_CATEGORIES.get(col, sorted(df[col].unique()))
            for cat in known_cats[1:]:   # drop_first equivalent
                df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
            df.drop(columns=[col], inplace=True)

        # 5. Interaction / derived features
        if "PoliceReportFiled" in df.columns and "WitnessPresent" in df.columns:
            df["NoPolice_NoWitness"] = (
                (df["PoliceReportFiled"] == 0) & (df["WitnessPresent"] == 0)
            ).astype(int)

        if "PastNumberOfClaims" in df.columns and "AgeOfVehicle" in df.columns:
            # Flag the genuinely suspicious combination:
            # multiple past claims on a new or nearly new vehicle.
            # Multiplication was wrong here — high × high = old vehicle with many claims
            # which is actually normal. This binary flag targets exactly what we want.
            df["HighClaims_NewVehicle"] = (
                (df["PastNumberOfClaims"] >= 2) &  # 2 maps to "2 to 4" claims
                (df["AgeOfVehicle"] <= 1)           # 0=new, 1=2 years old
            ).astype(int)

        if "AddressChange_Claim" in df.columns and "Days_Policy_Claim" in df.columns:
            df["AddressChange_x_QuickClaim"] = (
                df["AddressChange_Claim"] * (df["Days_Policy_Claim"] <= 1)
            ).astype(int)

        # 6. Ensure all boolean columns become int
        bool_cols = df.select_dtypes(include="bool").columns
        df[bool_cols] = df[bool_cols].astype(int)

        return df


def load_raw(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load raw CSV, split into X and y."""
    df = pd.read_csv(path)
    y = df["FraudFound_P"]
    X = df.drop(columns=["FraudFound_P"])
    return X, y
