"""
tests/test_pipeline.py — Automated tests for the fraud detection pipeline
"""
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features import FraudFeatureEngineer, load_raw

# ── Sample claim for testing ─────────────────────────────────────────────
SAMPLE_CLAIM = {
    "Month": "Jan", "WeekOfMonth": 3, "DayOfWeek": "Monday",
    "Make": "Honda", "AccidentArea": "Urban",
    "DayOfWeekClaimed": "Wednesday", "MonthClaimed": "Feb",
    "WeekOfMonthClaimed": 2, "Sex": "Male", "MaritalStatus": "Single",
    "Age": 28, "Fault": "Policy Holder", "PolicyType": "Sedan - Collision",
    "VehicleCategory": "Sedan", "VehiclePrice": "20000 to 29000",
    "PolicyNumber": 12345, "RepNumber": 3, "Deductible": 400,
    "DriverRating": 2, "Days_Policy_Accident": "more than 30",
    "Days_Policy_Claim": "more than 30", "PastNumberOfClaims": "none",
    "AgeOfVehicle": "3 years", "AgeOfPolicyHolder": "26 to 30",
    "PoliceReportFiled": "Yes", "WitnessPresent": "Yes",
    "AgentType": "Internal", "NumberOfSuppliments": "none",
    "AddressChange_Claim": "no change", "NumberOfCars": "1 vehicle",
    "Year": 1994, "BasePolicy": "Collision",
}


@pytest.fixture
def engineer():
    return FraudFeatureEngineer()


@pytest.fixture
def sample_df():
    return pd.DataFrame([SAMPLE_CLAIM])


def test_engineer_fit_returns_self(engineer, sample_df):
    """fit() must return self for Pipeline compatibility."""
    result = engineer.fit(sample_df)
    assert result is engineer


def test_engineer_output_is_dataframe(engineer, sample_df):
    """transform() must return a DataFrame."""
    result = engineer.transform(sample_df)
    assert isinstance(result, pd.DataFrame)


def test_engineer_drops_policy_number(engineer, sample_df):
    """PolicyNumber should be dropped — it is just an ID."""
    result = engineer.transform(sample_df)
    assert "PolicyNumber" not in result.columns


def test_engineer_drops_year(engineer, sample_df):
    """Year should be dropped — near zero variance."""
    result = engineer.transform(sample_df)
    assert "Year" not in result.columns


def test_engineer_no_string_columns(engineer, sample_df):
    """All columns must be numeric after transformation."""
    result = engineer.transform(sample_df)
    string_cols = result.select_dtypes(include="object").columns.tolist()
    assert len(string_cols) == 0, f"String columns remain: {string_cols}"


def test_engineer_no_nulls(engineer, sample_df):
    """No null values should remain after transformation."""
    result = engineer.transform(sample_df)
    null_cols = result.columns[result.isnull().any()].tolist()
    assert len(null_cols) == 0, f"Null columns: {null_cols}"


def test_age_encoding_missing(engineer):
    """Age == 0 (missing) should map to bucket 0."""
    df = pd.DataFrame([{**SAMPLE_CLAIM, "Age": 0}])
    result = engineer.transform(df)
    assert result["Age"].iloc[0] == 0


def test_age_encoding_adult(engineer):
    """Age 28 should map to bucket 3 (adult 26-35)."""
    df = pd.DataFrame([{**SAMPLE_CLAIM, "Age": 28}])
    result = engineer.transform(df)
    assert result["Age"].iloc[0] == 3


def test_binary_encoding_fault(engineer, sample_df):
    """Fault Policy Holder should encode to 0."""
    result = engineer.transform(sample_df)
    assert result["Fault"].iloc[0] == 0


def test_interaction_feature_exists(engineer, sample_df):
    """NoPolice_NoWitness interaction feature must be created."""
    result = engineer.transform(sample_df)
    assert "NoPolice_NoWitness" in result.columns


def test_column_count_stable(engineer):
    """Two identical claims must produce the same number of columns."""
    df1 = pd.DataFrame([SAMPLE_CLAIM])
    df2 = pd.DataFrame([SAMPLE_CLAIM])
    r1 = engineer.transform(df1)
    r2 = engineer.transform(df2)
    assert r1.shape[1] == r2.shape[1]