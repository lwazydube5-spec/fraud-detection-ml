"""
test_drift.py — Send drifted predictions to trigger Model Monitor violations
=============================================================================
Sends claims with out-of-distribution values that violate the baseline
constraints learned from training data.

Violations expected:
  - Make: 'Unknown'     → not in domain
  - Month: 'Xyz'        → not in domain
  - Age: -1             → violates non-negative constraint
  - WeekOfMonth: 99     → outside normal range

After running wait for the next hourly monitor execution then check:
    aws sagemaker list-monitoring-executions \
        --monitoring-schedule-name fraud-detection-monitor \
        --region us-east-1
"""

import boto3
import json
import time

REGION        = 'us-east-1'
ENDPOINT_NAME = 'fraud-detection-endpoint'
S3_BUCKET     = 'fraud-detection-monitor-604223541374'

runtime = boto3.client('sagemaker-runtime', region_name=REGION)

# ── Drifted claims ────────────────────────────────────────────────────────
drifted_claims = [

    # Claim 1 — unknown Make not seen in training
    {
        "Month": "Jan", "WeekOfMonth": 3, "DayOfWeek": "Monday",
        "Make": "Unknown",              # ← not in domain
        "AccidentArea": "Urban",
        "DayOfWeekClaimed": "Wednesday", "MonthClaimed": "Feb",
        "WeekOfMonthClaimed": 2, "Sex": "Male", "MaritalStatus": "Single",
        "Age": 28, "Fault": "Policy Holder",
        "PolicyType": "Sedan - Collision", "VehicleCategory": "Sedan",
        "VehiclePrice": "more than 69000", "PolicyNumber": 99999,
        "RepNumber": 5, "Deductible": 400, "DriverRating": 1,
        "Days_Policy_Accident": "1 to 7", "Days_Policy_Claim": "8 to 15",
        "PastNumberOfClaims": "2 to 4", "AgeOfVehicle": "new",
        "AgeOfPolicyHolder": "26 to 30", "PoliceReportFiled": "No",
        "WitnessPresent": "No", "AgentType": "External",
        "NumberOfSuppliments": "3 to 5", "AddressChange_Claim": "under 6 months",
        "NumberOfCars": "1 vehicle", "Year": 1994, "BasePolicy": "Collision"
    },

    # Claim 2 — unknown Month not seen in training
    {
        "Month": "Xyz",                 # ← not in domain
        "WeekOfMonth": 3, "DayOfWeek": "Monday",
        "Make": "Honda", "AccidentArea": "Urban",
        "DayOfWeekClaimed": "Wednesday", "MonthClaimed": "Feb",
        "WeekOfMonthClaimed": 2, "Sex": "Male", "MaritalStatus": "Single",
        "Age": 28, "Fault": "Policy Holder",
        "PolicyType": "Sedan - Collision", "VehicleCategory": "Sedan",
        "VehiclePrice": "more than 69000", "PolicyNumber": 99999,
        "RepNumber": 5, "Deductible": 400, "DriverRating": 1,
        "Days_Policy_Accident": "1 to 7", "Days_Policy_Claim": "8 to 15",
        "PastNumberOfClaims": "2 to 4", "AgeOfVehicle": "new",
        "AgeOfPolicyHolder": "26 to 30", "PoliceReportFiled": "No",
        "WitnessPresent": "No", "AgentType": "External",
        "NumberOfSuppliments": "3 to 5", "AddressChange_Claim": "under 6 months",
        "NumberOfCars": "1 vehicle", "Year": 1994, "BasePolicy": "Collision"
    },

    # Claim 3 — negative Age violates non-negative constraint
    {
        "Month": "Jan", "WeekOfMonth": 3, "DayOfWeek": "Monday",
        "Make": "Honda", "AccidentArea": "Urban",
        "DayOfWeekClaimed": "Wednesday", "MonthClaimed": "Feb",
        "WeekOfMonthClaimed": 2, "Sex": "Male", "MaritalStatus": "Single",
        "Age": -1,                      # ← violates non-negative
        "Fault": "Policy Holder",
        "PolicyType": "Sedan - Collision", "VehicleCategory": "Sedan",
        "VehiclePrice": "more than 69000", "PolicyNumber": 99999,
        "RepNumber": 5, "Deductible": 400, "DriverRating": 1,
        "Days_Policy_Accident": "1 to 7", "Days_Policy_Claim": "8 to 15",
        "PastNumberOfClaims": "2 to 4", "AgeOfVehicle": "new",
        "AgeOfPolicyHolder": "26 to 30", "PoliceReportFiled": "No",
        "WitnessPresent": "No", "AgentType": "External",
        "NumberOfSuppliments": "3 to 5", "AddressChange_Claim": "under 6 months",
        "NumberOfCars": "1 vehicle", "Year": 1994, "BasePolicy": "Collision"
    },

    # Claim 4 — WeekOfMonth 99 — extreme outlier
    {
        "Month": "Jan", "WeekOfMonth": 99,  # ← extreme outlier
        "DayOfWeek": "Monday",
        "Make": "Honda", "AccidentArea": "Urban",
        "DayOfWeekClaimed": "Wednesday", "MonthClaimed": "Feb",
        "WeekOfMonthClaimed": 2, "Sex": "Male", "MaritalStatus": "Single",
        "Age": 28, "Fault": "Policy Holder",
        "PolicyType": "Sedan - Collision", "VehicleCategory": "Sedan",
        "VehiclePrice": "more than 69000", "PolicyNumber": 99999,
        "RepNumber": 5, "Deductible": 400, "DriverRating": 1,
        "Days_Policy_Accident": "1 to 7", "Days_Policy_Claim": "8 to 15",
        "PastNumberOfClaims": "2 to 4", "AgeOfVehicle": "new",
        "AgeOfPolicyHolder": "26 to 30", "PoliceReportFiled": "No",
        "WitnessPresent": "No", "AgentType": "External",
        "NumberOfSuppliments": "3 to 5", "AddressChange_Claim": "under 6 months",
        "NumberOfCars": "1 vehicle", "Year": 1994, "BasePolicy": "Collision"
    },
]

# ── Send drifted predictions ──────────────────────────────────────────────
print('=' * 58)
print('  Sending Drifted Predictions to SageMaker')
print('=' * 58)
print(f'\n  Endpoint : {ENDPOINT_NAME}')
print(f'  Claims   : {len(drifted_claims)} drifted claims\n')

for i, claim in enumerate(drifted_claims):
    try:
        response = runtime.invoke_endpoint(
            EndpointName = ENDPOINT_NAME,
            ContentType  = 'application/json',
            Body         = json.dumps(claim),
        )
        result = json.loads(response['Body'].read().decode())
        print(f'  Claim {i+1}: fraud_probability={result["fraud_probability"]} '
              f'risk_tier={result["risk_tier"]}')
    except Exception as e:
        print(f'  Claim {i+1}: Error — {e}')

# Send 50 more normal claims mixed in — makes drift more visible
print('\n► Sending 50 normal claims for comparison...')
normal_claim = {
    "Month": "Jan", "WeekOfMonth": 3, "DayOfWeek": "Monday",
    "Make": "Honda", "AccidentArea": "Urban",
    "DayOfWeekClaimed": "Wednesday", "MonthClaimed": "Feb",
    "WeekOfMonthClaimed": 2, "Sex": "Male", "MaritalStatus": "Single",
    "Age": 28, "Fault": "Third Party",
    "PolicyType": "Sedan - Collision", "VehicleCategory": "Sedan",
    "VehiclePrice": "20000 to 29000", "PolicyNumber": 12345,
    "RepNumber": 3, "Deductible": 300, "DriverRating": 2,
    "Days_Policy_Accident": "more than 30", "Days_Policy_Claim": "more than 30",
    "PastNumberOfClaims": "none", "AgeOfVehicle": "5 years",
    "AgeOfPolicyHolder": "31 to 35", "PoliceReportFiled": "Yes",
    "WitnessPresent": "Yes", "AgentType": "Internal",
    "NumberOfSuppliments": "none", "AddressChange_Claim": "no change",
    "NumberOfCars": "1 vehicle", "Year": 1994, "BasePolicy": "Liability"
}

for i in range(50):
    runtime.invoke_endpoint(
        EndpointName = ENDPOINT_NAME,
        ContentType  = 'application/json',
        Body         = json.dumps(normal_claim),
    )

print('  ✓ 50 normal claims sent')
print('\n✓ Done — total 54 predictions sent to endpoint')
print('\nNow wait for the next hourly monitor execution.')
print('Check violations with:')
print('  python check_monitor.py')