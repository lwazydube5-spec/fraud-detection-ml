"""
test_sagemaker.py — Test the SageMaker endpoint and verify data capture
"""

import boto3
import json
import time

REGION        = 'us-east-1'
ENDPOINT_NAME = 'fraud-detection-endpoint'
S3_BUCKET     = 'fraud-detection-monitor-604223541374'

# ── Test prediction ───────────────────────────────────────────────────────
runtime = boto3.client('sagemaker-runtime', region_name=REGION)

claim = {
    "Month": "Jan", "WeekOfMonth": 3, "DayOfWeek": "Monday",
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
}

print('► Sending prediction to SageMaker endpoint...')
response = runtime.invoke_endpoint(
    EndpointName = ENDPOINT_NAME,
    ContentType  = 'application/json',
    Body         = json.dumps(claim),
)

result = json.loads(response['Body'].read().decode())
print(f'\n✓ Prediction received:')
print(f'  fraud_probability : {result["fraud_probability"]}')
print(f'  fraud_predicted   : {result["fraud_predicted"]}')
print(f'  risk_tier         : {result["risk_tier"]}')
print(f'  inference_ms      : {result["inference_ms"]}')

# ── Check S3 for captured data ────────────────────────────────────────────
print('\n► Checking S3 for captured data (wait 30 seconds)...')
time.sleep(30)

s3 = boto3.client('s3', region_name=REGION)
response = s3.list_objects_v2(
    Bucket = S3_BUCKET,
    Prefix = 'data-capture/',
)

if response.get('Contents'):
    print(f'✓ Data captured to S3:')
    for obj in response['Contents']:
        print(f'  {obj["Key"]}')
else:
    print('  No captured data yet — may take a few more minutes')
