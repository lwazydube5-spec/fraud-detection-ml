import boto3
import json
import time

runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

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

for i in range(10):
    response = runtime.invoke_endpoint(
        EndpointName = 'fraud-detection-endpoint',
        ContentType  = 'application/json',
        Body         = json.dumps(claim),
    )
    result = json.loads(response['Body'].read().decode())
    print(f'Prediction {i+1}: {result["fraud_probability"]}')

print('✓ 10 predictions sent')
print('Waiting 2 minutes for S3 capture...')
time.sleep(120)

s3 = boto3.client('s3', region_name='us-east-1')
response = s3.list_objects_v2(
    Bucket = 'fraud-detection-monitor-604223541374',
    Prefix = 'data-capture/',
)

if response.get('Contents'):
    print('✓ Data captured to S3:')
    for obj in response['Contents']:
        print(f'  {obj["Key"]}')
else:
    print('No captured data yet — check S3 console directly')
