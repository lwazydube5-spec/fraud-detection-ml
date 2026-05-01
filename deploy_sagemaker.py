"""
deploy_sagemaker.py — Deploy fraud detection model to SageMaker
Uses boto3 directly instead of sagemaker SDK for Python 3.14 compatibility.
"""

import boto3
import json
import time

# ── Config ────────────────────────────────────────────────────────────────
REGION         = 'us-east-1'
ACCOUNT_ID     = '604223541374'
ROLE_ARN       = f'arn:aws:iam::{ACCOUNT_ID}:role/SageMakerExecutionRole'
IMAGE_URI      = f'{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/fraud-api:v2'
MODEL_NAME     = 'fraud-detection-model'
CONFIG_NAME    = 'fraud-detection-config'
ENDPOINT_NAME  = 'fraud-detection-endpoint'
INSTANCE_TYPE  = 'ml.t2.medium'
S3_BUCKET      = f'fraud-detection-monitor-{ACCOUNT_ID}'
S3_CAPTURE_URI = f's3://{S3_BUCKET}/data-capture/'

# ── Clients ───────────────────────────────────────────────────────────────
sm = boto3.client('sagemaker', region_name=REGION)

print('=' * 58)
print('  Deploying Fraud Detection to SageMaker')
print('=' * 58)
print(f'\n  Image     : {IMAGE_URI}')
print(f'  Endpoint  : {ENDPOINT_NAME}')
print(f'  Instance  : {INSTANCE_TYPE}')
print(f'  Capture   : {S3_CAPTURE_URI}')

# ── Step 1 — create model ─────────────────────────────────────────────────
print('\n► Creating SageMaker model...')
try:
    sm.delete_model(ModelName=MODEL_NAME)
    print(f'  Deleted existing model: {MODEL_NAME}')
except:
    pass

sm.create_model(
    ModelName        = MODEL_NAME,
    ExecutionRoleArn = ROLE_ARN,
    PrimaryContainer = {
        'Image': IMAGE_URI,
    },
)
print(f' Model created: {MODEL_NAME}')

# ── Step 2 — create endpoint config with data capture ────────────────────
print('\n► Creating endpoint config with data capture...')
try:
    sm.delete_endpoint_config(EndpointConfigName=CONFIG_NAME)
    print(f'  Deleted existing config: {CONFIG_NAME}')
except:
    pass

sm.create_endpoint_config(
    EndpointConfigName = CONFIG_NAME,
    ProductionVariants = [{
        'VariantName'         : 'primary',
        'ModelName'           : MODEL_NAME,
        'InitialInstanceCount': 1,
        'InstanceType'        : INSTANCE_TYPE,
    }],
    DataCaptureConfig = {
        'EnableCapture'       : True,
        'InitialSamplingPercentage': 100,
        'DestinationS3Uri'    : S3_CAPTURE_URI,
        'CaptureOptions'      : [
            {'CaptureMode': 'Input'},
            {'CaptureMode': 'Output'},
        ],
        'CaptureContentTypeHeader': {
            'JsonContentTypes': ['application/json'],
        },
    },
)
print(f'  ✓ Endpoint config created: {CONFIG_NAME}')

# ── Step 3 — create or update endpoint ───────────────────────────────────
print('\n► Deploying endpoint (this takes 5-10 minutes)...')
try:
    sm.create_endpoint(
        EndpointName       = ENDPOINT_NAME,
        EndpointConfigName = CONFIG_NAME,
    )
    print(f'  Creating new endpoint: {ENDPOINT_NAME}')
except sm.exceptions.ClientError as e:
    if 'Cannot create already existing endpoint' in str(e):
        sm.update_endpoint(
            EndpointName       = ENDPOINT_NAME,
            EndpointConfigName = CONFIG_NAME,
        )
        print(f'  Updating existing endpoint: {ENDPOINT_NAME}')
    else:
        raise

# ── Step 4 — wait for endpoint to be InService ───────────────────────────
print('\n► Waiting for endpoint to be InService...')
while True:
    response = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
    status   = response['EndpointStatus']
    print(f'  Status: {status}')
    if status == 'InService':
        break
    elif status == 'Failed':
        print(f'  Failed reason: {response.get("FailureReason")}')
        break
    time.sleep(30)

print(f'\n√ Endpoint deployed : {ENDPOINT_NAME}')
print(f'√ Data capture      : {S3_CAPTURE_URI}')
print('\nNext step: create baseline and schedule Model Monitor')
