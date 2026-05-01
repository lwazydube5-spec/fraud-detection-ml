"""
setup_model_monitor.py — Create baseline and schedule Model Monitor
===================================================================
Steps:
  1. Create baseline from training data in S3
  2. Schedule hourly monitoring job
  3. Monitor compares live predictions against baseline
     and raises violations when distributions drift

Usage:
    python setup_model_monitor.py
"""

import boto3
import json
import time

REGION        = 'us-east-1'
ACCOUNT_ID    = '604223541374'
ROLE_ARN      = f'arn:aws:iam::{ACCOUNT_ID}:role/SageMakerExecutionRole'
ENDPOINT_NAME = 'fraud-detection-endpoint'
S3_BUCKET     = f'fraud-detection-monitor-{ACCOUNT_ID}'
BASELINE_DATA = f's3://{S3_BUCKET}/baseline-data/fraud_data.csv'
BASELINE_RESULTS = f's3://{S3_BUCKET}/baseline-results/'
MONITOR_RESULTS  = f's3://{S3_BUCKET}/monitor-results/'
SCHEDULE_NAME    = 'fraud-detection-monitor'

sm = boto3.client('sagemaker', region_name=REGION)

print('=' * 58)
print('  Setting Up SageMaker Model Monitor')
print('=' * 58)

# ── Step 1 — create baseline processing job ───────────────────
print('\n► Creating baseline from training data...')
print(f'  Input  : {BASELINE_DATA}')
print(f'  Output : {BASELINE_RESULTS}')

BASELINE_JOB = 'fraud-baseline-job-v4'

try:
    sm.create_processing_job(
        ProcessingJobName = BASELINE_JOB,
        ProcessingResources = {
            'ClusterConfig': {
                'InstanceCount'   : 1,
                'InstanceType'    : 'ml.m5.large',
                'VolumeSizeInGB'  : 20,
            }
        },
        AppSpecification = {
            'ImageUri': f'156813124566.dkr.ecr.{REGION}.amazonaws.com/sagemaker-model-monitor-analyzer',
        },
        ProcessingInputs = [{
            'InputName'  : 'baseline_dataset_input',
            'AppManaged' : False,
            'S3Input'    : {
                'S3Uri'           : BASELINE_DATA,
                'LocalPath'       : '/opt/ml/processing/input/baseline',
                'S3DataType'      : 'S3Prefix',
                'S3InputMode'     : 'File',
                'S3DataDistributionType': 'FullyReplicated',
                'S3CompressionType': 'None',
            }
        }],
        ProcessingOutputConfig = {
            'Outputs': [{
                'OutputName' : 'monitoring_output',
                'AppManaged' : False,
                'S3Output'   : {
                    'S3Uri'        : BASELINE_RESULTS,
                    'LocalPath'    : '/opt/ml/processing/output',
                    'S3UploadMode' : 'EndOfJob',
                }
            }]
        },
        Environment = {
            'dataset_format'          : '{"csv": {"header": true}}',
            'dataset_source'          : '/opt/ml/processing/input/baseline',
            'output_path'             : '/opt/ml/processing/output',
            'publish_cloudwatch_metrics' : 'Disabled',
        },
        RoleArn = ROLE_ARN,
    )
    print(f'  ✓ Baseline job created: {BASELINE_JOB}')
except sm.exceptions.ResourceInUseException:
    print(f'  Baseline job already exists: {BASELINE_JOB}')

# ── Wait for baseline job to complete ────────────────────────
print('\n► Waiting for baseline job to complete (5-10 minutes)...')
while True:
    response = sm.describe_processing_job(ProcessingJobName=BASELINE_JOB)
    status   = response['ProcessingJobStatus']
    print(f'  Status: {status}')
    if status == 'Completed':
        break
    elif status == 'Failed':
        print(f'  Failed: {response.get("FailureReason")}')
        exit(1)
    time.sleep(30)

print(f'  ✓ Baseline complete → {BASELINE_RESULTS}')

# ── Step 2 — schedule monitoring ─────────────────────────────
print('\n► Scheduling hourly Model Monitor...')

try:
    sm.delete_monitoring_schedule(MonitoringScheduleName=SCHEDULE_NAME)
    print(f'  Deleted existing schedule: {SCHEDULE_NAME}')
    time.sleep(5)
except:
    pass

sm.create_monitoring_schedule(
    MonitoringScheduleName = SCHEDULE_NAME,
    MonitoringScheduleConfig = {
        'ScheduleConfig': {
            'ScheduleExpression': 'cron(0 * ? * * *)',  # hourly
        },
        'MonitoringJobDefinition': {
            'BaselineConfig': {
                'StatisticsResource' : {'S3Uri': f'{BASELINE_RESULTS}statistics.json'},
                'ConstraintsResource': {'S3Uri': f'{BASELINE_RESULTS}constraints.json'},
            },
            'MonitoringInputs': [{
                'EndpointInput': {
                    'EndpointName'    : ENDPOINT_NAME,
                    'LocalPath'       : '/opt/ml/processing/input/endpoint',
                    'S3InputMode'     : 'File',
                    'S3DataDistributionType': 'FullyReplicated',
                }
            }],
            'MonitoringOutputConfig': {
                'MonitoringOutputs': [{
                    'S3Output': {
                        'S3Uri'        : MONITOR_RESULTS,
                        'LocalPath'    : '/opt/ml/processing/output',
                        'S3UploadMode' : 'EndOfJob',
                    }
                }]
            },
            'MonitoringResources': {
                'ClusterConfig': {
                    'InstanceCount' : 1,
                    'InstanceType'  : 'ml.t3.medium',
                    'VolumeSizeInGB': 20,
                }
            },
            'MonitoringAppSpecification': {
                'ImageUri': f'156813124566.dkr.ecr.{REGION}.amazonaws.com/sagemaker-model-monitor-analyzer',
            },
            'RoleArn': ROLE_ARN,
        }
    }
)

print(f'  ✓ Monitor scheduled: {SCHEDULE_NAME}')
print(f'  ✓ Runs hourly')
print(f'  ✓ Results → {MONITOR_RESULTS}')

print('\n' + '=' * 58)
print('  Model Monitor Setup Complete')
print('=' * 58)
print(f'\n  Baseline data    : {BASELINE_DATA}')
print(f'  Baseline results : {BASELINE_RESULTS}')
print(f'  Monitor results  : {MONITOR_RESULTS}')
print(f'  Schedule         : hourly')
print('\n  Next: wait 1 hour for first monitoring run')
print('  Then check: aws sagemaker list-monitoring-executions')
print('              --monitoring-schedule-name fraud-detection-monitor\n')