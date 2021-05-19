
import os, shutil, tarfile
## 1. select pretrained models by language and copy them to target folder
model_folder='./model'
version='v1'
model_name=f'model_{version}'
target_folder='./'
tar_name=f'bert_sa_{version}.tar.gz'
s3_tar_key='sbert/'+tar_name
if not os.path.exists(model_name):
    shutil.copytree(os.path.join(model_folder, model_name), os.path.join(target_folder, model_name))
print('directory copied')
os.rename(os.path.join(target_folder, model_name), os.path.join(target_folder, 'ko-KR'))


print('rename complete')
## 2. rename them and zip them in one tar file
with tarfile.open(tar_name, "w:gz") as tar:
    tar.add(os.path.join(target_folder, 'ko-KR'))
    tar.add(os.path.join(target_folder, 'code'))

print('tar make complete')
if os.path.exists(os.path.join(target_folder, 'ko-KR')):
    shutil.rmtree(os.path.join(target_folder, 'ko-KR'))
print('delete temp folders')


!aws s3 cp {tar_name} {'s3://di-meta-model/'+s3_tar_key}

from datetime import datetime
model_deploy_time=datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

model_s3_url=f's3://di-meta-model/{s3_tar_key}'
postfix=f'bert-sa-{version}-{model_deploy_time}'

sg_model_name=f'model-{postfix}'
sg_endpoint_config_name=f'endpoint-config-{postfix}'
sg_endpoint_name=f'endpoint-{postfix}'

import boto3
client = boto3.client('sagemaker')
from sagemaker import get_execution_role
role=get_execution_role()

image = '301217895009.dkr.ecr.us-west-2.amazonaws.com/sagemaker-inference-pytorch:1.5-cpu-py3'
create_model_api_response = client.create_model(
                                ModelName=sg_model_name,
                                    PrimaryContainer={
                                        'Image': image,
                                        'ModelDataUrl': model_s3_url,
                                        'Environment': {
                                            'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                                            'SAGEMAKER_PROGRAM': 'inference.py',
                                            'SAGEMAKER_REGION': 'us-west-2',
                                            'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/model/code',
                                            'MMS_DEFAULT_RESPONSE_TIMEOUT': '500',
                                            'SAGEMAKER_MODEL_SERVER_WORKERS': '4'
                                        }
                                    },
                               ExecutionRoleArn=role
                            )

print ("create_model API response", create_model_api_response)

create_endpoint_config_api_response = client.create_endpoint_config(
                                            EndpointConfigName=sg_endpoint_config_name,
                                            ProductionVariants=[
                                                {
                                                    'VariantName': 'AllTraffic',
                                                    'ModelName': sg_model_name,
                                                    'InitialInstanceCount': 1,
                                                    'InstanceType': 'ml.m4.xlarge'
                                                },
                                            ]
                                        )

print ("create_endpoint_config API response", create_endpoint_config_api_response)

# create sagemaker endpoint
create_endpoint_api_response = client.create_endpoint(
                                    EndpointName=sg_endpoint_name,
                                    EndpointConfigName=sg_endpoint_config_name,
                                )

print ("create_endpoint API response", create_endpoint_api_response)

delete_model_api_response = client.delete_model(ModelName=sg_model_name)
print(f'delete model api response : {delete_model_api_response}')

delete_endpoint_config_api_response = client.delete_endpoint_config(EndpointConfigName=sg_endpoint_config_name)
print(f'delete endpoint config api response : {delete_endpoint_config_api_response}')

delete_endpoint_api_response = client.delete_endpoint(EndpointName=sg_endpoint_name)
print(f'delete endpoint api response : {delete_endpoint_api_response}')