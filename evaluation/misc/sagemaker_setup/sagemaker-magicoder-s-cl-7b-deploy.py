# Partial reference (incredibly bad documentation and outdated to top it off): https://github.com/huggingface/notebooks/blob/main/sagemaker/11_deploy_model_from_hf_hub/deploy_transformer_model_from_hf_hub.ipynb
import json
import sagemaker
import boto3
import time
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

try:
	role_name = input("Enter the AWS role name, of the AWS account which you will use to create a Sagemaker model/endpoint: ")
	sagemaker_execution_role = input("Enter the AWS SageMaker execution role, which will be used to create a Sagemaker model/endpoint: ") # e.g., AmazonSageMaker-ExecutionRole-20240504T121584
	my_session = boto3.session.Session(profile_name=role_name)
	sagemaker_session = sagemaker.Session(my_session)
	# print(sagemaker_session.boto_session.region_name)
	# role = sagemaker.get_execution_role(sagemaker_session=sagemaker_session)
	iam = my_session.client('iam')
	role = iam.get_role(RoleName=sagemaker_execution_role)['Role']['Arn']
	# print(sagemaker_session.get_caller_identity_arn())
	print(role)
	# role = sagemaker.get_execution_role()
	
	print("Finished sagemaker session setup.")
except ValueError:
	my_session = boto3.session.Session(profile_name=role_name)
	# sagemaker.Session(my_session)
	# sagemaker_session = 1
	iam = my_session.client('iam')
	role = iam.get_role(RoleName=sagemaker_execution_role)['Role']['Arn']
	print(role)

# Hub Model configuration. https://huggingface.co/models
hub = {
	'HF_MODEL_ID':'ise-uiuc/Magicoder-S-CL-7B', # https://huggingface.co/ise-uiuc/Magicoder-S-CL-7B
	# 'HF_MODEL_ID':'ise-uiuc/Magicoder-S-DS-6.7B', # https://huggingface.co/ise-uiuc/Magicoder-S-DS-6.7B?sagemaker_deploy=true
	# 'HF_MODEL_ID':'Salesforce/codegen-16B-multi', # https://huggingface.co/Salesforce/codegen-16B-multi?sagemaker_deploy=true
	'SM_NUM_GPUS': json.dumps(1), # add back later. for bert testing. # Only 1 GPU is avail for g5.2xlarge instance https://aws.amazon.com/ec2/instance-types/g5/
	'HF_TASK':'text-generation',
	# 'HF_TASK':'question-answering' # NLP task you want to use for predictions
	# "region": "us-east-1",
	# https://datamagiclab.com/input-validation-error-inputs-tokens-max_new_tokens-must-be-2048/ # https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/discussions/199
	"MAX_INPUT_LENGTH": '4000', # put here any value upto 32768 as per your requirement.
	"MAX_TOTAL_TOKENS": '5000',
	"MAX_BATCH_PREFILL_TOKENS": '5000',
	"MAX_BATCH_TOTAL_TOKENS": '5000',
}

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
	image_uri=get_huggingface_llm_image_uri("huggingface",version="1.4.2", 
	session=sagemaker_session
	),
	env=hub,
	role=role, 
	sagemaker_session=sagemaker_session,
	# Remove later, for bert testing:
	# transformers_version='4.37.0',
	# pytorch_version='2.1.0',
	# py_version='py310',
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
	initial_instance_count=1,
	instance_type="ml.g5.2xlarge",
	# instance_type="ml.g5.4xlarge",
	# instance_type="ml.m5.xlarge",
	container_startup_health_check_timeout=300, # add back later. for bert testing. 
  )

MAGICODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{}

@@ Response
""".format("You are TerraformAI, an AI agent that builds and deploys Cloud Infrastructure written in Terraform HCL. Generate a description of the Terraform program you will define, followed by a single Terraform HCL program in response to each of my Instructions. Make sure the configuration is deployable. Create IAM roles as needed. If variables are used, make sure default values are supplied. \n Here is the actual prompt: Create an AWS VPC resource with an Internet Gateway attached to it")
  
# send request
print(predictor.predict({
	"inputs": MAGICODER_PROMPT,
}))


time.sleep(60)

print(predictor.predict({
	"inputs": "My name is Julien and I like to",
}))
# # Cleanup:
# predictor.delete_model()
# predictor.delete_endpoint()