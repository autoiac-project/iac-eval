import replicate
import google.generativeai as genai
import os
import requests
import boto3
import sagemaker
import json
import subprocess
import time

# GPT evaluation generation
# set your API key here, need to be hided
def GPT3_5(preprompt, prompt, client):
    # message = query_message(prompt) if KNOWLEDGE_EMBEDDING_FLAG else prompt
    messages = [
        {"role": "system", "content": preprompt},
        {"role": "user", "content": prompt},
    ]

    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
            )
            # print(response.choices[0].message.content)
            # print("GPT3.5 eval done")
            return response.choices[0].message.content
        except Exception as e:
            s = str(e)
            if "Rate limit is exceeded" in s:
                time.sleep(30)
                continue
            else:
                return ""

def GPT4(preprompt, prompt, client):
    # print("current GPT4 working directory is",os.getcwd())
    # message = query_message(prompt) if KNOWLEDGE_EMBEDDING_FLAG else prompt
    messages = [
        {"role": "system", "content": preprompt},
        {"role": "user", "content": prompt},
    ]
    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=messages,
            )

            # print(response.choices[0].message.content)
            # print("GPT4 eval done")
            return response.choices[0].message.content
        except Exception as e:
            s = str(e)
            if "Rate limit is exceeded" in s:
                time.sleep(30)
                continue
            else:
                return ""

def Codellama7b(preprompt, prompt):
    for i in range(2):
        try:
            output = replicate.run(
                "meta/codellama-7b-instruct:aac3ab196f8a75729aab9368cd45ea6ad3fc793b6cda93b1ded17299df369332",
                input={
                    "top_k": 250,
                    "top_p": 0.95,
                    "prompt": prompt,
                    "max_tokens": 500,
                    "temperature": 0.95,
                    "system_prompt": preprompt,
                    "repeat_penalty": 1.1,
                    "presence_penalty": 0,
                    "frequency_penalty": 0,
                },
            )
            output_string = ""
            for item in output:
                output_string += item
            print(output_string)
            return output_string
        except Exception as e:
            s = str(e)
            print(s)
            if "status: 502" in s or "Prediction interrupted" in s:
                time.sleep(10)
    return ""


def Codellama13b(preprompt, prompt):
    for i in range(2):
        try:
            output = replicate.run(
                "meta/codellama-13b-instruct:a5e2d67630195a09b96932f5fa541fe64069c97d40cd0b69cdd91919987d0e7f",
                input={
                    "top_k": 250,
                    "top_p": 0.95,
                    "prompt": prompt,
                    "max_tokens": 500,
                    "temperature": 0.95,
                    "system_prompt": preprompt,
                    "repeat_penalty": 1.1,
                    "presence_penalty": 0,
                    "frequency_penalty": 0,
                },
            )
            output_string = ""
            for item in output:
                output_string += item
            print(output_string)
            return output_string
        except Exception as e:
            s = str(e)
            print(s)
            if "status: 502" in s or "Prediction interrupted" in s:
                time.sleep(10)
    return ""


def Codellama34b(preprompt, prompt):
    for i in range(2):
        try:
            output = replicate.run(
                "meta/codellama-34b-instruct:eeb928567781f4e90d2aba57a51baef235de53f907c214a4ab42adabf5bb9736",
                input={
                    "top_k": 250,
                    "top_p": 0.95,
                    "prompt": prompt,
                    "max_tokens": 500,
                    "temperature": 0.95,
                    "system_prompt": preprompt,
                    "repeat_penalty": 1.1,
                    "presence_penalty": 0,
                    "frequency_penalty": 0,
                },
            )
            output_string = ""
            for item in output:
                output_string += item
            print(output_string)
            return output_string
        except Exception as e:
            s = str(e)
            print(s)
            if "status: 502" in s or "Prediction interrupted" in s:
                time.sleep(10)
    return ""

# Codegen model variants: https://replicate.com/andreasjansson/codegen
def Codegen16b(preprompt, prompt):
    API_URL = "https://api-inference.huggingface.co/models/gpt2"
    HF_API_TOKEN = os.environ['HF_API_TOKEN ']
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json=prompt)

    return response.json()

# https://replicate.com/rhamnett/wizardcoder-34b-v1.0
def Wizardcoder34b(preprompt, prompt):
    for i in range(2):
        try:
            output = replicate.run(
                "rhamnett/wizardcoder-34b-v1.0:bae902bd8a4032fcf2295523b38da90aae7cc8ca2260e7ca9b8434a981d32278",
                input={
                    "top_k": 250,
                    "top_p": 0.95,
                    "prompt": prompt,
                    "max_tokens": 500,
                    "temperature": 0.95,
                    "system_prompt": preprompt,
                    "repeat_penalty": 1.1,
                    "presence_penalty": 0,
                    "frequency_penalty": 0,
                },
            )
            output_string = ""
            for item in output:
                output_string += item
            print(output_string)
            return output_string
        except Exception as e:
            s = str(e)
            print(s)
            if "status: 502" in s or "Prediction interrupted" in s:
                time.sleep(10)
    return ""
        

# WizardCoder-Python-34B-V1.0	https://replicate.com/lucataco/wizardcoder-python-34b-v1.0

# WizardCoder-33B-V1.1	: https://replicate.com/lucataco/wizardcoder-33b-v1.1-gguf
def Wizardcoder33b(preprompt, prompt):
    for i in range(2):
        try:
            output = replicate.run(
                "lucataco/wizardcoder-33b-v1.1-gguf:bbf93cee2c2b446f0ff426ae81a9b61c5ebd8972a21f734fe035513b6fafe615",
                input={
                    "top_k": 250,
                    "top_p": 0.95,
                    "prompt": prompt,
                    "max_tokens": 500,
                    "temperature": 0.95,
                    "system_prompt": preprompt,
                    "repeat_penalty": 1.1,
                    "presence_penalty": 0,
                    "frequency_penalty": 0,
                },
            )
            output_string = ""
            for item in output:
                output_string += item
            print(output_string)
            return output_string
        except Exception as e:
            s = str(e)
            print(s)
            if "status: 502" in s or "Prediction interrupted" in s:
                time.sleep(10)
    return ""

def Magicoder_S_CL_7B(preprompt, prompt):
    while True: 
        try:
            my_session = boto3.session.Session(profile_name=os.environ["AWS_ROLE_NAME"])
            session = sagemaker.Session(my_session)
            # endpoint_name = "huggingface-pytorch-tgi-inference-2024-05-06-03-21-48-837" # ise-uiuc/Magicoder-S-CL-7B endpoint
            endpoint_name = os.environ["MAGICODER_SAGEMAKER_ENDPOINT"] # ise-uiuc/Magicoder-S-CL-7B endpoint with 5000 MAX_TOTAL_TOKENS
            predictor = sagemaker.predictor.RealTimePredictor(endpoint_name=endpoint_name, sagemaker_session=session)
            MAGICODER_PROMPT = """
You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{}

@@ Response
    """.format(preprompt + "\n" + prompt)
            input_data = {
                "inputs": MAGICODER_PROMPT,
                'parameters': {"stop": ["<|endoftext|>", "</s>"], "max_new_tokens": 1000} # https://stackoverflow.com/a/76763217/13336187 # NOTE: 1800 max_new_tokens I initially used
            }
            payload = json.dumps(input_data).encode("utf-8")
            result = predictor.predict(payload, initial_args={'ContentType': 'application/json'}) # https://stackoverflow.com/a/65448063/13336187
            json_data = json.loads(result.decode('utf-8'))
            generated_text = json_data[0]['generated_text'].split("@@ Response")[1]
            return generated_text
        except Exception as e:
            s = str(e)
            if "Your invocation timed out" in s:
                print("Error encoutered. Considered model unable to output correct answer: ", s)
                return ""
            else:
                print("Error: ", s)
                raise e 

# WizardCoder-15B-V1.0	https://replicate.com/lucataco/wizardcoder-15b-v1.0
# def Wizardcoder33b(preprompt, prompt):
#     output = replicate.run(
#         "lucataco/wizardcoder-33b-v1.1-gguf:bbf93cee2c2b446f0ff426ae81a9b61c5ebd8972a21f734fe035513b6fafe615",
#         input={
#             "top_k": 250,
#             "top_p": 0.95,
#             "prompt": prompt,
#             "max_tokens": 500,
#             "temperature": 0.95,
#             "system_prompt": preprompt,
#             "repeat_penalty": 1.1,
#             "presence_penalty": 0,
#             "frequency_penalty": 0,
#         },
#     )
#     output_string = ""
#     for item in output:
#         output_string += item
#     print(output_string)
#     return output_string


# Gemini: https://ai.google.dev/gemini-api/docs/quickstart https://github.com/alibaba/CloudEval-YAML/blob/main/models/palm.py
def gemini(preprompt, prompt):
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
    # for m in genai.list_models():
    #     if 'generateContent' in m.supported_generation_methods:
    #         print(m.name)

    safety_settings = [
        {
            "category": "HARM_CATEGORY_DANGEROUS",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]

    model = genai.GenerativeModel('gemini-pro')
    # print(preprompt + "\n" + prompt)
    while True:
        try:
            response = model.generate_content(preprompt + "\n" + prompt, safety_settings=safety_settings)
            break
        except Exception as e:
            s = str(e)
            if "Resource has been exhausted" in s:
                print(s)
                print("^ Resource has been exhausted. Please wait for 100 seconds.")
                time.sleep(100)
    # print(len(response.candidates))
    # for candidate in response.candidates:
    #     print(len(candidate.content.parts))
        # print ([part.text for part in candidate.content.parts])
    
    # return response.text
    try:
        res = response.candidates[0].content.parts[0].text # https://github.com/google-gemini/generative-ai-python/issues/170
        return res 
    except:
        return ""

# gemini("You are TerraformAI, an AI agent that builds and deploys Cloud Infrastructure written in Terraform HCL. Generate a description of the Terraform program you will define, followed by a single Terraform HCL program in response to each of my Instructions. Make sure the configuration is deployable. Create IAM roles as needed.", "create a AWS codebuild project resource with example iam role, example GITHUB source, and a logs config")

# print(Magicoder_S_CL_7B("You are TerraformAI, an AI agent that builds and deploys Cloud Infrastructure written in Terraform HCL. Generate a description of the Terraform program you will define, followed by a single Terraform HCL program in response to each of my Instructions. Make sure the configuration is deployable. Create IAM roles as needed. If variables are used, make sure default values are supplied.", "Here is the actual prompt: Create an AWS VPC resource with an Internet Gateway attached to it"))

# print(Wizardcoder34b("You are TerraformAI, an AI agent that builds and deploys Cloud Infrastructure written in Terraform HCL. Generate a description of the Terraform program you will define, followed by a single Terraform HCL program in response to each of my Instructions. Make sure the configuration is deployable. Create IAM roles as needed. If variables are used, make sure default values are supplied.", "Here is the actual prompt: Create an AWS VPC resource with an Internet Gateway attached to it"))