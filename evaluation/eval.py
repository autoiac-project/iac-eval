import os
import shutil
import csv
import sys
from pathlib import Path
from openai import OpenAI
import numpy as np
import pandas as pd
import subprocess
import json
import getpass
import uuid
import re
import time
import logging
import models
import prompt_templates
import data
import math

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "retriever"))
)
import llama_index_retriever
import click
from typing import List, Tuple

DEFAULT_LOG_FILE = "logs/eval.log"
DELIMITERS = ["```hcl", "```json", "```HCL", "```Terraform", "```terraform", "```"]
# ``` needs to be in the end, as it is a final "else" case

# Default configurations if not specified
NUM_SAMPLES_PER_TASK = 20  # n
EVAL_MODELS = [
    "gpt3.5",
    "gpt4",
    "gemini-1.0-pro",
    "codellama-7b",
    "codellama-13b",
    "codellama-34b",
    "Magicoder_S_CL_7B",
    "Wizardcoder33b",
    "Wizardcoder34b",
]  # default to all available models

PROMPT_ENHANCEMENT_STRATS = ["RAG", "COT", "FSP", "multi-turn", ""]

class CustomFormatter(logging.Formatter):
    # https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
    grey = "\x1b[38;20m"
    cyan = "\x1b[36;20m"
    blue = "\x1b[34:20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: cyan + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger("iac-eval")  # get logger


# https://stackoverflow.com/questions/12507206/how-to-completely-traverse-a-complex-dictionary-of-unknown-depth
def dict_generator(indict, pre=None):
    pre = pre[:] if pre else []
    if isinstance(indict, dict):
        for key, value in indict.items():
            if isinstance(value, dict):
                for d in dict_generator(value, pre + [key]):
                    yield d
            elif isinstance(value, list) or isinstance(value, tuple):
                for v in value:
                    for d in dict_generator(v, pre + [key]):
                        yield d
            else:
                yield pre + [key, value]
    else:
        yield pre + [indict]


def rag_knowledge(Retriever, query):
    knowledge = ""
    questions = Retriever.generate_prompt_for_index(query)
    context = Retriever.query_documents(questions)
    for i, c in enumerate(context):
        if i >= 4:
            break
        knowledge += f"Context {i}: \n"
        knowledge += c
    return knowledge


# remove unwanted text for output
def remove_unwanted_characters(text):
    if text is None:
        return None

    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    text = ansi_escape.sub("", text)
    unwanted_pattern = re.compile(r"[^\x00-\x7F]+")  # Non-ASCII characters
    text = unwanted_pattern.sub("", text)

    return text


def delete_all_files_in_directory(folder):
    if not os.path.isdir(folder):
        return
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


# setup aws credential
def set_aws_credentials():
    # prompt the user for AWS credentials
    # set the credentials as environment variables
    if "AWS_ACCESS_KEY_ID" not in os.environ:
        aws_access_key_id = input("Enter AWS Access Key ID: ")
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    if "AWS_SECRET_ACCESS_KEY" not in os.environ:
        aws_secret_access_key = getpass.getpass("Enter AWS Secret Access Key: ")
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
    if "AWS_ROLE_NAME" not in os.environ:
        aws_role_name = input("Enter AWS Role Name to be used (press Enter to use the default value 'default'): " or "default")
        os.environ["AWS_ROLE_NAME"] = aws_role_name

def set_google_credentials():
    # For gemini:
    if "GOOGLE_API_KEY" not in os.environ:
        gemini_secret_access_key = getpass.getpass(
            "Enter Google API Key (for Gemini): "
        )
        os.environ["GOOGLE_API_KEY"] = gemini_secret_access_key


def set_replicate_credentials():
    # For Replicate:
    if "REPLICATE_API_TOKEN" not in os.environ:
        replicate_secret_access_key = getpass.getpass(
            "Enter Replicate API Key (for various models): "
        )
        os.environ["REPLICATE_API_TOKEN"] = replicate_secret_access_key


def set_huggingface_credentials():
    # For gemini:
    if "HF_API_TOKEN" not in os.environ:
        hf_secret_access_key = getpass.getpass(
            "Enter Huggingface API Token (for use of various models): "
        )
        os.environ["HF_API_TOKEN"] = hf_secret_access_key


# split the code for results
def separate_answer_and_code(text, delimiters=["```hcl"]):
    for delimiter in delimiters:
        # split the text at the point where the code block starts
        parts = text.split(delimiter)
        # print(len(parts))
        if len(parts) < 2:
            # delimiter not found, return original text and empty code
            answer = text.strip()
            code = ""
            continue
        # the first part is the answer
        answer = parts[0].strip()
        # the second part is the code, re-adding the "```hcl" and removing the trailing "```"
        code = parts[1].strip()
        code = code.rsplit("```", 1)[0].strip()
        if code != "":
            return answer, code
    return answer, code


# find each subdirectory
def list_all_subdirectories_and_eval(
    data_dir, base_eval_dir, final_eval_dir, PROMPT_ENHANCEMENT_STRAT, Retriever
):
    for path, _, _ in os.walk(data_dir):
        subdir = path.removeprefix(
            data_dir + "/"
        )  # FIX?: temp fix for now, the way to go is prob using pathlib
        create_evaluation_directories(subdir, base_eval_dir=base_eval_dir, final_eval_dir=final_eval_dir)
        file_dir = os.listdir(path)
        for file in file_dir:
            if file.endswith(".csv"):
                file_path = os.path.abspath(os.path.join(path, file))
                print(file_path)
                # Perform evaluation:
                for model in EVAL_MODELS:
                    # Note: Do not overwrite existing files, and do not evaluate if file exists already
                    eval_filepath, final_file_path, file_exists, NUM_EXISTING_SAMPLES = (
                        copy_csv_to_evaluation(
                            file_path,
                            subdir,
                            model,
                            PROMPT_ENHANCEMENT_STRAT,
                            base_eval_dir=base_eval_dir,
                            final_eval_dir=final_eval_dir,
                        )
                    )
                    if file_exists:
                        continue
                    read_models(
                        model,
                        PROMPT_ENHANCEMENT_STRAT,
                        NUM_EXISTING_SAMPLES,
                        eval_filepath,
                        final_file_path,
                        Retriever,
                    )


# function to create evaluation directories based on a given subdirectory
def create_evaluation_directories(subdir, base_eval_dir="evaluation/tmp", final_eval_dir="results"):
    for model in EVAL_MODELS:
        # construct the new directory path
        eval_dir_path = os.path.join(base_eval_dir, model, subdir)
        final_eval_dir_path = os.path.join(final_eval_dir, model, subdir)
        # create the directory if it does not exist
        os.makedirs(eval_dir_path, exist_ok=True)
        os.makedirs(final_eval_dir_path, exist_ok=True)


def make_column_names_unique(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [
            dup + "." + str(i) if i != 0 else dup for i in range(sum(cols == dup))
        ]
    df.columns = cols
    # print(cols.tolist())
    # while True:
    #     x=1
    return df


def fix_duplicate_columns(dest_file_path):
    """
    Deduplicated csv is written to original file path
    """
    # FIX?: this is ignoring the header in the original file and manually extracting the first line???
    df = pd.read_csv(dest_file_path, header=None)
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    df.reset_index(drop=True, inplace=True)

    if not df.columns.is_unique:  # First check if there are duplicate columns:
        df = make_column_names_unique(df)
        df.to_csv(dest_file_path, index=False, encoding="utf-8")
        logger.info(
            f"Evaluation file {dest_file_path} had duplicate columns, deduplicated them."
        )


def determine_eval_samples(dest_file_path):
    """
    Determine the number of samples currently present in a given evaluated dataset file.
    Also determines columns to remove (i.e., which are empty, because a previous evaluation run was not complete, which can only occur if copy_csv was successful but read_models was interrupted)
    Note: this is a variant of the same function used in llm-judge-eval.py
    """

    drop_cols = []

    df = pd.read_csv(dest_file_path, header=None)
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    df.reset_index(drop=True, inplace=True)

    num_samples = 0
    for col in df.columns:
        if "LLM Correct?" in col:
            if not pd.isnull(
                df[col].iloc[0]
            ):  # this means that the column is not empty (i.e., prev evaluation passed through successfully)
                num_samples += 1
            else:
                cols_to_drop = [
                    "LLM Output #",
                    "LLM Plannable? #",
                    "LLM Correct? #",
                    "LLM Plan Phase Error #",
                    "LLM OPA match phase Error #",
                    "LLM Notes #",
                ]  # drop all columns for this sample
                drop_cols.extend(
                    [col_base + str(col.split("#")[1]) for col_base in cols_to_drop]
                )

    return num_samples, drop_cols


# function to copy CSV to the new evaluation directory and rename it
def copy_csv_to_evaluation(
    src_file_path,
    subdir,
    model,
    PROMPT_ENHANCEMENT_STRAT,
    base_eval_dir="evaluation/tmp",
    final_eval_dir="results",
):
    # FIX?: this should be passed in as Path? if we are using Path, we should use it outside instead of os.
    # dataset_file_name = os.path.basename(src_file_path)
    dataset_file_name = Path(
        src_file_path
    ).stem  # https://stackoverflow.com/questions/678236/how-do-i-get-the-filename-without-the-extension-from-a-path-in-python
    eval_filename_prefix = "evaluation-dataset-for-{}".format(dataset_file_name)
    # eval_filename = eval_file_name_prefix + "-" + dataset_file_name

    eval_filename = (
        "{}-{}.csv".format(eval_filename_prefix, PROMPT_ENHANCEMENT_STRAT)
        if PROMPT_ENHANCEMENT_STRAT != ""
        else "{}.csv".format(eval_filename_prefix)
    )

    dest_file_path = os.path.join(base_eval_dir, model, subdir, eval_filename)

    final_file_path = os.path.join(final_eval_dir, model, subdir, eval_filename)

    df = pd.read_csv(src_file_path, header=None)
    # set the third row as the header
    # FIX?: if this is the case for every file, why are they even saved?

    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    # reset the index of the DataFrame
    df.reset_index(drop=True, inplace=True)

    num_eval_samples = 0  # current eval samples in the evaluated dataset file.

    # Skip if file already exists and contains enough samples:
    if os.path.isfile(dest_file_path):
        fix_duplicate_columns(dest_file_path)
        num_eval_samples, drop_cols = determine_eval_samples(dest_file_path)
        if num_eval_samples >= NUM_SAMPLES_PER_TASK:
            logger.info(
                f"Skipping evaluation for {dest_file_path} as file already exists, and it has enough samples."
            )
            return dest_file_path, None, True, num_eval_samples

        logger.info(
            f"Evaluation file {dest_file_path} already exists, but has not enough samples (required: {NUM_SAMPLES_PER_TASK}, existing: {num_eval_samples}), will continue evaluation."
        )
        # Replace df with existing evaluated dataset file df:
        # FIX?: this is ignoring the header in the original file and manually extracting the first line??? why?????
        df = pd.read_csv(dest_file_path, header=None)
        new_header = df.iloc[0]
        df = df[1:]
        df.columns = new_header
        # reset the index of the DataFrame
        df.reset_index(drop=True, inplace=True)

        if len(drop_cols) > 0:
            df.drop(columns=drop_cols, inplace=True)

    logger.info(f"Performing evaluation on {dest_file_path}.")
    # read the rest of the file starting from the fourth row without a header

    # add new columns only to df
    for i in range(num_eval_samples, NUM_SAMPLES_PER_TASK):
        for col_base in [
            "LLM Output #",
            "LLM Plannable? #",
            "LLM Correct? #",
            "LLM Plan Phase Error #",
            "LLM OPA match phase Error #",
            "LLM Notes #",
        ]:
            df[col_base + str(i)] = ""

    df.to_csv(dest_file_path, index=False, encoding="utf-8")

    return dest_file_path, final_file_path, False, num_eval_samples


# gpt result
def read_models(
    model,
    PROMPT_ENHANCEMENT_STRAT,
    NUM_EXISTING_SAMPLES,
    eval_filepath,
    final_filepath,
    Retriever,
):
    # read the first four lines to determine the header

    uuid_1 = get_unique_uuid()

    with open("prompt-templates/system-prompt.txt", "r") as file2:
        preprompt = file2.read()

    # Read from evaluation dataset file:
    df = pd.read_csv(eval_filepath, header=0)

    for index, row in df.iterrows():
        # iterate every row
        # find specific column
        model_evaluation(
            row,
            preprompt,
            df,
            index,
            model,
            PROMPT_ENHANCEMENT_STRAT,
            NUM_EXISTING_SAMPLES,
            Retriever,
            uuid_1,
        )

    df.to_csv(eval_filepath, index=False, encoding="utf-8")
    df.to_csv(final_filepath, index=False, encoding="utf-8")

    logger.info(f"Finished evaluation for {model}")


def get_plan_result_template():
    return {
        "terraform_plan_success": False,
        "terraform_output": "No output",
        "terraform_plan_error": "No error",
        "opa_evaluation_result": "No opa_result",
        "opa_evaluation_error": "None",
        "notes": "",
    }


def empty_code_error():
    logger.info("Plan considered failed since answer contains no code output.")
    plan_result = get_plan_result_template()
    plan_result["terraform_plan_success"] = False
    plan_result["terraform_output"] = "No output"
    plan_result["terraform_plan_error"] = "Empty code"
    plan_result["notes"] = (
        "Terraform plan considered failed since answer contains no code output."
    )
    return plan_result


def prompt_enhancements(prompt, PROMPT_ENHANCEMENT_STRAT, Retriever):
    if PROMPT_ENHANCEMENT_STRAT == "RAG":
        knowledge = rag_knowledge(Retriever, prompt)
        prompt = prompt_templates.RAG_prompt(knowledge, prompt)
    elif PROMPT_ENHANCEMENT_STRAT == "COT":
        prompt = prompt_templates.CoT_prompt(prompt)
    elif PROMPT_ENHANCEMENT_STRAT == "FSP":
        prompt = prompt_templates.FSP_prompt(prompt)
    else:
        prompt = "Here is the actual prompt: " + prompt
    return prompt


def model_evaluation(
    row,
    preprompt,
    df,
    index,
    model,
    PROMPT_ENHANCEMENT_STRAT,
    NUM_EXISTING_SAMPLES,
    Retriever,
    uuid_1,
):
    """
    Note: Multi-turn implies 2 turns only
    """
    prompt = row["Prompt"]

    # Skip empty rows
    if isinstance(row["Prompt"], float):
        if math.isnan(row["Prompt"]):
            return

    prompt = prompt_enhancements(prompt, PROMPT_ENHANCEMENT_STRAT, Retriever)

    policy_file = row["Rego intent"]
    num_correct = 0
    logger.info(f"Begin testing model: {model}")
    for i in range(NUM_EXISTING_SAMPLES, NUM_SAMPLES_PER_TASK):
        multi_turn_count = 1
        while True:
            is_empty_code = False
            logger.info(f"Sample {i} for model {model}")
            logger.info(f"Preprompt: {preprompt}")
            logger.info(f"Prompt: {prompt}")
            if model == "gpt4":
                text = models.GPT4(preprompt, prompt, gpt_client)
            elif model == "gpt3.5":
                text = models.GPT3_5(preprompt, prompt, gpt_client)
            elif model == "gemini-1.0-pro":
                text = models.gemini(preprompt, prompt)
            elif model == "codellama-13b":
                text = models.Codellama13b(preprompt, prompt)
            elif model == "codellama-7b":
                text = models.Codellama7b(preprompt, prompt)
            elif model == "codellama-34b":
                text = models.Codellama34b(preprompt, prompt)
            elif model == "Magicoder_S_CL_7B":
                text = models.Magicoder_S_CL_7B(preprompt, prompt)
            elif model == "Wizardcoder33b":
                text = models.Wizardcoder33b(preprompt, prompt)
            elif model == "Wizardcoder34b":
                text = models.Wizardcoder34b(preprompt, prompt)

            logger.info(f"Model raw output: {text}")

            answer, code = separate_answer_and_code(text, DELIMITERS)
            if code == "":
                logger.error("Error: Answer contains no code, skipping eval_pipeline.")
                is_empty_code = True
            logger.info("Answer is: {}".format(answer))
            logger.info("Code is: {}".format(code))

            df.at[index, "LLM Output #" + str(i)] = text
            if is_empty_code:
                x = empty_code_error()
            else:
                x = eval_pipeline(code, policy_file, prompt, uuid_1)
            df.at[index, "LLM Plannable? #" + str(i)] = x["terraform_plan_success"]
            df.at[index, "LLM Correct? #" + str(i)] = x["opa_evaluation_result"]
            df.at[index, "LLM Plan Phase Error #" + str(i)] = x["terraform_plan_error"]
            df.at[index, "LLM OPA match phase Error #" + str(i)] = x[
                "opa_evaluation_error"
            ]
            df.at[index, "LLM Notes #" + str(i)] = x["notes"]
            logging.info("Plan Result Summary:")
            for key, value in x.items():
                logging.info(f"{key}: {value}")

            if x["opa_evaluation_result"] == "success":
                num_correct += 1
                break
            elif PROMPT_ENHANCEMENT_STRAT == "multi-turn":
                if multi_turn_count == 2:  # only do 2 turns
                    break
                multi_turn_count += 1
                if code == "":
                    continue
                preprompt = prompt_templates.multi_turn_system_prompt()
                if not x["terraform_plan_success"]:
                    prompt = prompt_templates.multi_turn_plan_error_prompt(
                        row["Prompt"], code, x["terraform_plan_error"]
                    )
                elif x["opa_evaluation_result"] == "Failure":
                    prompt = prompt_templates.multi_turn_rego_error_prompt(
                        row["Prompt"], code, policy_file, x["opa_evaluation_error"]
                    )
                continue
            else:
                break


# used to modify main.tf
def write_to_terraform(result, terraform_dir="./terraform_config"):
    # define the path to the main.tf file
    os.makedirs(terraform_dir, exist_ok=True)
    terraform_file_path = terraform_dir + "/main.tf"
    # open the file in write mode ('w') and write the result to it
    # print("CWD", os.getcwd())
    logger.debug("CWD: {}".format(os.getcwd()))
    with open(terraform_file_path, "w+", encoding="utf-8", errors="ignore") as file:
        file.write(result)
    # print(f"Updated main.tf at {terraform_file_path}")
    logger.info(f"Updated main.tf at {terraform_file_path}")


# used for modify policy.rego
def write_to_rego(policy_content, rego_policy_filepath):
    # define the path to the policy.rego file
    # rego_file_path = "./rego_config/policy.rego"
    # ensure the rego_config directory exists
    os.makedirs(os.path.dirname(rego_policy_filepath), exist_ok=True)
    # open the file in write mode ('w') and write the policy content to it
    with open(rego_policy_filepath, "w", encoding="utf-8", errors="ignore") as file:
        file.write(policy_content)
    logger.info(f"Updated policy.rego at {rego_policy_filepath}")


def get_unique_uuid():
    terraform_dir_prefix = "./tmp/terraform_config/"
    uuid_1 = ""
    while True:
        uuid_1 = str(uuid.uuid4())
        terraform_dir = terraform_dir_prefix + uuid_1
        if os.path.isfile(terraform_dir):
            continue
        else:
            break
    return uuid_1


def eval_pipeline(result, policy_file, prompt, uuid_1):
    """
    TF Plan -> OPA Rego
    """

    # object to store the results
    plan_result = get_plan_result_template()
    # Clear the terraform_config directory

    # Generate unique filename suffix:
    terraform_dir_prefix = "./tmp/terraform_config/"
    terraform_dir = terraform_dir_prefix + uuid_1

    delete_all_files_in_directory(terraform_dir)

    # write result to main.tf
    write_to_terraform(result, terraform_dir)

    # run terraform plan and capture the output and errors
    plan_file = "plan.out"
    plan_output, plan_error, plan_success = run_terraform_plan(
        terraform_dir, plan_file, prompt
    )
    # print("plan output: ", plan_output)
    logger.info("plan_output: {}".format(plan_output))
    # print("plan_error: ", plan_error)
    logger.error("plan_error occurred: {}".format(plan_error), exc_info=True)
    # print("plan_success: ", plan_success)
    logger.debug("plan_success: {}".format(plan_success))
    plan_output = remove_unwanted_characters(plan_output)
    plan_error = remove_unwanted_characters(plan_error)
    plan_result["terraform_plan_success"] = plan_success
    plan_result["terraform_output"] = plan_output
    plan_result["terraform_plan_error"] = plan_error

    if plan_success:
        # print("Plan succeeded.")
        logger.info("Plan succeeded.")
        # go to opa rego check
        rego_dir = "./tmp/rego_config/" + uuid_1
        rego_policy_filepath = rego_dir + "/policy.rego"
        write_to_rego(policy_file, rego_policy_filepath)
        generate_terraform_plan_json("plan.json", plan_file, terraform_dir)
        tf_json_plan_filepath = os.path.join(terraform_dir, "plan.json")

        # run OPA evaluation and capture the result
        opa_result, opa_error = OPA_Rego_evaluation(
            tf_json_plan_filepath, rego_policy_filepath
        )
        # print("OPA result: ", opa_result)
        logger.info("OPA result: {}".format(opa_result))
        # print("OPA current directory: ", os.getcwd())
        # print("OPA error: ", opa_error)
        logger.error("OPA error occurred: {}".format(opa_error), exc_info=True)
        plan_result["opa_evaluation_result"] = opa_result
        plan_result["opa_evaluation_error"] = opa_error
    else:
        # print("Plan failed.")
        logger.info("Plan failed.")
        plan_result["notes"] = "Terraform plan failed."
    return plan_result


def run_terraform_plan(terraform_directory, plan_file, prompt):
    cur_dir = os.getcwd()
    # change to the Terraform directory
    os.chdir(terraform_directory)
    # run init before plan
    subprocess.run(["terraform", "init"], capture_output=True, text=True)

    # run 'terraform plan'
    # result = subprocess.run(["terraform", "plan"], capture_output=True, text=True)

    result_returned = False
    # generate Terraform plan with the -no-color flag
    for i in range(2):  # try twice
        try:
            result = subprocess.run(
                ["terraform", "plan", "-out", plan_file, "-no-color"],
                capture_output=True,
                text=True,
                timeout=100,  # 5 minutes timeout (assume failed if timeout)
            )
            if "Inconsistent dependency lock file" in result.stderr:
                subprocess.run(["terraform", "init"], capture_output=True, text=True)
                time.sleep(10)
                continue

            result_returned = True
            break
        except Exception as e:
            logging.error(
                'Error occurred for prompt "{}": {}'.format(prompt, e), exc_info=True
            )

    # Return to parent directory
    os.chdir(cur_dir)

    if not result_returned:
        return "Plan timed-out. No output", "Plan timed-out. No error", False

    # check the exit code and return the output, error message, and success flag
    success = result.returncode == 0
    output = result.stdout if not success else "success"
    error = result.stderr if not success else "No error"
    return output, error, success


def check_if_rego_v1(policy_file):
    with open(policy_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            # if "package" in line:
            if "import rego.v1" in line:
                return True
    return False


def OPA_Rego_evaluation(plan_file, policy_file):
    # print("opa current directory: ", os.getcwd())
    # assumes the current working directory is correct
    try:
        is_rego_v1 = check_if_rego_v1(policy_file)

        if is_rego_v1:
            result = subprocess.run(
                [
                    "opa",
                    "eval",
                    "--v1-compatible",
                    "-i",
                    plan_file,
                    "-d",
                    policy_file,
                    "data",
                ],
                capture_output=True,
                text=True,
            )
        else:
            result = subprocess.run(
                [
                    "opa",
                    "eval",
                    "-i",
                    plan_file,
                    "-d",
                    policy_file,
                    "data",
                ],
                capture_output=True,
                text=True,
            )
    except Exception as e:
        opa_result = "OPA exception occurred."
        opa_error = "OPA exception occurred: {}".format(e)
        return opa_result, opa_error

    # check the exit code and return the result and error message
    # success = result.returncode == 0
    # key_val = next(iter( json.loads(result.stdout)["result"][0]["expressions"][0]["value"].items() ))
    # get the first key-value pair: https://stackoverflow.com/a/39292086/13336187
    # key_val = key_val[1]
    # print(key_val)
    results = [
        i[-1]
        for i in dict_generator(
            json.loads(result.stdout)["result"][0]["expressions"][0]["value"]
        )
    ]
    # print(results)
    # print(key_val)
    success = False if False in results else True
    opa_result = "Success" if success else "Failure"
    opa_error = "No error"
    if not success:
        opa_error = "Rule violation found. OPA complete output logged here: " + str(
            json.loads(result.stdout)
        )
    # print("OPA error: ", opa_error)
    return opa_result, opa_error


def generate_terraform_plan_json(
    output_json_file, plan_file="plan.out", terraform_dir="./terraform_config"
):
    try:
        cur_dir = os.getcwd()
        os.chdir(terraform_dir)
        # init_result = subprocess.run(["terraform", "init"], check=True)

        # # generate Terraform plan with the -no-color flag
        # plan_file = "plan.out"
        # plan_result = subprocess.run(
        #     ["terraform", "plan", "-out", plan_file, "-no-color"], check=True
        # )

        # convert the plan to JSON and store it
        with open(
            output_json_file, "w", encoding="utf-8", errors="ignore"
        ) as json_file:
            subprocess.run(
                ["terraform", "show", "-json", plan_file], check=True, stdout=json_file
            )
        os.chdir(cur_dir)
    except subprocess.CalledProcessError as e:
        # print(f"An error occurred: {e}")
        logger.error("An error occurred: {}".format(e), exc_info=True)
        return None


def read_eval_models(ctx: click.Context, _, arg: str) -> List[str]:
    return arg.split(sep=",")


def read_config_file(path: Path) -> Tuple[int, List[str]]:
    try:
        with path.open("r") as file:
            config = json.load(file)
            return config.get("samples", NUM_SAMPLES_PER_TASK), config.get(
                "models", EVAL_MODELS
            )
    except BaseException:
        print("Invalid config file.", file=sys.stderr)
        sys.exit(1)


def set_logger(log_file: Path):
    """
    Set global logger with log file
    """
    logger.setLevel(logging.DEBUG)
    # Setup File handler: https://stackoverflow.com/a/24507130/13336187
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(CustomFormatter())
    file_handler.setLevel(logging.DEBUG)
    # Setup Stream Handler (i.e. console)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    # Log to both file and console:
    logger.addHandler(ch)
    logger.addHandler(file_handler)


def setup_gpt_client():
    global gpt_client
    global embeddings_model
    if "OPENAI_API_KEY" not in os.environ:
        api_key = input("Enter OpenAI API key:")
        os.environ["OPENAI_API_KEY"] = api_key

    api_key = os.environ["OPENAI_API_KEY"]

    gpt_client= OpenAI(api_key=api_key)


def setup_magicoder_params():
    if "MAGICODER_SAGEMAKER_ENDPOINT" not in os.environ:
        endpoint = input("Enter Magicoder Sagemaker endpoint:") # E.g., "huggingface-pytorch-tgi-inference-2024-05-09-15-37-08-362"
        os.environ["MAGICODER_SAGEMAKER_ENDPOINT"] = endpoint


@click.command()
@click.option(
    "--samples",
    "-s",
    type=int,
    help="Number of samples per task.",
    default=NUM_SAMPLES_PER_TASK,
)
@click.option(
    "--quick-test",
    "-q",
    "quick_test",
    is_flag=True,
    help="Perform quick evaluation on only 2 rows within the main dataset.",
    default=False,
)
@click.option(
    "--models",
    "-m",
    type=str,
    help=f"List of evaluation models. Available models: {' '.join(EVAL_MODELS)}",
    callback=read_eval_models,
    default=EVAL_MODELS,
)
@click.option(
    "--config",
    "--file",
    "-c",
    "-f",
    type=click.Path(path_type=Path, exists=True),
    help="Path to config file for command line options.",
)
@click.option(
    "--log-file",
    "-l",
    "log_file",
    type=click.Path(path_type=Path),
    help="Path to log file.",
    default=DEFAULT_LOG_FILE,
)
@click.option(
    "--enhance-strat",
    "-e",
    "enhance_strat",
    type=click.Choice(PROMPT_ENHANCEMENT_STRATS),
    help=f"Prompt enhancement strategy. Available strategies: {' '.join(PROMPT_ENHANCEMENT_STRATS)}",
    default="",
)
# @click.argument("enhance_strat", nargs=1, type=str, default="")
def main(
    samples: int, models: List[str], config: Path, log_file: Path, enhance_strat: str, quick_test: bool
):
    """
    Evaluate models.
    Available enhancement strategy: "RAG", "COT", "FSP", or "multi-turn".
    Config file takes precedence over command line options.
    """  # FIX

    if config is not None:
        samples, models = read_config_file(config)

    # changing config variables basing on command line options
    global NUM_SAMPLES_PER_TASK
    NUM_SAMPLES_PER_TASK = samples
    global EVAL_MODELS
    EVAL_MODELS = models

    set_logger(log_file)

    print(samples)
    print(models)

    PROMPT_ENHANCEMENT_STRAT = (
        enhance_strat
        # FIX?: should this be changed to an option instead of argument
    )

    # Setup environment variables:
    set_aws_credentials()
    set_replicate_credentials()
    # set_huggingface_credentials()

    if "gemini-1.0-pro" in models:
        set_google_credentials()

    if "gpt3.5" in models or "gpt4" in models:
        setup_gpt_client()

    if "Magicoder_S_CL_7B" in models:
        setup_magicoder_params()

    # Setup retriever:
    if "RAG" in PROMPT_ENHANCEMENT_STRAT:
        Retriever = llama_index_retriever.Retriever(
            stored_index="../retriever/aws-index",
            path="../retriever/terraform-provider-aws/website/docs/r",
        )
    else:
        Retriever = None

    # Import dataset:
    if not quick_test:
        data.import_dataset()
    else:
        data.import_dataset(quick_test=True)

    # specify the directory you want to search
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    base_eval_dir = os.path.join(
        data_dir, "..", "evaluation/tmp"
    )  # should be the same as script_dir
    final_eval_dir = os.path.join(data_dir, "..", "evaluation/results")
    # Create evaluation directories for each data directory
    # and perform model evaluation:
    list_all_subdirectories_and_eval(
        data_dir, base_eval_dir, final_eval_dir, PROMPT_ENHANCEMENT_STRAT, Retriever
    )


if __name__ == "__main__":
    main()
