# Assumption: potentially multiple files within the same eval folder (e.g.,results-for-iac-eval (backup)/standard/codellama-7b/george), where each file must contains n (or more) number of evaluation passes
# Task: combines all "output" related columns and combines them into one single dataframe, saving it to a new folder within pass_k_calculation/combined_eval_dataset/ with a folder layout of (pass_k_calculation/combined_eval_dataset/standard/codellama-7b/george/), then passes through each file and calculates pass@k metrics for various k assuming some fixed N. And constructs an output_composition_dict.json similar to that used in ablation/
import csv
import pandas as pd
import logging
import os
from pathlib import Path
from openai import AzureOpenAI
import pandas as pd
import json
import subprocess
import copy
from io import StringIO
import re
import sys
import math
import click
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import eval
import numpy as np
import metrics

DELIMITERS = ["```hcl", "```json", "```HCL", "```Terraform", "```terraform", "```"] # ``` needs to be in the end, as it is a final "else" case

LOG_FILE = "logs/eval-repair.log"

class CustomFormatter(logging.Formatter):
    # https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
    grey = "\x1b[38;20m"
    cyan = "\x1b[36;20m"
    blue = "\x1b[34:20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: cyan + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

logger = logging.getLogger("iac-eval-repair")
logger.setLevel(logging.DEBUG)
#Setup File handler: https://stackoverflow.com/a/24507130/13336187 
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(CustomFormatter())
file_handler.setLevel(logging.DEBUG)
#Setup Stream Handler (i.e. console)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
# Log to both file and console:
logger.addHandler(ch)
logger.addHandler(file_handler)

def pretty_json(obj):
    return json.dumps(obj, indent=4, default=str)

def estimator(n: int, c: int, k: int) -> float:
    """
    Calculates 1 - comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def make_column_names_unique(df):
    for i in range(2): # repeat twice, since after the first time we may still have duplicates: e.g., if #0.1 already exists, and in the initial run we have duplicate #0
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique(): 
            cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
        df.columns = cols
    # print(cols.tolist())
    # while True:
    #     x=1
    return df

def fix_duplicate_columns(dest_file_path):
    """
        Deduplicated dataframe is returned to the user. dest_file_path is not overwritten. 
    """
    df = pd.read_csv(dest_file_path, header=None)
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    df.reset_index(drop=True, inplace=True)

    if not df.columns.is_unique: # First check if there are duplicate columns:
        df = make_column_names_unique(df)
        logger.info(f"Evaluation file {dest_file_path} had duplicate columns, deduplicated them. evaluation file not overwritten.")
    return df 

def list_all_subdirectories(composition_dict, data_dir, excluded_dirs, strats=["multi-turn", "RAG", "COT", "FSP"]):
    # onlyfiles = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) if dirpath not in excluded_dirs in os.walk(data_dir) for f in filenames] 
    onlyfiles = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        # print(dirpath)
        skip_dirname = False
        # if "standard" in dirpath:
        #     print(dirpath)
        for dir1 in excluded_dirs:
            if dir1 in dirpath:
                skip_dirname = True

        if not skip_dirname:
            for f in filenames: 
                skip_file = False
                for exc in excluded_dirs:
                    if exc in f:
                        skip_file = True
                if not skip_file:
                    onlyfiles.append(os.path.join(dirpath, f))
    
    # onlyfiles = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(data_dir) for f in filenames] 
    # print(onlyfiles)
    onlyfilescsv = [f for f in onlyfiles if f.endswith(".csv")]

    # composition_dict = {
    #     "gpt4": {
    #         "COT": {
    #             "weijun": [],

    # Append to composition dict: 
    for file1 in onlyfilescsv:
        included = False
        is_standard = True
        for i in strats:
            if i in file1:
                is_standard = False
        
        for model, model_dict in composition_dict.items():
            for strat, student_dict in model_dict.items():
                if is_standard:
                    if strat != "Standard":
                        continue    
                for student, dataset_list in student_dict.items():
                    is_file = False
                    if is_standard:
                        if student in file1 and model in file1:
                            is_file = True
                    else:
                        if student in file1 and strat in file1 and model in file1:
                            is_file = True
                    if is_file:
                        if not included:
                            composition_dict[model][strat][student].append(file1)
                            included = True
                        else:
                            print("WARNING: This file attempted to be included multiple times: ", file1)
        if not included:
            print("WARNING: This file was excluded: ", file1)

    return onlyfilescsv, composition_dict

def create_new_dirs_for_metric(new_base_difficulty_dir, base_dataset_files):
    os.makedirs(new_base_difficulty_dir, exist_ok=True)
    for file in base_dataset_files:
        containing_folder = os.path.basename(os.path.dirname(file)) # e.g., weijun
        filename = os.path.basename(file) # e.g., plain-dataset.csv
        # create the directory if it does not exist
        new_dir_path = os.path.join(new_base_difficulty_dir, containing_folder)
        # print(new_dir_path)
        os.makedirs(new_dir_path, exist_ok=True)

def metric_calculation_loop(composition_dict, dataset_dir, output_composition_dict_filepath, pass_k_parameters):

    """ 
    NOTE: pass@k calculation is limited to standard only for now. 
    NOTE: if pass_k_parameters is not None, then we will only calculate pass@k scores, and for standard only. 

    pass_k_parameters format example: {
        "n" = 5,
        "k" = [1,2,5],
    }
    Output format example (added in place to composition_dict): {
        "gpt4": {
            "Standard": {
                ..., # whatever that is already in composition_dict

                "iac_eval_accuracy": 0.3,
                "pass@k" : { # iac-eval
                    "n": 5 # just for record
                    "1": 0.5,
                    "2": 0.6,
                    "5": 0.7,
                },
                "pass@k-tf-plan" : {
                    "n": 5 # just for record
                    "1": 0.5,
                    "2": 0.6,
                    "5": 0.7,
                },
            }, 
            "COT": ...
        },
        "gpt3.5": {
            ...
        },
        ...
    }
    """

    output_composition_dict = copy.deepcopy(composition_dict) # https://stackoverflow.com/questions/5105517/deep-copy-of-a-dict-in-python

    for model, model_dict in composition_dict.items():
        for strat, student_dict in model_dict.items():   

            pass_k_scores = {}
            pass_k_scores_tf_plan = {}
            for i in pass_k_parameters["k"]:
                pass_k_scores[i] = {
                    "score_now": 0,
                    "num_rows": 0
                }
                pass_k_scores_tf_plan[i] = {
                    "score_now": 0,
                    "num_rows": 0
                }
            # pass_k_score_now = 0
            # num_rows = 0

            passed_once = False

            iac_eval_complexity_accuracy = {
                "1": {
                    "num_success_both": 0,
                    "num_rows": 0,
                },
                "2": {
                    "num_success_both": 0,
                    "num_rows": 0,
                },
                "3": {
                    "num_success_both": 0,
                    "num_rows": 0,
                },
                "4": {
                    "num_success_both": 0,
                    "num_rows": 0,
                },
                "5": {
                    "num_success_both": 0,
                    "num_rows": 0,
                },
                "6": {
                    "num_success_both": 0,
                    "num_rows": 0,
                }
            }

            for student, dataset_list in student_dict.items():
                # n_total = 0 # number of trials that this student has been evaluated for. Used for pass@k's n parameter
                for file1 in dataset_list:
                    # Go through each row
                    df = pd.read_csv(file1) # this is now redundant since the next line also reads the file 
                    df = fix_duplicate_columns(file1)
                    print(f"Begin calc metrics for dataset {file1}")

                    # iterate through each row
                    for index, row in df.iterrows():

                        passed_once = True
                        # find specific column
                        # print(index)
                        # while True:
                        #     x=1

                        if isinstance(row["Prompt"], float):
                            continue

                        # dataset_df, dataset_row = extract_dataset_row(row, index, df, dataset_dir, file1) # extract the corresponding row from the "data" dataset file

                        # complexity = difficulty_retrieval(dataset_row)
                        # Check all samples: Loop through columns whose names match the substring
                        num_samples = 0
                        num_correct = 0
                        num_tf_plan_correct = 0
                        for column in df.filter(like="LLM Output #"):
                            num_samples += 1
                            # print(f"Processing column: {column}")
                            sample_count = column.split("#")[1]
                            plannable = row["LLM Plannable? #{}".format(sample_count)]
                            correct = row["LLM Correct? #{}".format(sample_count)]
                            if plannable == "True" or plannable == True or plannable == "TRUE": # needed these extra conditions because fixing duplicate columns causes for example TRUE which would normally be evaluated into the boolean True, to be evaluated as a string TRUE instead..
                                num_tf_plan_correct += 1
                                if correct == "Success":
                                    num_correct += 1
                                    # iac_eval_complexity_accuracy[complexity]["num_success_both"] += 1

                            if num_samples == pass_k_parameters["n"]: # only evaluate up till n samples, even if there are more
                                break 
                        
                        if "Wizard" in file1:
                            print(num_correct)
                        # if "Wizard" in file1:
                        #     while True:
                        #         x=1

                        assert num_samples == pass_k_parameters["n"] # prereq: must have at least n samples
                        for k, attrs in pass_k_scores.items():
                            attrs["score_now"] += estimator(
                                        pass_k_parameters["n"], num_correct, k
                                    )
                            attrs["num_rows"] += 1
                        for k, attrs in pass_k_scores_tf_plan.items():
                            attrs["score_now"] += estimator(
                                        pass_k_parameters["n"], num_tf_plan_correct, k
                                    )
                            attrs["num_rows"] += 1
                    
                    print(f"Finished calc metrics for dataset {file1}")

            if not passed_once:
                # print(strat)
                # while True:
                #     x=1
                continue

            # Calculate pass@k scores                
            output_composition_dict[model][strat]["pass@k"] = {
                                                                "n": str(pass_k_parameters["n"])
                                                            }
            output_composition_dict[model][strat]["pass@k-tf-plan"] = {
                                                                "n": str(pass_k_parameters["n"])
                                                            }
            for k, attrs in pass_k_scores.items():
                output_composition_dict[model][strat]["pass@k"][k] = str(attrs["score_now"]/attrs["num_rows"])
            for k, attrs in pass_k_scores_tf_plan.items():
                output_composition_dict[model][strat]["pass@k-tf-plan"][k] = str(attrs["score_now"]/attrs["num_rows"])
                
            # Copy over the IAC eval accuracy from the ablation output composition dict: (for plotting purposes)
            with open(os.path.join(os.path.dirname(output_composition_dict_filepath), "../ablation", "output_composition_dict.json"), 'r') as fp:
                ablation_output_composition_dict = json.load(fp)
                output_composition_dict[model][strat]["iac_eval_accuracy"] = str(ablation_output_composition_dict[model][strat]["iac_eval_accuracy"])

            with open(output_composition_dict_filepath, 'w') as fp: # incremental updates
                json.dump(output_composition_dict, fp)

    return output_composition_dict

def extract_dataset_row(eval_row, eval_index, eval_df, dataset_dir, eval_filename, strats=["multi-turn", "RAG", "COT", "FSP"]):
    """
    Example eval_filename: '/home/ubuntu/autoiac-tasks/evaluation/misc/ablation/../../../results-for-iac-eval (backup)/multi-turn/codellama-34b/george/evaluation-dataset-for-plain-dataset-george-existing-multi-turn.csv'
    """

    # Extract corresponding "data" dataset file name:
    file_stem = Path(eval_filename).stem # https://stackoverflow.com/questions/678236/how-do-i-get-the-filename-without-the-extension-from-a-path-in-python
    file_stem = file_stem.split("evaluation-dataset-for-")[1]
    filename = ""
    for i in strats:
        if i in file_stem:
            filename = file_stem.split("-{}".format(i))[0] + ".csv"
    if filename == "":
        filename = file_stem + ".csv"

    filename = "difficulty-included-" + filename
    
    containing_folder = os.path.basename(os.path.dirname(eval_filename)) # e.g., weijun
    dataset_filename = os.path.join(dataset_dir, containing_folder, filename)

    # Access dataset file and extract required row
    df = pd.read_csv(dataset_filename, header=None)

    # set the third row as the header
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    # # reset the index of the DataFrame
    df.reset_index(drop=True, inplace=True)

    # print(df.at[eval_index, "Prompt"])
    # while True:
    #     x=1

    # print(df)

    # assert df.at[eval_index, "Prompt"] == eval_row["Prompt"] # add this back in as needed. Removed for now since I might have updated some dataset rows after some eval was performed on the old version of the dataset..

    return df, df.iloc[eval_index]

def difficulty_retrieval(row, difficulty_header="Calculated Complexity"):
    prompt = row["Prompt"]
    if isinstance(prompt, float):
        if math.isnan(prompt):
            return
    reference = row["Desired output"]
    policy = row["Rego intent"]

    print("Prompt:", prompt)

    return row[difficulty_header]

def difficulty_calculation(row, df, index, new_difficulty_headers=["Calculated Complexity"]):
    prompt = row["Prompt"]
    if isinstance(prompt, float):
        if math.isnan(prompt):
            return
    reference = row["Desired output"]
    policy = row["Rego intent"]

    print("Prompt:", prompt)

    complexity_level = calc_complexity(reference, prompt)
    # ambiguity_level = calc_ambiguity(policy, prompt)
    # print(complexity_level, ambiguity_level)
    # print("Complexity level: ", complexity_level)
    # print("Ambiguity level: ", ambiguity_level)

    for header in new_difficulty_headers:
        if "Complexity" in header:
            df.at[index, header] = complexity_level
        # elif "Ambiguity" in header:
        #     df.at[index, header] = ambiguity_level

    return complexity_level

def calc_complexity(reference, prompt):
    """
        Calculate complexity based on LOC, num resources, and number of interconnections. 
    """
    # Calculate LOC:
    LOC = sum(not line.isspace() for line in StringIO(reference))
    # print("LOC", LOC)

    # Calculate number of resources:
    # Count the number of occurences of the word "resource" in the reference string
    num_resources = reference.count("resource")
    # print("num_resources", num_resources)

    # Calculate number of interconnections:
    write_to_terraform(reference)
    # run terraform plan and capture the output and errors
    plan_file = "plan.out"
    terraform_dir = "./misc/ablation/terraform_config"
    output_json_file = "plan.json"
    output_json_filepath = os.path.join(terraform_dir, output_json_file)
    generate_terraform_plan_json(prompt, terraform_dir, plan_file, output_json_file)
    # read the json file
    # print("about to read json..")
    with open(output_json_filepath, "r") as json_file:
        data = json.load(json_file)
        # extract the number of interconnections from the json file
        config_graph = data["configuration"]["root_module"]["resources"]
        references_list = list(findkeys(config_graph, "references")) # just a list with the word "references" repeated
        # print(references_list)
        num_interconnections = len(references_list)
        # print("num_interconnections", num_interconnections)
    # Determine complexity:
    return get_complexity_level(LOC, num_resources, num_interconnections)

def get_complexity_level(LOC, num_resources, num_interconnections):
    if LOC < 10 and num_resources < 2 and num_interconnections < 2:
        return "1"
    if LOC < 20 and num_resources < 4 and num_interconnections < 4:
        return "2"
    if LOC < 40 and num_resources < 6 and num_interconnections < 6:
        return "3"
    if LOC < 60 and num_resources < 8 and num_interconnections < 8:
        return "4"
    if LOC < 80 and num_resources < 10 and num_interconnections < 10:
        return "5"
    if LOC >= 80 or num_resources >= 10 or num_interconnections >= 10:
        return "6"

def findkeys(node, kv):
    # Modified from: https://stackoverflow.com/a/19871956/13336187
    if isinstance(node, list):
        for i in node:
            for x in findkeys(i, kv):
               yield x
    elif isinstance(node, dict):
        if kv in node:
            yield kv
        for j in node.values():
            for x in findkeys(j, kv):
                yield x

# print(list(findkeys(d, 'id')))

def write_to_terraform(result, terraform_dir="./misc/ablation/terraform_config"):
    # define the path to the main.tf file
    terraform_file_path = terraform_dir + "/metric-measurement-main.tf"
    # open the file in write mode ('w') and write the result to it
    # print("CWD", os.getcwd())
    # print("CWD: {}".format(os.getcwd()))
    with open(terraform_file_path, "w+", encoding="utf-8", errors="ignore") as file:
        file.write(result)
    # print(f"Updated main.tf at {terraform_file_path}")
    # print(f"Updated main.tf at {terraform_file_path}")
    
def generate_terraform_plan_json(prompt, terraform_dir="./misc/ablation/terraform_config", plan_file="plan.out", output_json_file="plan.json"):
    cwd = os.getcwd()
    # change to the Terraform directory
    os.chdir(terraform_dir)
    # run init before plan
    init_result = subprocess.run(["terraform", "init"], capture_output=True, text=True)

    # run 'terraform plan'
    # result = subprocess.run(["terraform", "plan"], capture_output=True, text=True)

    result_returned = False
    # generate Terraform plan with the -no-color flag
    for i in range(2): # try twice
        try:
            result = subprocess.run(
                ["terraform", "plan", "-out", plan_file, "-no-color"], capture_output=True, text=True, timeout=300 # 5 minutes timeout (assume failed if timeout)
            )

            with open(
                output_json_file, "w", encoding="utf-8", errors="ignore"
            ) as json_file:
                show_result = subprocess.run(
                    ["terraform", "show", "-json", plan_file], check=True, stdout=json_file
                )

            result_returned = True
            break
        except Exception as e:
            print("Error occurred for prompt \"{}\": {}".format(prompt, e))

    # Return to parent directory
    os.chdir(cwd)

    if result_returned == False:
        return "Plan timed-out. No output", "Plan timed-out. No error", False

@click.command()
@click.option(
    "--samples",
    "-n",
    "samples",
    type=int,
    help="Number of samples per task.",
    default=20,
)
def main(samples: int):
    # Find dataset files: 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_dataset_dir = os.path.join(script_dir, "../..", "results")
    dataset_dir = os.path.join(script_dir, "../complete-dataset-measurement/complete-dataset")
    # print()
    excluded_dirs = ["llm-judge-evaluation-metric", "pass", # skip filenames containing "pass". will merge them into a regular file once done with eval..
    "multi-turn", "RAG", "COT", "FSP"] # also skip these for now since we don't have the data: just add these back in once the data is ready and we're good to go
    composition_dict = {
        "gpt4": {
            "COT": {
                "complete": [],
            },
            "FSP": {
                "complete": [],
            },
            "multi-turn": {
                "complete": [],
            },
            "RAG": {
                "complete": [],
            },
            "Standard": {
                "complete": [],
            },
        },
        "gpt3.5": {
            "COT": {
                "complete": [],
            },
            "FSP": {
                "complete": [],
            },
            "multi-turn": {
                "complete": [],
            },
            "RAG": {
                "complete": [],
            },
            "Standard": {
                "complete": [],
            },
        },
        "gemini-1.0-pro": {
            "COT": {
                "complete": [],
            },
            "FSP": {
                "complete": [],
            },
            "multi-turn": {
                "complete": [],
            },
            "RAG": {
                "complete": [],
            },
            "Standard": {
                "complete": [],
            },
        },
        "codellama-34b": {
            "COT": {
                "complete": [],
            },
            "FSP": {
                "complete": [],
            },
            "multi-turn": {
                "complete": [],
            },
            "RAG": {
                "complete": [],
            },
            "Standard": {
                "complete": [],
            },
        },
        "codellama-13b": {
            "COT": {
                "complete": [],
            },
            "FSP": {
                "complete": [],
            },
            "multi-turn": {
                "complete": [],
            },
            "RAG": {
                "complete": [],
            },
            "Standard": {
                "complete": [],
            },
        },
        "codellama-7b": {
            "COT": {
                "complete": [],
            },
            "FSP": {
                "complete": [],
            },
            "multi-turn": {
                "complete": [],
            },
            "RAG": {
                "complete": [],
            },
            "Standard": {
                "complete": [],
            },
        },
        "Magicoder_S_CL_7B": {
            "COT": {
                "complete": [],
            },
            "FSP": {
                "complete": [],
            },
            "multi-turn": {
                "complete": [],
            },
            "RAG": {
                "complete": [],
            },
            "Standard": {
                "complete": [],
            },
        },
        "Wizardcoder33b": {
            "COT": {
                "complete": [],
            },
            "FSP": {
                "complete": [],
            },
            "multi-turn": {
                "complete": [],
            },
            "RAG": {
                "complete": [],
            },
            "Standard": {
                "complete": [],
            },
        }
    }
    eval_dataset_files, composition_dict = list_all_subdirectories(composition_dict, eval_dataset_dir, excluded_dirs)
    eval_dataset_files = [f for f in eval_dataset_files if "test" not in f]
    # print(eval_dataset_files)

    with open(os.path.join(script_dir, 'input_composition_dict.json'), 'w') as fp:
        json.dump(composition_dict, fp)

    # while True:
    #     x=1

    print(composition_dict)
    # metrics_included_dataset_dir = script_dir + "/ablation-dataset"
    # create_new_dirs_for_metric(metrics_included_dataset_dir, dataset_files)
    # dst_filenames = copy_csv_to_metric(dataset_files, difficulty_included_dataset_dir)
    # # print(dst_filenames)

    pass_k_parameters = {
        "n": samples,
        "k": [x for x in range(1, samples+1)]
    }

    output_composition_dict_filepath = os.path.join(script_dir, 'pass-k-output_composition_dict.json')

    # Calculate regular metrics
    output_composition_dict = metric_calculation_loop(composition_dict, dataset_dir, output_composition_dict_filepath, pass_k_parameters)
    print("Output composition dict: ", pretty_json(output_composition_dict))
    # print("complexity_distribution: ", pretty_json(complexity_distribution))

    with open(output_composition_dict_filepath, 'w') as fp:
        json.dump(output_composition_dict, fp)

    # with open(os.path.join(script_dir, 'complexity_distribution.json'), 'w') as fp:
    #     json.dump(complexity_distribution, fp)

    # Calculate pass@k metric:
    # print(metric_calculation_loop(composition_dict, pass_k_parameters))

    # Display number of rows for each complexity level:

if __name__ == "__main__":
    main()