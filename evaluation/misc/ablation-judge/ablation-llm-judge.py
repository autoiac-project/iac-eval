# Assumption: single file for each evaluated dataset. (e.g., no -pass5 and -pass1 in the same folder.) 
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import eval
import metrics
import math

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
    return json.dumps(obj, sort_keys=True, indent=4, default=str)

def estimator(n: int, c: int, k: int) -> float:
    """
    Calculates 1 - comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def list_all_subdirectories(composition_dict, data_dir, excluded_dirs, strats=["multi-turn", "RAG", "COT", "FSP"], mandatory_substring="llm-judge-evaluation-metric"):
    onlyfiles = []
    for dirpath, dirnames, filenames in os.walk(data_dir):

        skip_dirname = False

        # Skip eval directories that were used to evaluate custom strats (e.g., RAG...), i.e., only keep standard 
        for dir1 in strats:
            if dir1 in dirpath:
                skip_dirname = True
        # Keep only LLM-judge output directories
        if not skip_dirname:
            if mandatory_substring not in dirpath:
                skip_dirname = True

        if not skip_dirname:
            for f in filenames: 
                skip_file = False
                for exc in excluded_dirs:
                    if exc in f:
                        skip_file = True
                if not skip_file:
                    onlyfiles.append(os.path.join(dirpath, f))
    
    onlyfilescsv = [f for f in onlyfiles if f.endswith(".csv")]

    # print(onlyfilescsv)
    # while True:
    #     x=1

    # Append to composition dict: 
    for file1 in onlyfilescsv:
        dirname = os.path.dirname(file1)
        # print(os.path.dirname(file1))
        # while True:
        #     x=1
        included = False
        is_standard = True
        for i in strats:
            if i in dirname:
                is_standard = False
        
        for model, model_dict in composition_dict.items():
            for strat, student_dict in model_dict.items():
                if is_standard:
                    if strat != "Standard":
                        continue    
                for student, dataset_list in student_dict.items():
                    is_file = False
                    if is_standard:
                        if student in dirname and model in dirname:
                            is_file = True
                    else:
                        if student in dirname and strat in dirname and model in dirname:
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

def metric_calculation_loop(composition_dict, dataset_dir, output_composition_dict_filepath):

    """ 
    Output format example (added in place to composition_dict): {
        "gpt4": {
            "Standard": {
                ..., # whatever that is already in composition_dict
                "llm-judge": {
                    "accuracy": 0.8,
                    "precision": 0.5,
                    "recall": 0.3
                },
                "iac_eval_accuracy": XYZ, # copied from ablation/output_composition_dict.json
            }, 
            "COT": <current implementation not supported>
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
            num_success = 0 # row considered correct by LLM-judge pipeline correctly
            num_rows = 0

            true_positive_count = 0
            false_positive_count = 0
            false_negative_count = 0

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
                    df = pd.read_csv(file1)
                    print(f"Begin calc metrics for dataset {file1}")

                    # iterate through each row
                    for index, row in df.iterrows():
                        
                        # Skip empty rows
                        if isinstance(row["Prompt"], float):
                            if math.isnan(row["Prompt"]):
                                continue

                        print("Current Prompt: ", row["Prompt"])

                        passed_once = True
                        num_rows += 1

                        # TODO: May add this back in later if need it for appendix evaluation 
                        # dataset_df, dataset_row = extract_dataset_row(row, index, df, dataset_dir, file1) # extract the corresponding row from the "data" dataset file
                        # complexity = difficulty_retrieval(dataset_row) 

                        iac_eval_success = False

                        if row["LLM Plannable? #0"] == True: # we use #0 even if it is possible that a mistake was made somewhere and we evaluated e.g., FSP twice. 
                            if row["LLM Correct? #0"] == "Success":
                                iac_eval_success = True
                                # iac_eval_complexity_accuracy[complexity]["num_success_both"] += 1

                        edge_case = False
                        if isinstance(row["gpt4 Judge verdict #0 for LLM output #0"], float):
                            edge_case = True
                        else:
                            if "Incorrect" in row["gpt4 Judge verdict #0 for LLM output #0"]:
                                if iac_eval_success:
                                    false_negative_count += 1
                            elif "Correct" in row["gpt4 Judge verdict #0 for LLM output #0"]:
                                num_success += 1
                                if not iac_eval_success:
                                    false_positive_count += 1
                                else:
                                    true_positive_count += 1
                            else:
                                edge_case = True

                        if edge_case:
                            # in the rare edge case that this happens, we retrieve our original results and try and extract a line containing: "Rating: X":
                            found_rating = False
                            print(row["Prompt"])
                            print(row["gpt4 Judge Output #0 for LLM output #0"])
                            if "Rating: " in row["gpt4 Judge Output #0 for LLM output #0"]:
                                judgement_str = row["gpt4 Judge Output #0 for LLM output #0"].split("\n")
                                for line in judgement_str:
                                    if "Rating: Incorrect" in line or "Rating: Correct" in line:
                                        if "Incorrect" in line:
                                            if iac_eval_success:
                                                false_negative_count += 1
                                        elif "Correct" in line:
                                            num_success += 1
                                            if not iac_eval_success:
                                                false_positive_count += 1
                                            else:
                                                true_positive_count += 1
                                        found_rating = True
                                        break
                            if not found_rating:
                                # log it but continue. Currently, we observe that GPT4 rarely does not follow our instruction to output the line "Rating: Incorrect" or "Rating: Correct".
                                print("WARNING: Unexpected value in gpt4 Judge verdict #0 for LLM output #0: ", row["gpt4 Judge verdict #0 for LLM output #0"])
                            # assert False

                    print(f"Finished calc metrics for dataset {file1}")

            if not passed_once:
                continue
            
            # Calculate metrics
            if true_positive_count + false_positive_count == 0:
                precision = 0
            else:
                precision = true_positive_count / (true_positive_count + false_positive_count)

            if true_positive_count + false_negative_count == 0:
                recall = 0
            else:
                recall = true_positive_count / (true_positive_count + false_negative_count)

            output_composition_dict[model][strat]["llm-judge"] = {
                "accuracy": num_success / num_rows, # in terms of the perspective of the LLM-judge, what it deems to be correct. Not the same as calculating confusion matrix accuracy
                "precision": precision,
                "recall": recall
            }

            # Copy over the IAC eval accuracy from the ablation output composition dict:
            with open(os.path.join(os.path.dirname(output_composition_dict_filepath), "../ablation", "output_composition_dict.json"), 'r') as fp:
                ablation_output_composition_dict = json.load(fp)
                output_composition_dict[model][strat]["iac_eval_accuracy"] = ablation_output_composition_dict[model][strat]["iac_eval_accuracy"]

            # for level, attrs in iac_eval_complexity_accuracy.items():
            #     output_composition_dict[model][strat]["iac_eval_complexity_accuracy"][level] = attrs["num_success_both"]/attrs["num_rows"]

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

    filename = "completed-" + filename
    
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

def difficulty_retrieval(row, difficulty_header="Complexity"):
    prompt = row["Prompt"]
    if isinstance(prompt, float):
        if math.isnan(prompt):
            return
    reference = row["Reference output"]
    policy = row["Rego intent"]

    print("Prompt:", prompt)

    return row[difficulty_header]

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


def main():
    # Find dataset files: 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_dataset_dir = os.path.join(script_dir, "../..", "results")
    dataset_dir = os.path.join(script_dir, "../complete-dataset-measurement/complete-dataset")
    # print()
    excluded_dirs = ["pass"] # skip filenames containing "pass". will merge them into a regular file once done with eval..
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

    with open(os.path.join(script_dir, 'input_composition_dict.json'), 'w') as fp:
        json.dump(composition_dict, fp)
    print(composition_dict)
    output_composition_dict_filepath = os.path.join(script_dir, 'llm_judge_output_composition_dict.json')

    # Calculate regular metrics
    output_composition_dict = metric_calculation_loop(composition_dict, dataset_dir, output_composition_dict_filepath)
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