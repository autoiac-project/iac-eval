# Assumption: single file for each evaluated dataset. (e.g., no -pass5 and -pass1 in the same folder.) 
import csv
import pandas as pd
import logging
import os
from pathlib import Path
from openai import AzureOpenAI
import pandas as pd
import json
import math
import subprocess
import copy
from io import StringIO
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import eval
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
    return json.dumps(obj, sort_keys=True, indent=4, default=str)

def estimator(n: int, c: int, k: int) -> float:
    """
    Calculates 1 - comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def list_all_subdirectories(composition_dict, data_dir, excluded_dirs, strats=["multi-turn", "RAG", "COT", "FSP"]):
    # onlyfiles = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) if dirpath not in excluded_dirs in os.walk(data_dir) for f in filenames] 
    onlyfiles = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        # print(dirpath)
        skip_dirname = False
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

def metric_calculation_loop(composition_dict, dataset_dir, output_composition_dict_filepath, pass_k_parameters=None):

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
                "tf_plan_only_accuracy": 0.4 
                "iac_eval_complexity_accuracy" : {
                    "1": 0,
                    "2": 0,
                    "3": 0,
                    "4": 0,
                    "5": 0,
                    "6": 0,
                },
                "pass@k" : {
                    "n": 5 # just for record
                    "1": 0.5,
                    "2": 0.6,
                    "5": 0.7,
                }
                "bleu_accuracy": 0.8,
                "exact_match_accuracy": 0.5
                # will add this in later once patch up llm-judge-eval.py:
                # "llm-judge": {
                #     "accuracy": 0.8,
                #     "precision": 0.5,
                #     "recall": 0.3
                # }
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

    # complexity_distribution = {
    #     "level-1": 0,
    #     "level-2": 0,
    #     "level-3": 0,
    #     "level-4": 0,
    #     "level-5": 0,
    #     "level-6": 0,
    # }

    for model, model_dict in composition_dict.items():
            for strat, student_dict in model_dict.items():
                if strat != "Standard":
                    continue   
                num_success_both = 0 # pass iac_eval pipeline correctly
                num_rows = 0
                num_plan_success = 0
                bleu_score_total = 0
                bleu_score_total_correct = 0
                bleu_score_total_incorrect = 0
                exact_match_count = 0
                false_positive_count = 0
                false_negative_count = 0

                codebert_metrics_total = {
                    "precision": 0,
                    "recall": 0,
                    "f1": 0,
                    "f3": 0,
                    "f1-correct": 0,
                    "f1-incorrect": 0,
                }

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

                if strat == "Standard" and pass_k_parameters:
                    pass_at_k_scores = {}
                    for i in pass_k_parameters:
                        pass_at_k_scores[i] = {
                            "score_now": 0,
                            "num_rows": 0
                        }

                for student, dataset_list in student_dict.items():
                    # n_total = 0 # number of trials that this student has been evaluated for. Used for pass@k's n parameter
                    for file1 in dataset_list:
                        # Go through each row
                        df = pd.read_csv(file1)
                        print(f"Begin calc metrics for dataset {file1}")

                        # iterate through each row
                        for index, row in df.iterrows():

                            if isinstance(row["Prompt"], float):
                                if math.isnan(row["Prompt"]):
                                    continue

                            passed_once = True
                            # find specific column
                            # print(index)
                            # while True:
                            #     x=1

                            dataset_df, dataset_row = extract_dataset_row(row, index, df, dataset_dir, file1) # extract the corresponding row from the "data" dataset file

                            complexity = difficulty_retrieval(dataset_row)

                            # Calculate pass@k scores. 
                            if strat == "Standard" and pass_k_parameters:
                                num_correct = 0
                                columns = df.columns
                                n_now = [c for c in columns if "LLM Plannable" in c]
                                assert n_now == pass_k_parameters["n"]
                                for i in range(n_now):
                                    if row["LLM Plannable? #{}".format(i)] == True:
                                        # num_plan_success += 1
                                        if row["LLM Correct? #{}".format(i)] == "Success":
                                            num_correct += 1

                                for k, attrs in pass_at_k_scores.items():
                                    attrs["score_now"] += estimator(
                                                pass_k_parameters["n"], num_correct, k
                                            )
                                    attrs["num_rows"] += 1

                            else: # this includes strat == "Standard" but pass_k_parameters is not provided
                                # print(row)
                                # if row["LLM Plannable? #0"] != False: 
                                #     while True:
                                #         x=1
                                reference = row["Reference output"]
                                try:
                                    answer, candidate = eval.separate_answer_and_code(row["LLM Output #0"], DELIMITERS)
                                except Exception as e:
                                    s = str(e)
                                    print(s)
                                    if "'float' object has no attribute 'split'" in s:
                                        candidate = ""
                                # print(complexity)
                                iac_eval_complexity_accuracy[complexity]["num_rows"] += 1
                                num_rows += 1
                                bleu_score_total += metrics.bleu_score(reference, candidate)
                                exact_match_count += 1 if metrics.exact_match(reference, candidate) else 0
                                
                                codebert_metrics = metrics.get_code_bert_score(reference, candidate, row["Prompt"])
                                codebert_metrics_total["precision"] += codebert_metrics["precision"]
                                codebert_metrics_total["recall"] += codebert_metrics["recall"]
                                codebert_metrics_total["f1"] += codebert_metrics["f1"]
                                codebert_metrics_total["f3"] += codebert_metrics["f3"]
                                is_correct = False
                                if row["LLM Plannable? #0"] == True: # we use #0 even if it is possible that a mistake was made somewhere and we evaluated e.g., FSP twice. 
                                    num_plan_success += 1
                                    print("num_plan_success: ", num_plan_success)
                                    if row["LLM Correct? #0"] == "Success":
                                            num_success_both += 1
                                            is_correct = True
                                            # while True:
                                            #     x=1
                                            print("num_success_both: ", num_success_both)
                                            iac_eval_complexity_accuracy[complexity]["num_success_both"] += 1

                                if is_correct:
                                    bleu_score_total_correct += metrics.bleu_score(reference, candidate)
                                    codebert_metrics_total["f1-correct"] += codebert_metrics["f1"]
                                else:
                                    bleu_score_total_incorrect += metrics.bleu_score(reference, candidate)   
                                    codebert_metrics_total["f1-incorrect"] += codebert_metrics["f1"]        

                            # complexity_distribution = update_complexity_distribution(complexity, complexity_distribution)
                            
                            # print(f"Complexity distribution now: {complexity_distribution}")
                        
                        # df.to_csv(file1, index=False, encoding="utf-8")
                        print(f"Finished calc metrics for dataset {file1}")

                if not passed_once:
                    continue

                # Calculate pass@k scores                
                if strat == "Standard" and pass_k_parameters: 
                    output_composition_dict[model][strat]["pass@k"] = {
                                                                        "n": pass_k_parameters["n"]
                                                                    }
                    for k, attrs in pass_at_k_scores.items():
                        output_composition_dict[model][strat]["pass@k"][k] = attrs["score_now"]/attrs["num_rows"]
                    
                else: # Calculate metric scores
                    output_composition_dict[model][strat]["iac_eval_accuracy"] = num_success_both / num_rows
                    output_composition_dict[model][strat]["tf_plan_only_accuracy"] = num_plan_success / num_rows
                    output_composition_dict[model][strat]["iac_eval_complexity_accuracy"] = {}
                    for level, attrs in iac_eval_complexity_accuracy.items():
                        # print("iac_eval_complexity_accuracy: ", iac_eval_complexity_accuracy)
                        # print("level: ", level)
                        # print("attrs: ", attrs)
                        # print("output_composition_dict[model][strat][iac_eval_complexity_accuracy] :", output_composition_dict[model][strat]["iac_eval_complexity_accuracy"])
                        # print(output_composition_dict[model][strat]["iac_eval_complexity_accuracy"][level])
                        # print(attrs["num_success_both"])
                        # print(attrs["num_rows"])
                        if attrs["num_rows"] == 0:
                            output_composition_dict[model][strat]["iac_eval_complexity_accuracy"][level] = 0
                        else:
                            output_composition_dict[model][strat]["iac_eval_complexity_accuracy"][level] = attrs["num_success_both"]/attrs["num_rows"]
                    output_composition_dict[model][strat]["bleu_accuracy"] = bleu_score_total / num_rows
                    if num_success_both == 0:
                        output_composition_dict[model][strat]["bleu_accuracy_correct"] = 0
                    else:
                        output_composition_dict[model][strat]["bleu_accuracy_correct"] = bleu_score_total_correct / num_success_both
                    output_composition_dict[model][strat]["bleu_accuracy_incorrect"] = bleu_score_total_incorrect / (num_rows - num_success_both)
                    output_composition_dict[model][strat]["exact_match_accuracy"] = exact_match_count / num_rows 
                    output_composition_dict[model][strat]["codebert_metrics"] = {
                        "precision": codebert_metrics_total["precision"] / num_rows,
                        "recall": codebert_metrics_total["recall"] / num_rows,
                        "f1": codebert_metrics_total["f1"] / num_rows,
                        "f3": codebert_metrics_total["f3"] / num_rows,
                        "f1-incorrect": codebert_metrics_total["f1-incorrect"] / (num_rows - num_success_both)
                    }
                    if num_success_both == 0:
                        output_composition_dict[model][strat]["codebert_metrics"]["f1-correct"] = 0
                    else:
                        output_composition_dict[model][strat]["codebert_metrics"]["f1-correct"] = codebert_metrics_total["f1-correct"] / num_success_both

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

    # filename = "completed-" + filename
    
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

def difficulty_retrieval(row, difficulty_header="Difficulty"):
    prompt = row["Prompt"]
    reference = row["Reference output"]
    policy = row["Rego intent"]

    print("Prompt:", prompt)

    return str(int(float(row[difficulty_header])))

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
    dataset_dir = os.path.join(script_dir, "../../../data")
    # print()
    excluded_dirs = ["llm-judge-evaluation-metric", "pass"] # skip filenames containing "pass". will merge them into a regular file once done with eval..
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

    print(composition_dict)

    output_composition_dict_filepath = os.path.join(script_dir, 'output_composition_dict.json')

    # Calculate regular metrics
    output_composition_dict = metric_calculation_loop(composition_dict, dataset_dir, output_composition_dict_filepath)
    print("Output composition dict: ", pretty_json(output_composition_dict))

    with open(output_composition_dict_filepath, 'w') as fp:
        json.dump(output_composition_dict, fp)

if __name__ == "__main__":
    main()