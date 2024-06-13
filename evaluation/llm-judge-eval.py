# Conceptual Reference: https://arxiv.org/pdf/2306.05685
import os
from pathlib import Path
from openai import OpenAI
import sys
import pandas as pd
import logging
from os import listdir
from os.path import isfile, join
import numpy as np
import math
import json
import metrics
import eval
import models

NUM_SAMPLES_PER_TASK = 1 # number of times we want to run the LLM judge for each eval output (i.e., LLM output # X) from the dataset

LOG_FILE = "logs/metrics_eval.log"
DELIMITERS = ["```hcl", "```json", "```HCL", "```Terraform", "```terraform", "```"] # ``` needs to be in the end, as it is a final "else" case

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

logger = logging.getLogger("iac-eval-metric")
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

def setup_gpt_client():
    global gpt_client
    global embeddings_model
    if "OPENAI_API_KEY" not in os.environ:
        api_key = input("Enter OpenAI API key:")
        os.environ["OPENAI_API_KEY"] = api_key

    api_key = os.environ["OPENAI_API_KEY"]

    gpt_client= OpenAI(api_key=api_key)

# split the code for results
def extract_rating_from_answer(text):
    rating_str = text.splitlines()[-1]
    if "Correct" or "Incorrect" in rating_str: 
        return rating_str
    else:
        logger.error("Error: Answer contains no code, skipping eval_pipeline.")    
        return rating_str

# Take our existing dataset, iterate through each row, add a new column with the LLM generated response, and a new column for the extracted verdict, and save the new dataset to a new file with a filename that is prefixed with the llm model-name. 
def list_all_subdirectories(data_dir):
    onlyfiles = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(data_dir) for f in filenames] 
    onlyfilescsv = [f for f in onlyfiles if f.endswith(".csv")]
    return onlyfilescsv

def create_new_dirs_for_metric(base_eval_files, new_base_metric_dir="llm-judge-evaluation-metric"):
    for file in base_eval_files:
        # create the directory if it does not exist
        new_dir_path = os.path.join(os.path.dirname(file), new_base_metric_dir)
        os.makedirs(new_dir_path, exist_ok=True)

def determine_eval_samples(df_columns):
    """
        Determine the number of samples per task
    """
    num_samples = 0
    for col in df_columns:
        if "LLM Correct?" in col:
            num_samples += 1
    return num_samples

def copy_csv_to_metric(base_eval_files, new_base_metric_dir, llm="gpt4"):
    """
        Example:
            base_eval_files: ["../results-for-iac-eval (backup)/george-existing/evaluation-dataset-george-gpt3.5.csv"]
            llm_base_eval_dir: "../results-for-iac-eval (backup)/george-existing/llm-judge-evaluation-metric"
    """
    dst_filenames = []
    dst_filenames_skipped = []
    llm_single_filename_prefix = "{}-single".format(llm)
    llm_reference_filename_prefix = "{}-reference".format(llm)
    for file in base_eval_files: 
        file_stem = Path(file).stem # https://stackoverflow.com/questions/678236/how-do-i-get-the-filename-without-the-extension-from-a-path-in-python
        filename = llm_single_filename_prefix + "-" + file_stem + ".csv"
        dest_file_path = os.path.join(os.path.dirname(file), new_base_metric_dir, filename)
        # print(dest_file_path)
        # Check if llm-judge file is already evaluated: skip if true
        if os.path.isfile(dest_file_path) and os.stat(dest_file_path).st_size != 0: # if file is not empty
            dst_df = pd.read_csv(dest_file_path, header=None)
            new_header = dst_df.iloc[0]
            dst_df = dst_df[1:]
            dst_df.columns = new_header
            dst_df.reset_index(drop=True, inplace=True)
            # print(dst_df['gpt4 Judge Output #0 for LLM output #0'].iloc[0])
            if not pd.isnull(dst_df['gpt4 Judge Output #0 for LLM output #0'].iloc[0]):
                # print("hi: ", dest_file_path)
                # while True:
                #     x=1
                dst_filenames_skipped.append(dest_file_path)
                continue
        # print("no hi", dest_file_path)
        # while True:
        #     x=1
        dst_filenames.append(dest_file_path)
        df = pd.read_csv(file, header=None)
        new_header = df.iloc[0]
        df = df[1:]
        df.columns = new_header
        NUM_EVAL_SAMPLES = determine_eval_samples(df.columns)
        # reset the index of the DataFrame
        df.reset_index(drop=True, inplace=True)
        # add new columns only to df
        for i in range(NUM_SAMPLES_PER_TASK):
            for col_base in [
                "{} Judge Output #{} for LLM output #{}".format(llm, str(i), str(0)),
                "{} Judge verdict #{} for LLM output #{}".format(llm, str(i), str(0)),
            ]:
                df[col_base] = ""

        df.to_csv(dest_file_path, index=False, encoding="utf-8")
        # print(f"Copied and modified CSV to: {dest_file_path}")
    return dst_filenames, dst_filenames_skipped

def llm_judge_eval_loop(dst_filenames, llm="", METRIC="llm-single"):
    for file in dst_filenames:
        df = pd.read_csv(file)
        logger.info(f"Begin extracting metric from dataset {file}")
        # while True:
        #     x=1
        # iterate through each row
        for index, row in df.iterrows():
        # iterate every row
        # find specific column
            llm_judge_eval(row, df, index, llm, METRIC)

        df.to_csv(file, index=False, encoding="utf-8")

        logger.info(f"Finished extracting metric from dataset {file}")

def get_prompt_from_metric(prompt, candidate, reference, llm="", METRIC="llm-single"):
    if METRIC == "llm-single":
        prompt = metrics.llm_as_judge_single(prompt, candidate)
    else:
        assert 1 == 0, "Invalid METRIC"
    return prompt

def llm_judge_eval(row, df, index, model="", METRIC="llm-single"):
    """
        Note: Multi-turn implies 2 turns only
    """
    prompt = row["Prompt"]
    if isinstance(prompt, float):
        if math.isnan(prompt):
            return
    reference = row["Reference output"]
    preprompt = ""
    # prompt = prompt_enhancements(prompt, PROMPT_ENHANCEMENT_STRAT)
    # while True:
    #     x=1
    policy_file = row["Rego intent"]
    # num_correct = 0
    # logger.info(f"Begin extracting metric {METRIC} from dataset:")
    NUM_EVAL_SAMPLES = determine_eval_samples(df.columns)
    for i in range(NUM_SAMPLES_PER_TASK):
        eval_answer = row["LLM Output #{}".format(str(0))]
        if isinstance(eval_answer, float) and math.isnan(eval_answer):
            text = "LLM output in dataset is Nan"
            rating_str = "No output"
        else:
            logger.info(f"Model raw output (already in the dataset itself, for when evaluating a model on the dataset): {eval_answer}")
            answer, candidate = eval.separate_answer_and_code(eval_answer, DELIMITERS)
            logger.info(f"Candidate config: {candidate}")
            
            prompt = get_prompt_from_metric(prompt, candidate=candidate, reference=reference, llm=model, METRIC=METRIC)

            logger.info(f"Sample {i} for metric {METRIC}")
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

            rating_str = extract_rating_from_answer(text)
            logger.info("Answer is: {}".format(text))
            logger.info("Rating is: {}".format(rating_str))

        df.at[index, "{} Judge Output #{} for LLM output #{}".format(model, str(i), str(0))] = text
        df.at[index, "{} Judge verdict #{} for LLM output #{}".format(model, str(i), str(0))] = rating_str

    # return text

# main:
def main(): # E.g.,: python3 llm-judge-eval.py gpt4 llm-single
    llm = sys.argv[1] # options: gpt4
    metric = sys.argv[2] # options: "llm-single"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_data_dir = os.path.join(script_dir, "results")
    base_eval_files = list_all_subdirectories(eval_data_dir)
    base_eval_files = [f for f in base_eval_files if "-single" not in f and "-reference" not in f]
    print(base_eval_files)

    base_eval_files = [f for f in base_eval_files if "COT" not in f and "FSP" not in f and "RAG" not in f and "multi-turn" not in f] # temporarily to save time...
    print(base_eval_files)

    # base_eval_files = ['/home/ubuntu/autoiac-tasks/evaluation/../results-for-iac-eval (backup)/multi-turn/gpt4/lei-existing/evaluation-dataset-multi-turn-lei-gpt4-test-1.csv'] # just for debugging purposes

    create_new_dirs_for_metric(base_eval_files, new_base_metric_dir="llm-judge-evaluation-metric")
    dst_filenames, dst_filenames_skipped = copy_csv_to_metric(base_eval_files, new_base_metric_dir="llm-judge-evaluation-metric", llm="gpt4")

    print("Filenames to evaluate: ", dst_filenames)
    print("------------------------------")
    print("Filenames skipped (since they have already been evaluated): ", dst_filenames_skipped)
    # Write dst_filenames to a file:
    # with open("delete-soon-archived/dst_filenames.txt", "w") as f:
    #     for item in dst_filenames:
    #         f.write("%s\n" % item)
    # while True:
    #     x=1
    llm_judge_eval_loop(dst_filenames, llm=llm, METRIC=metric) 
    # At the end, get a correctness percentage across all rows in the dataset (across all files). 

if __name__ == "__main__":
    setup_gpt_client()
    main()