# Complexity: LOC, num resources, and number of interconnections. 
# Ambiguity: LLM-judge
import os
from pathlib import Path
import pandas as pd
import json
import math
import statistics
import time
import subprocess
import numpy
from io import StringIO
import re
from collections import defaultdict
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "evaluation")))
import models
import eval

def extract_filename(input_string):
    # Define the regular expression pattern to match filenames within quotes that end with .html or .js
    pattern = r'"([^"]+\.(html|js|txt|csv|zip|py|pub|yml|yaml))"'
    
    # Search for the pattern in the input string
    match = re.search(pattern, input_string)
    
    # Return the matched filename without quotes or None if no match is found
    return match.group(1) if match else None

def list_all_subdirectories(data_dir):
    onlyfiles = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(data_dir) for f in filenames] 
    onlyfilescsv = [f for f in onlyfiles if f.endswith(".csv")]
    return onlyfilescsv

def create_new_dirs_for_metric(new_base_difficulty_dir, base_dataset_files):
    os.makedirs(new_base_difficulty_dir, exist_ok=True)
    for file in base_dataset_files:
        containing_folder = os.path.basename(os.path.dirname(file)) # e.g., weijun
        filename = os.path.basename(file) # e.g., plain-dataset.csv
        # create the directory if it does not exist
        new_dir_path = os.path.join(new_base_difficulty_dir, containing_folder)
        # print(new_dir_path)
        os.makedirs(new_dir_path, exist_ok=True)

def copy_csv_to_metric(base_files, new_base_difficulty_dir):
    """
        Example:
            base_files: ["../data/george/evaluation-dataset-george-gpt3.5.csv"]
    """
    dst_filenames = []
    for file in base_files: 
        file_stem = Path(file).stem # https://stackoverflow.com/questions/678236/how-do-i-get-the-filename-without-the-extension-from-a-path-in-python
        containing_folder = os.path.basename(os.path.dirname(file)) # e.g., weijun
        filename =  "completed-" + file_stem + ".csv"
        dest_file_path = os.path.join(new_base_difficulty_dir, containing_folder, filename)
        # print(dest_file_path)
        dst_filenames.append(dest_file_path)
        df = pd.read_csv(file, header=None)
        new_header = df.iloc[0] 
        df = df[1:]
        df.columns = new_header
        # reset the index of the DataFrame
        df.reset_index(drop=True, inplace=True)
        # add new columns only to df
        # for header in new_difficulty_headers:
        #     df[header] = ""

        df.to_csv(dest_file_path, index=False, encoding="utf-8")
        print(f"Copied and modified CSV to: {dest_file_path}")
    return dst_filenames

def difficulty_and_resource_calculation_loop(dst_filenames, script_dir):
    actual_resource_distribution = defaultdict(int)  # e.g., "aws_linux_virtual_machine..." : 1

    complexity_distribution = {
        "level-1": 0,
        "level-2": 0,
        "level-3": 0,
        "level-4": 0,
        "level-5": 0,
        "level-6": 0,
    }
    ambiguity_distribution = {
        "level-1": 0,
        "level-2": 0,
        "level-3": 0,
        "level-4": 0,
        "level-5": 0,
        "level-6": 0,
    }

    intent_loc_list = []
    
    config_metrics = {
        "loc_list": [],  # LOC for each row's config
        "interconnections_list": [], # num of interconnections for each row's config
        "resources_list": [] # num of resources for each row's config
    }

    for file in dst_filenames:
        df = pd.read_csv(file)
        print(f"Begin calc difficulty for dataset {file}")
        # iterate through each row
        for index, row in df.iterrows():
            # iterate every row
            # find specific column
            complexity, resources, intent_loc, config_metric = difficulty_and_resources_calculation(row, df, index)
            if complexity is None:
                continue

            intent_loc_list.append(intent_loc)

            config_metrics["loc_list"].append(config_metric[0])
            config_metrics["resources_list"].append(config_metric[1])
            config_metrics["interconnections_list"].append(config_metric[2])

            for resource in resources:
                actual_resource_distribution[resource] += 1

            complexity_distribution = update_complexity_distribution(complexity, complexity_distribution)
            # ambiguity_distribution = update_ambiguity_distribution(ambiguity, ambiguity_distribution)
            
            print(f"Complexity distribution now: {complexity_distribution}")
            print("Raw resource distribution: ", actual_resource_distribution)
            # print(f"Ambiguity distribution now: {ambiguity_distribution}")

        df.to_csv(file, index=False, encoding="utf-8")

        print(f"Finished calc difficulty for dataset {file}")

    print(f"Final Complexity distribution: {complexity_distribution}")
    with open(os.path.join(script_dir, 'complexity_distribution.json'), 'w') as fp:
        json.dump(complexity_distribution, fp)
    print("Final Raw resource distribution: ", actual_resource_distribution)
    with open(os.path.join(script_dir, 'actual_resource_distribution.json'), 'w') as fp:
        json.dump(actual_resource_distribution, fp)
    get_statistics(intent_loc_list, "Intent LOC")
    get_statistics(config_metrics["loc_list"], "Config LOC")
    get_statistics(config_metrics["resources_list"], "Config resource count")
    get_statistics(config_metrics["interconnections_list"], "Config interconnection count")
    refined_resource_distribution = get_refined_resource_distribution(actual_resource_distribution)
    print("Final refined resource distribution: ", refined_resource_distribution)
    with open(os.path.join(script_dir, 'refined_resource_distribution.json'), 'w') as fp:
        json.dump(refined_resource_distribution, fp)

def get_refined_resource_distribution(raw_dist):
    """
        raw_dist format: {
            "aws_lightsail_instance": 1,
            "aws_s3": 1,
            ...
        }
    """
    refined_dist = defaultdict(int)

    for resource, count in raw_dist.items():
        if "aws_lightsail" in resource:
            refined_dist["aws_lightsail"] += count
        elif "aws_s3" in resource or "aws_glacier" in resource:
            refined_dist["aws_s3"] += count
        elif "aws_sns" in resource:
            refined_dist["aws_sns"] += count
        elif "aws_iam" in resource:
            refined_dist["aws_iam"] += count
        elif "aws_instance" in resource or "aws_ec2" in resource or "aws_ami" in resource or "aws_launch_template" in resource or "aws_placement_group" in resource or "aws_key_pair" in resource:
            refined_dist["aws_ec2"] += count
        elif "aws_lb" in resource or "aws_elb" in resource:
            refined_dist["aws_elb"] += count
        elif "aws_vpc" in resource or "aws_subnet" in resource or "aws_eip" in resource or "aws_egress_only_internet_gateway" in resource or "aws_default_network_acl" in resource or "aws_internet_gateway" in resource or "aws_route_table" in resource or "aws_network_acl" in resource or "aws_vpc_peering_connection" in resource or "aws_nat_gateway" in resource:
            refined_dist["aws_vpc"] += count 
        elif "aws_db" in resource or "aws_rds_cluster" in resource:
            refined_dist["aws_rds"] += count
        elif "aws_security_group" in resource:
            refined_dist["aws_security_group"] += count
        elif "aws_cognito" in resource:
            refined_dist["aws_cognito"] += count
        elif "aws_secretsmanager" in resource:
            refined_dist["aws_secretsmanager"] += count
        elif "aws_backup" in resource:
            refined_dist["aws_backup"] += count
        elif "aws_dynamodb" in resource or "aws_dax" in resource:
            refined_dist["aws_dynamodb"] += count
        elif "aws_kms" in resource:
            refined_dist["aws_kms"] += count
        elif "aws_efs" in resource:
            refined_dist["aws_efs"] += count
        elif "aws_msk" in resource:
            refined_dist["aws_msk"] += count
        elif "aws_cloudwatch" in resource:
            refined_dist["aws_cloudwatch"] += count
        elif "aws_kinesis" in resource:
            refined_dist["aws_kinesis"] += count
        elif "aws_autoscaling_group" in resource:
            refined_dist["aws_autoscaling_group"] += count
        elif "aws_elasticache" in resource:
            refined_dist["aws_elasticache"] += count
        elif "aws_redshift" in resource:
            refined_dist["aws_redshift"] += count
        elif "aws_lambda" in resource:
            refined_dist["aws_lambda"] += count
        elif "aws_sagemaker" in resource:
            refined_dist["aws_sagemaker"] += count
        elif "aws_eks" in resource:
            refined_dist["aws_eks"] += count
        elif "aws_codebuild" in resource:
            refined_dist["aws_codebuild"] += count
        elif "aws_api_gateway" in resource:
            refined_dist["aws_api_gateway"] += count
        elif "aws_cloudfront" in resource:
            refined_dist["aws_cloudfront"] += count
        elif "aws_route53" in resource:
            refined_dist["aws_route53"] += count
        elif "aws_lex" in resource:
            refined_dist["aws_lex"] += count
        elif "aws_connect" in resource:
            refined_dist["aws_connect"] += count
        elif "aws_elasticsearch" in resource or "aws_opensearch" in resource:
            refined_dist["aws_elasticsearch"] += count
        elif "aws_kendra" in resource:
            refined_dist["aws_kendra"] += count
        elif "aws_elastic_beanstalk" in resource:
            refined_dist["aws_elastic_beanstalk"] += count
        elif "aws_sqs" in resource:
            refined_dist["aws_sqs"] += count
        elif "aws_neptune" in resource:
            refined_dist["aws_neptune"] += count
        elif "aws_chime" in resource:
            refined_dist["aws_chime"] += count
        else: 
            refined_dist[resource] += count

    return refined_dist

def get_statistics(list1, list1_name):
    list1 = sorted(list1)
    x = numpy.quantile(list1, [0,0.25,0.5,0.75,1])
    min1 = x[0]
    max1 = x[4]
    median = x[2]
    q1 = x[1]
    q3 = x[3]

    print(list1_name + " three quartiles: ", q1, median, q3)
    print(list1_name + " mean: ", statistics.mean(list1))
    print(list1_name + " min: ", min1)
    print(list1_name + " max: ", max1)

def update_complexity_distribution(complexity, complexity_distribution):
    if complexity == "1":
        complexity_distribution["level-1"] += 1
    elif complexity == "2":
        complexity_distribution["level-2"] += 1
    elif complexity == "3":
        complexity_distribution["level-3"] += 1
    elif complexity == "4":
        complexity_distribution["level-4"] += 1
    elif complexity == "5":
        complexity_distribution["level-5"] += 1
    elif complexity == "6":
        complexity_distribution["level-6"] += 1
    return complexity_distribution

def difficulty_and_resources_calculation(row, df, index):
    prompt = row["Prompt"]
    if isinstance(prompt, float):
        if math.isnan(prompt):
            return None, None, None, None
    reference = row["Reference output"]
    policy = row["Rego intent"]

    print("Prompt:", prompt)

    difficulty_level, resource_list, intent_loc, config_metric = calc_difficulty_and_resources(reference, policy, prompt)
    print("Difficulty level: ", difficulty_level)

    df.at[index, "Difficulty"] = difficulty_level
    df.at[index, "Resource"] = ', '.join(resource_list)

    return difficulty_level, resource_list, intent_loc, config_metric

def calc_difficulty_and_resources(reference, intent, prompt):
    """
        (1) Calculate difficulty based on LOC, num resources, and number of interconnections. 
        (2) Obtain all resources, and returns a resource list
        (3) calculate intent loc
    """
    # Calculate LOC:
    LOC = sum(not line.isspace() for line in StringIO(reference))
    print("LOC", LOC)

    # Calculate intent LOC: 
    intent_loc = sum(not line.isspace() for line in StringIO(intent))
    print("Intent LOC: ", intent_loc)

    # Calculate number of resources:
    # Count the number of occurences of the word "resource" in the reference string
    num_resources = reference.count("resource")
    print("num_resources", num_resources)

    terraform_dir = "./misc/complete-dataset-measurement/terraform_config"

    eval.delete_all_files_in_directory(terraform_dir)

    # Calculate number of interconnections:
    write_to_terraform(reference)
    # run terraform plan and capture the output and errors
    plan_file = "plan.out"
    output_json_file = "plan.json"
    output_json_filepath = os.path.join(terraform_dir, output_json_file)
    generate_terraform_plan_json(prompt, terraform_dir, plan_file, output_json_file)

    # Check that Rego intents are correct/parsable: 
    rego_dir = "./misc/complete-dataset-measurement/rego_config"
    rego_policy_filepath = rego_dir + "/policy.rego"
    eval.write_to_rego(intent, rego_policy_filepath)
    opa_result, opa_error = eval.OPA_Rego_evaluation(output_json_filepath, rego_policy_filepath) 
    if "OPA exception occurred" in opa_error:
        sys.exit(opa_error)

    # read the json file
    # print("about to read json..")
    resource_list = []
    with open(output_json_filepath, "r") as json_file:
        data = json.load(json_file)
        # extract the number of interconnections from the json file
        config_graph = data["configuration"]["root_module"]["resources"]
        for i in config_graph:
            resource_list.append(i["type"])
        references_list = list(findkeys(config_graph, "references")) # just a list with the word "references" repeated
        # print(references_list)
        num_interconnections = len(references_list)
        print("num_interconnections", num_interconnections)
    print("Resources found: ", resource_list)
    # Determine difficulty:
    return get_difficulty_level(LOC, num_resources, num_interconnections), resource_list, intent_loc, [LOC, num_resources, num_interconnections]

def get_difficulty_level(LOC, num_resources, num_interconnections):
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

def write_to_terraform(result, terraform_dir="./misc/complete-dataset-measurement/terraform_config"):
    # define the path to the main.tf file
    terraform_file_path = terraform_dir + "/complete-measurement-main.tf"
    os.makedirs(terraform_dir, exist_ok=True)
    # open the file in write mode ('w') and write the result to it
    # print("CWD", os.getcwd())
    print("CWD: {}".format(os.getcwd()))
    with open(terraform_file_path, "w+", encoding="utf-8", errors="ignore") as file:
        file.write(result)
    # print(f"Updated main.tf at {terraform_file_path}")
    print(f"Updated main.tf at {terraform_file_path}")

def plan_error_handling(result):
    error_handled = False

    if "Inconsistent dependency lock file" in result.stderr:
        init_result = subprocess.run(["terraform", "init"], capture_output=True, text=True)
        time.sleep(5)
        error_handled = True
    # assumes only one missing file, at least per iteration.. 
    elif "archive missing file: " in result.stderr: # error seen in data "archive_file"
        print("Missing archive file... trying again. Result stderr is: ")
        print(result.stderr)
        for item in result.stderr.split("\n"): # Get line containing string: https://stackoverflow.com/questions/2557808/search-and-get-a-line-in-python
            missing_filename = extract_filename(item)
            if missing_filename is not None: 
                with open(missing_filename, "w") as file:
                    print("Creating missing archive file: ", missing_filename)
                    file.write("random")
                    error_handled = True
    elif "no file exists at" in result.stderr: # error seen in file("sagemaker-human-task-ui-template.html") attribute value
        print("Missing file... trying again. Result stderr is: ")
        print(result.stderr)
        for item in result.stderr.split("\n"): # Get line containing string: https://stackoverflow.com/questions/2557808/search-and-get-a-line-in-python
            missing_filename = extract_filename(item)
            if missing_filename is not None: 
                with open(missing_filename, "w") as file:
                    print("Creating missing file: ", missing_filename)
                    file.write("random")
                    error_handled = True
    return error_handled
    
def generate_terraform_plan_json(prompt, terraform_dir="./misc/complete-dataset-measurement/terraform_config", plan_file="plan.out", output_json_file="plan.json"):
    cwd = os.getcwd()
    # change to the Terraform directory
    os.chdir(terraform_dir)

    for i in range(3): # try twice
        try:
            # run init before plan
            result = subprocess.run(["terraform", "init"], capture_output=True, text=True)

            error_handled = plan_error_handling(result)

            if error_handled:
                continue

            # generate Terraform plan with the -no-color flag
            result = subprocess.run(
                ["terraform", "plan", "-out", plan_file, "-no-color"], capture_output=True, text=True, timeout=60 
            )
            
            error_handled = plan_error_handling(result)

            if error_handled:
                continue
            
            with open(
                output_json_file, "w", encoding="utf-8", errors="ignore"
            ) as json_file:
                result = subprocess.run(
                    ["terraform", "show", "-json", plan_file], check=True, stdout=json_file
                )

            break
        except Exception as e:
            print("Error occurred for prompt \"{}\": {}".format(prompt, e))

    # Return to parent directory
    os.chdir(cwd)

def main():
    # Find dataset files: 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "../../..", "data")
    # dataset_dir = os.path.join(script_dir, "../../..", "delete-later-fake-complete")
    # print()
    dataset_files = list_all_subdirectories(dataset_dir)
    dataset_files = [f for f in dataset_files if "test" not in f]
    print(dataset_files)

    difficulty_included_dataset_dir = script_dir + "/complete-dataset"
    create_new_dirs_for_metric(difficulty_included_dataset_dir, dataset_files)
    dst_filenames = copy_csv_to_metric(dataset_files, difficulty_included_dataset_dir)
    # print(dst_filenames)
    difficulty_and_resource_calculation_loop(dst_filenames, script_dir) 

    # Display number of rows for each complexity level:

if __name__ == "__main__":
    main()
#     reference = """
# terraform {
#   required_providers {
#     aws = {
#       source  = "hashicorp/aws"
#       version = "~> 4.16"
#     }
#   }

#   required_version = ">= 1.2.0"
# }
# # Define the provider block for AWS
# provider "aws" {
#   region = "us-east-2" # Set your desired AWS region
# }

# variable "vpc_id" {
#   type        = string
#   description = "The VPC to deploy the components within"
#   default    = "vpc-12345678"
# }

# variable "pg_port" {
#   type        = number
#   description = "Postgres connection port"
#   default     = 5432
# }

# variable "pg_superuser_username" {
#   type        = string
#   description = "Username for the 'superuser' user in the Postgres instance"
#   default     = "superuser"
# }

# variable "pg_superuser_password" {
#   type        = string
#   sensitive   = true
#   description = "Password for the 'superuser' user in the Postgres instance"
#   default = "random-password"
# }

# resource "aws_db_subnet_group" "postgres" {
#   name       = "pgsubnetgrp"
#   subnet_ids = [aws_subnet.main1.id, aws_subnet.main2.id]
# }

# resource "aws_subnet" "main1" {
#   vpc_id     = var.vpc_id
#   cidr_block = "10.0.1.0/24"

#   tags = {
#     Name = "Main"
#   }
# }

# resource "aws_subnet" "main2" {
#   vpc_id     = var.vpc_id
#   cidr_block = "10.0.1.0/24"

#   tags = {
#     Name = "Main"
#   }
# }

# resource "aws_db_parameter_group" "postgres" {
#   name   = "pgparamgrp15"
#   family = "postgres15"

#   parameter {
#     name  = "password_encryption"
#     value = "scram-sha-256"
#   }

#   parameter {
#     name  = "rds.force_ssl"
#     value = "0"
#   }

#   lifecycle {
#     create_before_destroy = true
#   }
# }

# resource "aws_security_group" "pg" {
#   name   = "pg"
#   vpc_id = var.vpc_id

#   ingress {
#     description = "Postgres from internet"
#     from_port   = 5432
#     to_port     = 5432
#     cidr_blocks = ["0.0.0.0/0"]
#     protocol    = "TCP"
#     self        = false
#   }
#   egress {
#     description = "Postgres to internet"
#     from_port   = 5432
#     to_port     = 5432
#     cidr_blocks = ["0.0.0.0/0"]
#     protocol    = "TCP"
#     self        = false
#   }
# }

# resource "aws_kms_key" "rds_key" {
#   description             = "kmsrds"
#   deletion_window_in_days = 14
#   tags                    = { Name = "kmsrds" }
# }

# resource "aws_db_instance" "postgres" {
#   identifier                      = "pg"
#   final_snapshot_identifier       = "pgsnapshot"
#   allocated_storage               = 20
#   apply_immediately               = true
#   backup_retention_period         = 7
#   db_subnet_group_name            = aws_db_subnet_group.postgres.name
#   parameter_group_name            = aws_db_parameter_group.postgres.name
#   enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]
#   engine                          = "postgres"
#   engine_version                  = "15"
#   allow_major_version_upgrade     = true
#   instance_class                  = "db.t3.micro"
#   db_name                         = "postgres" # Initial database name
#   username                        = var.pg_superuser_username
#   port                            = var.pg_port
#   password                        = var.pg_superuser_password
#   vpc_security_group_ids          = [aws_security_group.pg.id]
#   # Other security settings
#   publicly_accessible = false
#   multi_az            = true
#   storage_encrypted   = true
#   kms_key_id          = aws_kms_key.rds_key.arn
#   # Default daily backup window
#   # https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/USER_WorkingWithAutomatedBackups.html#USER_WorkingWithAutomatedBackups.BackupWindow
# }
# """
#     prompt = "provisions a secure PostgreSQL database instance within a specified AWS VPC, leveraging AWS services like RDS, subnets, and KMS for encryption. It sets up two subnets within the VPC for the database, a custom parameter group for PostgreSQL settings, and a security group to manage access. The database instance is configured with specifics storage size is 20GB, engine version is 15,  multi-AZ deployment for high availability, and encryption using a KMS key."
#     print(calc_complexity(reference, prompt))