# BLEU
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import code_bert_score
# reference = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
# candidate = ['this', 'is', 'a', 'test']
# score = sentence_bleu(reference, candidate)
# print(score)
# Exact match:
def bleu_score(reference, candidate):
    reference_tokens = reference.split()
    result_tokens = candidate.split()
    if len(reference_tokens) < 4:
        print("Reference code has less than 4 tokens:".format(reference))
        return 0
    if len(result_tokens) < 4:
        print("Result code has less than 4 tokens:".format(candidate))
        return 0
    score = corpus_bleu([[reference_tokens]], [result_tokens], weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method3)
    # A perfect match results in a score of 1.0, whereas a perfect mismatch results in a score of 0.0. Standard implementation from NLTK based on the actual paper. 
    # return sentence_bleu(reference, candidate)
    return score

def get_code_bert_score(reference, candidate, prompt):
    # https://github.com/neulab/code-bert-score
    pred_results = code_bert_score.score(cands=[candidate], refs=[reference], lang='python', sources=[prompt]) # returns a 4-tuple of (precision, recall, F1, F3), where each is a 1-D tensor of scores for each prediction-reference pair
    return {
        "precision": pred_results[0].item(), 
        "recall": pred_results[1].item(), 
        "f1": pred_results[2].item(), 
        "f3": pred_results[3].item()
    }

def exact_match(reference, candidate):
    return reference.strip() == candidate.strip()

def llm_as_judge_single(prompt, candidate):
    # Fig 6 of the paper: https://arxiv.org/pdf/2306.05685
    template = """
Please act as an impartial judge and evaluate the correctness of the code provided by an AI assistant to the user question displayed below. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, please rate the response as either correct or incorrect by strictly following this format: "Rating: Correct" or "Rating: Incorrect".

[Question] 
{}

[The Start of Assistant’s Answer]
{}
[The End of Assistant’s Answer]
""".format(prompt, candidate)
    # return bleu_score(reference, candidate)
    return template

def llm_as_judge_reference(reference, candidate, llm="gpt4"):
    # Fig 8 of the paper: https://arxiv.org/pdf/2306.05685. Edited to not have an "assistant B". 
    template = """
Please act as an impartial judge and evaluate the correctness of the code provided by an AI assistant to the user question displayed below. You will be given a reference answer, and the assistant's answer. Your job is to evaluate if the assistant's answer is correct or incorrect, in comparison to the reference answer. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "Rating: Correct" or "Rating: Incorrect".

[User Question]
{}

[The Start of Reference Answer]
{}
[The End of Reference Answer]

[The Start of Assistant’s Answer]
{}
[The End of Assistant’s Answer]
""".format(prompt, reference, candidate)

    # return bleu_score(reference, candidate)
    return template
    
# reference = """
# terraform {
#   required_providers {
#     aws = {
#       source  = "hashicorp/aws"
#       version = "~> 4.16"
#     }
#   }

#   required_version = ">= 1.2.0"
# }

# provider "aws" {
#   region = "us-west-2"
# }

# resource "aws_iam_role" "test_role" {
#   name = "test_role"

#   assume_role_policy = jsonencode({
#     Version = "2012-10-17"
#     Statement = [
#       {
#         Action = "sts:AssumeRole"
#         Effect = "Allow"
#         Sid    = ""
#         Principal = {
#           Service = "codebuild.amazonaws.com"
#         }
#       },
#     ]
#   })

#   tags = {
#     tag-key = "test_role"
#   }
# }


# resource "aws_codebuild_project" "example" {
#   name          = "test-project"
#   service_role  = aws_iam_role.test_role.arn

#   artifacts {
#     type = "NO_ARTIFACTS"
#   }

#   environment {
#     compute_type                = "BUILD_GENERAL1_SMALL"
#     image                       = "aws/codebuild/amazonlinux2-x86_64-standard:4.0"
#     type                        = "LINUX_CONTAINER"
#     image_pull_credentials_type = "CODEBUILD"

#   }

#   source {
#     type            = "GITHUB"
#     location        = "https://github.com/neilbalch/SimplePythonTutorial.git"
#     git_clone_depth = 1
#   }
# }terraform {
#   required_providers {
#     aws = {
#       source  = "hashicorp/aws"
#       version = "~> 4.16"
#     }
#   }

#   required_version = ">= 1.2.0"
# }

# provider "aws" {
#   region = "us-west-2"
# }
# """


# candidate = """
# resource "aws_codebuild_project" "example" {
#   name = "example"
#   description = "Example CodeBuild project"
#   source {
#     type = "GITHUB"
#     location = "https://github.com/hashicorp/example-codebuild"
#   }
#   artifacts {
#     type = "CODEPIPELINE"
#     location = "codepipeline"
#     path = "output"
#     namespace_type = "NONE"
#   }
#   environment {
#     type = "LINUX_CONTAINER"
#     image = "aws/codebuild/standard:5.0"
#     compute_type = "BUILD_GENERAL1_SMALL"
#   }
#   service_role = "arn:aws:iam::111122223333:role/example-codebuild-role"
# }

# resource "aws_iam_role" "example" {
#   name = "example-codebuild-role"
#   assume_role_policy = jsonencode({
#     Version = "2012-10-17"
#     Statement = [{
#       Action = "sts:AssumeRole"
#       Effect = "Allow"
#       Principal = {
#         Service = "codebuild.amazonaws.com"
#       }
#     }]
#   })

#   inline_policy {
#     name = "example"
#     policy = jsonencode({
#       Version = "2012-10-17"
#       Statement = [{
#         Action = ["codebuild:BatchGetBuilds", "codebuild:StartBuild"],
#         Effect = "Allow"
#         Resource = "*"
#       }, {
#         Action = ["ecr:GetAuthorizationToken", "ecr:PutImage"],
#         Effect = "Allow"
#         Resource = "arn:aws:ecr:us-west-2:111122223333:repository/example"
#       }, {
#         Action = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
#         Effect = "Allow"
#         Resource = "arn:aws:logs:us-west-2:111122223333:log-group:/aws/codebuild/*"
#       }, {
#         Action = ["s3:GetObject", "s3:GetObjectVersion", "s3:PutObject"],
#         Effect = "Allow"
#         Resource = "arn:aws:s3:::example/*"
#       }, {
#         Action = ["secretsmanager:GetSecretValue"],
#         Effect = "Allow"
#         Resource = "arn:aws:secretsmanager:us-west-2:111122223333:secret:example"
#       }]
#     })
#   }
# }
# """

# print(bleu_score(reference, candidate))