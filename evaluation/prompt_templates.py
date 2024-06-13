def CoT_prompt(question_prompt):
    with open('prompt-templates/CoT.txt', 'r') as file:
        data = file.read()
    prompt = data + question_prompt # https://www.promptingguide.ai/techniques/cot#zero-shot-cot-prompting
    return prompt

def FSP_prompt(question_prompt):
    with open('prompt-templates/few-shot.txt', 'r') as file:
        data = file.read()
    prompt = data + question_prompt
    return prompt

def RAG_prompt(context, question_prompt): 
    template = """
Here is some additional knowledge/context retrieved from Terraform documentation, that may (or may not) potentially help you answer the question:
{}

Here is the actual prompt to answer:
{}
    """.format(context, question_prompt)
    return template

def multi_turn_system_prompt():
    with open('prompt-templates/multi-turn-system-prompt.txt', 'r') as file:
        data = file.read()
    return data

def multi_turn_plan_error_prompt(question_prompt, candidate_config, error_message):
    prompt = """
Here is the original prompt:
{}

Here is the incorrect configuration:
{}

Here is the Terraform plan error message (potentially empty):
{}
""".format(question_prompt, candidate_config, error_message)
    return prompt

def multi_turn_rego_error_prompt(question_prompt, candidate_config, rego_policy, error_message):
    prompt = """
Here is the original prompt:
{}

Here is the incorrect configuration:
{}

Here is the Rego OPA policy associated with this configuration:
{}

Here is the Rego OPA policy error message:
{}
""".format(question_prompt, candidate_config, rego_policy, error_message)

    return prompt