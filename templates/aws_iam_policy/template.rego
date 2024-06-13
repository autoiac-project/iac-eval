package aws_iam_policy

import future.keywords.in

# description - optional
# name - optional (conflicts with name_prefix)
# name_prefix - optional (conflicts with name)
# path - optional
# policy - required
# tags - optional

default iam_policy_valid := false

iam_policy_valid {
    some iam_policy in input.resource_changes
    iam_policy.type == "aws_iam_policy"
    
    expressions := iam_policy.change.after

    # expressions.name == "[name]" # if intended to be specific value

    # expressions.name_prefix.constant_value == "[name_prefix]" # if intended to be specific value

    # expressions.path.constant_value == "[path]" # if intended to be specific value

    # TODO: How to verify if policies are valid
    # expressions.policy 

    # expressions.tags.[tag name] == "[tag value]" # if a tag is intended to be a specific value
}
