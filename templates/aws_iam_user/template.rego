package aws_iam_user

import future.keywords.in

# name - required
# path - optional
# permissions_boundary - optional
# force_destroy - optional
# tags - optional

default iam_user_valid := false

iam_user_valid {
    some iam_user in input.configuration.root_module.resources
    iam_user.type == "aws_iam_user"
    
    expressions := iam_user.expressions

    # expressions.name == "[name]" # if intended to be specific value

    # expressions.path.constant_value == "[path]" # if intended to be specific value

    # TODO: permissions_boundary

    # expressions.force_destroy.constant_value == [true/false]

    # expressions.tags.[tag name] == "[tag value]" # if a tag is intended to be a specific value
}
