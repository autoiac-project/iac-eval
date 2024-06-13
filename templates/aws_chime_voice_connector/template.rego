package aws_chime_voice_connector

# name - required
# require_encryption - required
# aws_region - optional
# tags - optional

import future.keywords.in

default chime_voice_connector_valid := false

chime_voice_connector_valid {
    some chime_v_c in input.configuration.root_module.resources
    chime_v_c.type == "aws_chime_voice_connector"

    expressions := chime_v_c.expressions
    # expressions.name == ""
    is_boolean(expressions.require_encryption.constant_value)
    # expressions.aws_region.constant_value == ""
    # expressions.tags.constant_value.[tag name] == ""
}
