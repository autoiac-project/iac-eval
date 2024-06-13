package aws_neptune_cluster

import future.keywords.in

# all attributes are technically optional
# tags - may only contain unicode letters, digits, whitespace, or one of these symbols: _ . : / = + - @
# availability_zone_allowed_values := {"us-east-1a", "us-east-1b", "us-east-1c", "us-east-1d", "us-east-1e", "us-east-1f", ...}

default neptune_cluster_valid := false
default cluster_parameter_group_valid := false

neptune_cluster_valid {
	some cluster in input.configuration.root_module.resources
	cluster.type == "aws_neptune_cluster"
 
	# cluster.expressions.allow_major_version_upgrade.constant_value == true/false # if intended to be specific value
	# is_boolean(cluster.expressions.allow_major_version_upgrade.constant_value) # if intended to be defined but value doesn't matter

	# cluster.expressions.apply_immediately.constant_value == true/false # if intended to be specific value
	# is_boolean(cluster.expressions.apply_immediately.constant_value) # if intended to be defined but value doesn't matter

	# some availability_zone in cluster.expressions.availability_zones.constant_value 
	# availability_zone == "" # if intended to be specific value
	# availability_zone in availability_zone_allowed_values # if intended to be defined but value doesn't matter
    
    # cluster_parameter_group_valid
    
    # TODO: ALL OTHER ATTRIBUTES (they don't seem that important)
}

cluster_parameter_group_valid {
	some cluster in input.configuration.root_module.resources
	cluster.type == "aws_neptune_cluster"
    
    some cluster_parameter_group in input.configuration.root_module.resources
    cluster_parameter_group.type == "aws_neptune_cluster_parameter_group"
    cluster_parameter_group.address in cluster.expressions.neptune_cluster_parameter_group_name.references
    
    # See for more info: https://docs.aws.amazon.com/neptune/latest/userguide/parameter-groups.html
    cluster.expressions.engine_version.constant_value < "1.2.0.0"
    cluster_parameter_group.expressions.family.constant_value == "neptune1"   
}

cluster_parameter_group_valid {
	some cluster in input.configuration.root_module.resources
	cluster.type == "aws_neptune_cluster"
    
    some cluster_parameter_group in input.configuration.root_module.resources
    cluster_parameter_group.type == "aws_neptune_cluster_parameter_group"
    cluster_parameter_group.address in cluster.expressions.neptune_cluster_parameter_group_name.references
    
    # See for more info: https://docs.aws.amazon.com/neptune/latest/userguide/parameter-groups.html
    cluster.expressions.engine_version.constant_value >= "1.2.0.0"
    cluster_parameter_group.expressions.family.constant_value == "neptune1.2"   
}

cluster_parameter_group_valid {
	some cluster in input.configuration.root_module.resources
	cluster.type == "aws_neptune_cluster"
    
    some cluster_parameter_group in input.configuration.root_module.resources
    cluster_parameter_group.type == "aws_neptune_cluster_parameter_group"
    cluster_parameter_group.address in cluster.expressions.neptune_cluster_parameter_group_name.references
    
    # See for more info: https://docs.aws.amazon.com/neptune/latest/userguide/parameter-groups.html
    not cluster.expressions.engine_version.constant_value # defaults as latest version
    cluster_parameter_group.expressions.family.constant_value == "neptune1.2"   
}