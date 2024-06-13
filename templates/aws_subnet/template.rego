package aws_subnet
import future.keywords.in

# vpc_id - required
# assign_ipv6_address_on_creation - optional
# availability_zone - optional
# availability_zone_id - optional - DO NOT USE (ACCORDING TO DOCUMENTATION). USE availability_zone INSTEAD
availability_zone_allowed_values := {"us-east-1a", "us-east-1b", "us-east-1c", "us-east-1d", "us-east-1e", "us-east-1f"} # this list goes on for a while...
# cidr_block - optional
# customer_owned_ipv4_pool - optional
# enable_dns64 - optional
# enable_lni_at_device_index - optional (is number >= 0)
# enable_resource_name_dns_aaaa_record_on_launch - optional
# enable_resource_name_dns_a_record_on_launch - optional
# ipv6_cidr_block - optional
# ipv6_native - optional
# map_customer_owned_ip_on_launch - optional - if true, customer_owned_ipv4_pool and outpost_arn must be specified
# map_public_ip_on_launch - optional
# outpost_arn - optional
# private_dns_hostname_type_on_launch - optional
private_dns_hostname_type_on_launch_allowed_values := {"ip-name", "resource-name"}
# tags - optional

default subnet_valid := false

default map_customer_owned_ip_on_launch_valid := false


subnet_valid {
	some subnet in input.configuration.root_module.resources
	subnet.type == "aws_subnet"
    expressions := subnet.expressions
    
    # expressions.assign_ipv6_address_on_creation.constant_value == "" # if intended to be specific value
    # is_boolean(expressions.assign_ipv6_address_on_creation.constant_value) # if intended to be defined but value doesn't matter

    # expressions.availability_zone.constant_value == "" # if intended to be specific value
    # expressions.availability_zone.constant_value in availability_zone_allowed_values # if intended to be defined but value doesn't matter

    # expressions.cidr_block.constant_value == "" # if intended to be specific value
    # net.cidr_is_valid(expressions.cidr_block.constant_value) # if intended to be defined but value doesn't matter

    # TODO: customer_owned_ipv4_pool

    # expressions.enable_dns64.constant_value == "" # if intended to be specific value
    # is_boolean(expressions.enable_dns64.constant_value) # if intended to be defined but value doesn't matter

    # expressions.enable_lni_at_device_index.constant_value ==  # if intended to be specific value
    # expressions.enable_lni_at_device_index.constant_value >= 0 # if intended to be defined but value doesn't matter

    # expressions.enable_resource_name_dns_aaaa_record_on_launch.constant_value == "" # if intended to be specific value
    # is_boolean(expressions.enable_resource_name_dns_aaaa_record_on_launch.constant_value) # if intended to be defined but value doesn't matter

    # expressions.enable_resource_name_dns_a_record_on_launch.constant_value == "" # if intended to be specific value
    # is_boolean(expressions.enable_resource_name_dns_a_record_on_launch.constant_value) # if intended to be defined but value doesn't matter
    
    # expressions.ipv6_cidr_block.constant_value == "" # if intended to be specific value
    # net.cidr_is_valid(expressions.ipv6_cidr_block.constant_value) # if intended to be defined but value doesn't matter

    # expressions.ipv6_native.constant_value == "" # if intended to be specific value
    # is_boolean(expressions.ipv6_native.constant_value) # if intended to be defined but value doesn't matter

    map_customer_owned_ip_on_launch_valid

    # expressions.map_public_ip_on_launch.constant_value == "" # if intended to be specific value
    # is_boolean(expressions.map_public_ip_on_launch.constant_value) # if intended to be defined but value doesn't matter

    # TODO: outpost_arn

    # expressions.private_dns_hostname_type_on_launch.constant_value == "" # if intended to be specific value
    # expressions.private_dns_hostname_type_on_launch.constant_value in availability_zone_allowed_values # if intended to be defined but value doesn't matter

    some vpc in input.configuration.root_module.resources
	vpc.type == "aws_vpc"
	vpc.address in expressions.vpc_id.references

    # expressions.tags.constant_value.[tag name] == "" # if a tag is intended to be a specific value
}



map_customer_owned_ip_on_launch_valid { # if not defined (default initialized)
    some subnet in input.configuration.root_module.resources
	subnet.type == "aws_subnet"
	not subnet.expressions.map_customer_owned_ip_on_launch
}
map_customer_owned_ip_on_launch_valid { # if defined and false
    some subnet in input.configuration.root_module.resources
	subnet.type == "aws_subnet"
    subnet.expressions.map_customer_owned_ip_on_launch.constant_value == false
}
map_customer_owned_ip_on_launch_valid { # if defined and true
    some subnet in input.configuration.root_module.resources
	subnet.type == "aws_subnet"
    subnet.expressions.map_customer_owned_ip_on_launch.constant_value == true

	subnet.expressions.customer_owned_ipv4_pool # required attribute
	subnet.expressions.outpost_arn # required attribute
}