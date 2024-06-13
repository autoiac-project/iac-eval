package aws_vpc

# all vpc attributes - optional

import future.keywords.in

default vpc_valid := false

default cidr_block_is_valid := false
instance_tenancy_allowed_values := {"default", "dedicated"}
default ipv4_ipam_pool_id_is_valid := false
default ipv4_netmask_length_is_valid := false
default ipv6_cidr_block_is_valid := false
default ipv6_ipam_pool_id_is_valid := false
default ipv6_netmask_length_is_valid := false
default assign_generated_ipv6_cidr_block_is_valid := false

vpc_valid {
	some vpc in input.configuration.root_module.resources
	vpc.type == "aws_vpc"
    expressions := vpc.expressions

	cidr_block_is_valid

    # expressions.instance_tenancy.constant_value == "" # if intended to be specific value
    # expressions.instance_tenancy.constant_value in instance_tenancy_allowed_values # if intended to be defined but value doesn't matter

    ipv4_ipam_pool_id_is_valid
    ipv4_netmask_length_is_valid

    ipv6_cidr_block_is_valid
    ipv6_ipam_pool_id_is_valid
    ipv6_netmask_length_is_valid

    # expressions.enable_dns_support.constant_value == "" # if intended to be specific value
    # is_boolean(expressions.enable_dns_support.constant_value) # if intended to be defined but value doesn't matter

    # expressions.enable_network_address_usage_metrics.constant_value == "" # if intended to be specific value
    # is_boolean(expressions.enable_network_address_usage_metrics.constant_value) # if intended to be defined but value doesn't matter

    # expressions.enable_dns_hostnames.constant_value == "" # if intended to be specific value
    # is_boolean(expressions.enable_dns_hostnames.constant_value) # if intended to be defined but value doesn't matter

    assign_generated_ipv6_cidr_block_is_valid

    # expressions.tags.constant_value.[tag name] == "" # if a tag is intended to be a specific value
}


cidr_block_is_valid { # if not defined (default initialized)
	some vpc in input.configuration.root_module.resources
	vpc.type == "aws_vpc"
	not vpc.expressions.cidr_block 
}
cidr_block_is_valid { # if defined
	some vpc in input.configuration.root_module.resources
	vpc.type == "aws_vpc"
	net.cidr_is_valid(vpc.expressions.cidr_block)
    not vpc.expressions.ipv4_ipam_pool_id # conflicting attribute
}



ipv4_ipam_pool_id_is_valid { # if not defined (default initialized)
    some vpc in input.configuration.root_module.resources
	vpc.type == "aws_vpc"
	not vpc.expressions.ipv4_ipam_pool_id
}
ipv4_ipam_pool_id_is_valid { # if defined
    some vpc in input.configuration.root_module.resources
	vpc.type == "aws_vpc"

    some ipam_pool in input.configuration.root_module.resources
	ipam_pool.type == "aws_vpc_ipam_pool"
    ipam_pool.expressions.address_family.constant_value == "ipv4"
    ipam_pool.address in vpc.expressions.ipv4_ipam_pool_id.references
    
    not vpc.expressions.cidr_block # conflicting attributes
}



ipv4_netmask_length_is_valid { # if not defined (default initialized)
    some vpc in input.configuration.root_module.resources
	vpc.type == "aws_vpc"
	not vpc.expressions.ipv4_netmask_length
}
ipv4_netmask_length_is_valid { # if defined
    some vpc in input.configuration.root_module.resources
	vpc.type == "aws_vpc"
    vpc.expressions.ipv4_ipam_pool_id # required attributes

	vpc.expressions.ipv4_netmask_length.constant_value <= 32
	vpc.expressions.ipv4_netmask_length.constant_value >= 0 
}



ipv6_cidr_block_is_valid {
    some vpc in input.configuration.root_module.resources
	vpc.type == "aws_vpc"
	not vpc.expressions.ipv6_cidr_block
}
ipv6_cidr_block_is_valid {
    some vpc in input.configuration.root_module.resources
	vpc.type == "aws_vpc"
	net.cidr_is_valid(vpc.expressions.ipv6_cidr_block)

    not vpc.expressions.ipv6_ipam_pool_id # conflicting attribute
}



ipv6_ipam_pool_id_is_valid {
    some vpc in input.configuration.root_module.resources
	vpc.type == "aws_vpc"
	not vpc.expressions.ipv6_ipam_pool_id
}
ipv6_ipam_pool_id_is_valid {
    some vpc in input.configuration.root_module.resources
	vpc.type == "aws_vpc"

	some ipam_pool in input.configuration.root_module.resources
	ipam_pool.type == "aws_vpc_ipam_pool"
    ipam_pool.expressions.address_family.constant_value == "ipv6"
    ipam_pool.address in vpc.expressions.ipv6_ipam_pool_id.references

    not vpc.expressions.ipv6_cidr_block # conflicting attribute
}



ipv6_netmask_length_is_valid { # if not defined (default initialized)
    some vpc in input.configuration.root_module.resources
	vpc.type == "aws_vpc"
	not vpc.expressions.ipv6_netmask_length
}
ipv6_netmask_length_is_valid { # if defined
    some vpc in input.configuration.root_module.resources
	vpc.type == "aws_vpc"
    vpc.expressions.ipv6_ipam_pool_id # required attributes
	vpc.expressions.ipv6_netmask_length.constant_value == 56
}



assign_generated_ipv6_cidr_block_is_valid {
    some vpc in input.configuration.root_module.resources
	vpc.type == "aws_vpc"
	not vpc.expressions.assign_generated_ipv6_cidr_block
}
assign_generated_ipv6_cidr_block_is_valid {
    some vpc in input.configuration.root_module.resources
	vpc.type == "aws_vpc"
	is_boolean(vpc.expressions.assign_generated_ipv6_cidr_block)

    not vpc.expressions.ipv6_ipam_pool_id # conflicting resource
}