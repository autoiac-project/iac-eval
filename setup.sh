#!/bin/bash

set -Eeuo pipefail

# This is basically what `conda init` adds to bashrc
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

# set -x

# Check if terraform is installed
if ! which terraform &> /dev/null; then 
    echo -e 'Make sure you have Terraform installed. Check README for how.' >&2
    exit 1
fi

# Check if opa is installed
if ! which opa &> /dev/null; then
    echo -e 'Make sure you have OPA installed. Check REAMD for how.' >&2
    exit 1
fi

# Create the conda environment if it doesn't exist
if ! conda list --name iac-eval &> /dev/null; then
    echo -e 'iac-eval conda environment not yet created. Creating the environment...'
    conda env create -f environment.yml
fi

# Activate the conda environment
conda activate iac-eval
