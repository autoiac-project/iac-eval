#!/bin/bash

set -Eeuo pipefail

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

# Create the conda environment if it doesn't exist
if ! conda list --name iac-eval &> /dev/null; then
    echo -e 'iac-eval conda environment not yet created. Creating the environment...'
    conda env create -f ../environment.yml
fi

# Activate the conda environment
conda activate iac-eval

# Clone the git repository if it doesn't exist
if [ ! -d "terraform-provider-aws" ]; then
    git clone https://github.com/hashicorp/terraform-provider-aws
fi

python3 llama_index_retriever.py
