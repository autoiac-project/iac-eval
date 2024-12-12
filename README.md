<p align="center">| <a href="https://huggingface.co/datasets/autoiac-project/iac-eval"><u>Dataset</u></a> | 🏆 <a href="https://huggingface.co/datasets/autoiac-project/iac-eval"><u>Leaderboard TBD</u></a> | 📖 <a href="https://www.cs-pk.com/preprint-iac-eval.pdf"><u>NeurIPS 2024 Paper</u></a> |</p>

# IaC-Eval---first edition

IaC-Eval is a comprehensive framework for quantitatively evaluating the capabilities of large language models in cloud IaC code generation. Infrastructure-as-Code (IaC) is an important component of cloud computing, that allows the definition of cloud infrastructure in high-level programs. Our framework targets Terraform specifically for now. We leave integration of other IaC tools as future work. 

IaC-Eval also provides the first human-curated and challenging Infrastructure-as-Code (IaC) dataset containing 458 questions ranging from simple to difficult across various cloud services (targeting AWS for now), which can be found in our [HuggingFace repository](https://huggingface.co/datasets/autoiac-project/iac-eval).

**We are actively developing and patching the project. However, as of now, IaC-Eval is not production-ready.** 

## Installation

1. Install Terraform (also [install AWS CLI and setup credentials](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/aws-build#prerequisites))
2. Install [Opa](https://www.openpolicyagent.org/docs/latest/#1-download-opa) (make sure to add opa to path).
3. <sup>*</sup>Obtain the following LLM model inference API keys as appropriate, depending on which of our currently supported models you want to perform evaluation on:
- [OpenAI API token](https://platform.openai.com/docs/quickstart/account-setup): for GPT-3.5-Turbo and GPT-4
- [Google API token](https://ai.google.dev/gemini-api/docs/quickstart?lang=python#set-up-api-key): for Gemini-1.0-Pro
- [Replicate API token](https://replicate.com/): for CodeLlama and WizardCoder variants

<sup>*</sup> Our evaluation against MagiCoder was performed on a manually deployed AWS SageMaker instance inference endpoint. We provide more details on our setup script, see `evaluation/README.md`, if that is of interest.  

### Using the Evaluation Pipeline

To access and utilize the evaluation pipeline, you need to switch to a specific branch of this repository and set up the environment. Follow these steps:

1. Ensure you have the `main` branch of the project checked out.

2. Install the Conda environment by running:

   ```shell
   conda env create -f environment.yml
   ```

3. Activate the newly created Conda environment named `iac-eval`:

   ```shell
   conda activate iac-eval
   ```

   Note: before `conda activate` you might need to do `conda init SHELL_NAME` on your preferred shell (e.g. `conda init bash`). If you run into problems initializing the shell session, try referring to [this GitHub issue](https://github.com/conda/conda/issues/13423#issuecomment-2113968807) for a fix.
 
4. (Optional) Preconfigure the retriever database (if you would like to use the RAG strategy): refer to instructions in `retriever/README.md`.

5. See instructions in `evaluation/README.md` for details on how to use the main pipeline: `eval.py`, and other scripts.

Note: You can run `./setup.sh` to check if you have Terraform and OPA installed. It will also create and activate the necessary conda environment. The shell script assumes you are using `bash`, change `#!/bin/SHELL` to your preferred shell in the script.

## Contributing

We welcome all forms of contribution! IaC-Eval aims to quantitatively and comprehensively evaluate the IaC code generation capabilities of large language models. If you find bugs or have ideas, please share them via GitHub Issues. This includes contributions to IaC-Eval's dataset, whose format can be found in it's [HuggingFace repository](https://huggingface.co/datasets/autoiac-project/iac-eval).


## Acknowledgments

<https://github.com/openai/human-eval/tree/master>
