from datasets import load_dataset
import pandas as pd
import os

def import_dataset(quick_test=False):
    # Load the entire dataset
    if not quick_test:
        dataset = load_dataset("autoiac-project/iac-eval", split="test")
    else:
        dataset = load_dataset("autoiac-project/iac-eval", "quick_test", split="test")

    # convert the entire dataset to a Pandas DataFrame
    df = dataset.to_pandas()

    # Create directory if not exists:
    os.makedirs("../data/complete", exist_ok=True)

    # Save the dataset to a CSV file
    df.to_csv("../data/complete/data.csv", index=False)