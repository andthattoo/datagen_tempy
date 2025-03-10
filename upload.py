import os
import json
import glob
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login
import pandas as pd
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Collect markdown files and push to Hugging Face')
    parser.add_argument('--dataset_name', type=str, default="andthattoo/router-r1-7b-5k",
                        help='Name of the dataset on Hugging Face (default: andthattoo/router-r1-7b-5k)')
    parser.add_argument('--model_name', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help='Name of the model used to generate the data (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)')
    return parser.parse_args()


def collect_md_files_and_push_to_hf():
    # Parse command-line arguments
    args = parse_arguments()
    dataset_name = args.dataset_name
    model_name = args.model_name

    # Extract model short name for the directory
    model_short_name = model_name.split("/")[-1]
    data_dir = os.path.join("data", model_short_name)

    print(f"Model: {model_name}")
    print(f"Dataset destination: {dataset_name}")
    print(f"Looking for .md files in {data_dir}")

    # Get all .md files in the directory
    md_files = glob.glob(os.path.join(data_dir, "*.md"))
    print(f"Found {len(md_files)} .md files")

    if not md_files:
        print("No .md files found. Exiting.")
        return

    # Initialize empty lists to store data
    all_data = []

    # Process each file
    for file_path in md_files:
        try:
            with open(file_path, 'r') as f:
                file_content = f.read()

            # Parse JSON content
            data = json.loads(file_content)

            # Convert label string to boolean if needed
            if isinstance(data.get("label"), str):
                if data["label"].isdigit():
                    data["label"] = bool(int(data["label"]))
                else:
                    data["label"] = data["label"].lower() == "true"

            all_data.append(data)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    print(f"Successfully processed {len(all_data)} files")

    # Convert to DataFrame and then to Dataset
    df = pd.DataFrame(all_data)
    dataset = Dataset.from_pandas(df)

    # Create dataset dictionary
    dataset_dict = DatasetDict({
        "train": dataset
    })

    print("Dataset created. Preparing to push to HuggingFace...")

    # Push to HuggingFace
    dataset_dict.push_to_hub(
        dataset_name,  # Make sure you have set HF_TOKEN in your environment
        private=False
    )

    print(f"Dataset successfully pushed to HuggingFace: {dataset_name}")

    # Optionally, you can also use the HfApi to add repository metadata
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.upload_file(
        path_or_fileobj=bytes(json.dumps({
            "tags": ["math", model_short_name]
        }, indent=2).encode("utf-8")),
        path_in_repo=".tags",
        repo_id=dataset_name,
        repo_type="dataset"
    )

    print("Dataset tags updated")


if __name__ == "__main__":
    login()
    collect_md_files_and_push_to_hf()