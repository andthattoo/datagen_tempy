import os
import json
import glob
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
import pandas as pd


def collect_md_files_and_push_to_hf():
    # Configuration
    dataset_name = "andthattoo/router-r1-1.5b-5k"
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    data_dir = os.path.join("data", model_name)

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
        dataset_name,
        token=os.environ.get("HF_TOKEN"),  # Make sure you have set HF_TOKEN in your environment
        private=False
    )

    print(f"Dataset successfully pushed to HuggingFace: {dataset_name}")

    # Optionally, you can also use the HfApi to add repository metadata
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.upload_file(
        path_or_fileobj=bytes(json.dumps({
            "tags": ["math", f"{model_name}"]
        }, indent=2).encode("utf-8")),
        path_in_repo=".tags",
        repo_id=dataset_name,
        repo_type="dataset"
    )

    print("Dataset tags updated")


if __name__ == "__main__":
    collect_md_files_and_push_to_hf()