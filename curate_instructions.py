from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import random

# Load the dataset
ds = load_dataset("open-r1/OpenR1-Math-220k", "default")

num_samples = 500

# Convert to pandas DataFrame for easier manipulation
train_df = ds["train"].to_pandas()

# Filter out rows where all correctness_math_verify values are False
# And add a new column for the maximum length of correct generations
filtered_rows = []
for _, row in train_df.iterrows():
    # Get the boolean mask and generation texts
    correctness = row['correctness_math_verify']
    generations = row['generations']

    # Check if at least one generation is correct
    if any(correctness):
        # Get lengths of only the correct generations
        correct_generation_lengths = [len(gen) for gen, is_correct in zip(generations, correctness) if is_correct]

        # If we have any correct generations, calculate the max length
        if correct_generation_lengths:
            max_len = max(correct_generation_lengths)

            # Only include rows where the max_correct_generation_length is less than 4096
            if max_len < 8192:
                # Create a new row with the max_len column
                new_row = row.copy()
                new_row['max_correct_generation_length'] = max_len
                filtered_rows.append(new_row)

# Create new dataframe with filtered rows
filtered_df = pd.DataFrame(filtered_rows)

# Get all unique problem types
problem_types = filtered_df["problem_type"].unique()

# Sample examples from each problem type
sampled_dfs = []
for problem_type in problem_types:
    type_df = filtered_df[filtered_df["problem_type"] == problem_type]

    # If there are fewer than num_samples examples, take all of them
    if len(type_df) <= num_samples:
        sampled_df = type_df
        print(f"Problem type '{problem_type}' has only {len(type_df)} examples")
    else:
        # Sample num_samples examples randomly
        sampled_df = type_df.sample(n=num_samples, random_state=19)

    sampled_dfs.append(sampled_df)

# Combine all the sampled dataframes
final_df = pd.concat(sampled_dfs, ignore_index=True)

# Shuffle the combined dataset
final_df = final_df.sample(frac=1, random_state=19).reset_index(drop=True)

# Convert back to a Hugging Face Dataset
sampled_dataset = Dataset.from_pandas(final_df)

# Create a DatasetDict with train split
sampled_ds = DatasetDict({"train": sampled_dataset})

# Print some statistics
print(f"Original dataset size: {len(ds['train'])}")
print(f"Filtered dataset size (correct solutions with length < 4096): {len(filtered_df)}")
print(f"Sampled dataset size: {len(sampled_dataset)}")
print(f"Number of problem types: {len(problem_types)}")

# Print the distribution of problem types
for problem_type in problem_types:
    count = len(final_df[final_df["problem_type"] == problem_type])
    print(f"Problem type '{problem_type}': {count} examples")

# Print distribution of max_correct_generation_length
print("\nDistribution of max_correct_generation_length:")
print(f"Min length: {final_df['max_correct_generation_length'].min()}")
print(f"Max length: {final_df['max_correct_generation_length'].max()}")
print(f"Mean length: {final_df['max_correct_generation_length'].mean():.2f}")
print(f"Median length: {final_df['max_correct_generation_length'].median()}")

# Push the dataset to the Hub
sampled_ds.push_to_hub(
    "andthattoo/router_math_subset_below_8k",
    private=False,
    token=None
)