import json

from datasets import load_dataset


ds = load_dataset("andthattoo/router_math_subset_below_8k")

instructions = []
correct_gens = []
for d in ds["train"]:
    instruction = d["problem"]
    uuid = d["uuid"]
    gold = d["answer"]
    correct_gens.append(d["max_correct_generation_length"])
    instructions.append(
        {
            "instruction": instruction,
            "gold": gold,
            "uuid": uuid
        }
    )


with open("instructions.json", "w") as f:
    f.write(json.dumps(instructions, indent=2))


"""
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
data = np.array(correct_gens)
min_val = min(data)
max_val = max(data)
# Start from a multiple of 3000 below the minimum value
start_bin = (min_val // 3000) * 3000
# End at a multiple of 3000 above the maximum value
end_bin = ((max_val // 3000) + 1) * 3000
# Create the bin edges
bin_edges = np.arange(start_bin, end_bin + 3000, 3000)

# Create the histogram
plt.figure(figsize=(10, 6))
plt.hist(data, bins=bin_edges, edgecolor='black', alpha=0.7)

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram with Bin Size of 3000')

# Add grid for better readability
plt.grid(axis='y', alpha=0.75)

# Show the plot
plt.tight_layout()
plt.show()
"""