import json

from datasets import load_dataset


ds = load_dataset("andthattoo/router_math_subset")

instructions = []

for d in ds["train"]:
    instruction = d["problem"]
    uuid = d["uuid"]
    gold = d["answer"]
    instructions.append(
        {
            "instruction": instruction,
            "gold": gold,
            "uuid": uuid
        }
    )


with open("instructions.json", "w") as f:
    f.write(json.dumps(instructions, indent=2))