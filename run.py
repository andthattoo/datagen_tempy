from llm import LLM
import json
import asyncio
import os
from tqdm import tqdm
import uuid
from pydantic import BaseModel

class Output(BaseModel):
    instruction: str
    reasoning: str
    answer: str
    model: str
    gold: str
    label: bool
    uuid: str

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


async def main():
    # Create data folder and model-specific subfolder
    data_dir = os.path.join("data", model_name)
    os.makedirs(data_dir, exist_ok=True)

    with open("instructions.json", "r") as f:
        instructions = json.load(f)
    # Process in batches
    batch_size = 4
    instructions = instructions[:10]
    total_instructions = len(instructions)

    run_llm = LLM()

    # Overall progress bar
    overall_progress = tqdm(total=total_instructions, desc="Overall Progress", position=0)

    for i in range(0, total_instructions, batch_size):
        # Calculate actual batch size (might be smaller for the last batch)
        current_batch_size = min(batch_size, total_instructions - i)
        batch = instructions[i:i + current_batch_size]
        send_batch = [{"model_name": model_name, "user_query": b["instruction"]} for b in batch]
        # Create a specific progress bar for this batch
        batch_desc = f"Batch {i // batch_size + 1}/{(total_instructions + batch_size - 1) // batch_size}"
        with tqdm(total=current_batch_size, desc=batch_desc, position=1, leave=False) as batch_progress:

            completions = await run_llm.get_completions_batch(send_batch)
            print(completions[0])

            # Save each completion to its own file using the uuid from instructions
            for j, completion in enumerate(completions):
                if i + j < total_instructions:  # Safety check
                    instruction = batch[j]

                    # Extract UUID from instruction or generate one if not present
                    if "uuid" in instruction:
                        file_uuid = instruction["uuid"]
                    else:
                        file_uuid = str(uuid.uuid4())

                    # Create filename and save
                    filename = f"{file_uuid}.md"
                    filepath = os.path.join(data_dir, filename)

                    with open(filepath, "w") as f:
                        f.write(completion)

                    batch_progress.update(1)
                    overall_progress.update(1)

    overall_progress.close()
    print("\nAll batches processed and saved")


if __name__ == "__main__":
    asyncio.run(main())