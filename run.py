from llm import LLM
import json
import asyncio
import os
from tqdm import tqdm
import uuid
from pydantic import BaseModel
from math_verify import parse, verify


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

    # Process in smaller batches with fewer concurrent requests
    batch_size = 5  # Try a smaller batch size
    instructions = instructions[:10]
    total_instructions = len(instructions)

    run_llm = LLM()

    # Use a simpler progress tracking approach
    for i in range(0, total_instructions, batch_size):
        # Calculate actual batch size (might be smaller for the last batch)
        current_batch_size = min(batch_size, total_instructions - i)
        batch = instructions[i:i + current_batch_size]
        send_batch = [{"model_name": model_name, "user_query": b["instruction"]} for b in batch]

        print(f"Processing batch {i // batch_size + 1}/{(total_instructions + batch_size - 1) // batch_size}")

        try:
            # Add a timeout to the gather operation
            completions = await asyncio.wait_for(
                run_llm.get_completions_batch(send_batch),
                timeout=60  # 60 second timeout
            )

            # Process the completions
            for j, completion in enumerate(completions):
                if i + j < total_instructions:
                    instruction = batch[j]

                    # Extract UUID from instruction or generate one
                    file_uuid = instruction.get("uuid", str(uuid.uuid4()))

                    # Create filename and save
                    filename = f"{file_uuid}.md"
                    filepath = os.path.join(data_dir, filename)

                    answer = completion if "</think>" not in completion else completion.split("</think>")[1]
                    ans = parse(answer)
                    gold = parse(instruction["gold"])
                    label = verify(gold, ans)

                    data = json.dumps(
                        {
                            "instruction": instruction["instruction"],
                            "completion": completion,
                            "answer": ans,
                            "model": model_name,
                            "gold": instruction["gold"],
                            "label": label,
                            "uuid": file_uuid,
                        }
                    )
                    with open(filepath, "w") as f:
                        f.write(completion)

                    print(f"Saved completion {i + j + 1}/{total_instructions}")

        except asyncio.TimeoutError:
            print(f"Batch {i // batch_size + 1} timed out after 60 seconds")
        except Exception as e:
            print(f"Error processing batch: {e}")

        # Add a small delay between batches to prevent overloading the server
        if i + batch_size < total_instructions:
            await asyncio.sleep(1)

    print("\nAll batches processed and saved")


if __name__ == "__main__":
    asyncio.run(main())