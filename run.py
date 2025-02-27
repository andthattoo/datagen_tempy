from llm import LLM
import json
import asyncio
import os
from tqdm import tqdm
import uuid
from pydantic import BaseModel
from math_verify import parse, verify
import argparse


class Output(BaseModel):
    instruction: str
    reasoning: str
    answer: str
    model: str
    gold: str
    label: bool
    uuid: str


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run math problem evaluation with configurable parameters')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for processing instructions (default: 4)')
    parser.add_argument('--model_name', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help='Model name to use (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)')
    parser.add_argument('--timeout', type=int, default=100,
                        help='Timeout in seconds for batch processing (default: 80)')
    return parser.parse_args()


async def main():
    args = parse_arguments()
    model_name = args.model_name
    batch_size = args.batch_size
    timeout = args.timeout

    print(f"Using model: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"Timeout: {timeout} seconds")

    # Create data folder and model-specific subfolder
    data_dir = os.path.join("data", model_name.split("/")[-1])
    os.makedirs(data_dir, exist_ok=True)

    with open("instructions.json", "r") as f:
        instructions = json.load(f)

    total_instructions = len(instructions)
    print(f"Total instructions to process: {total_instructions}")

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
                timeout=timeout
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
                            "answer": str(ans),
                            "model": model_name,
                            "gold": instruction["gold"],
                            "label": str(int(label)),
                            "uuid": file_uuid,
                        }, indent=2
                    )
                    with open(filepath, "w") as f:
                        f.write(data)

                    print(f"Saved completion {i + j + 1}/{total_instructions}")

        except asyncio.TimeoutError:
            print(f"Batch {i // batch_size + 1} timed out after {timeout} seconds")
        except Exception as e:
            print(f"Error processing batch: {e}")

        # Add a small delay between batches to prevent overloading the server
        if i + batch_size < total_instructions:
            await asyncio.sleep(1)

    print("\nAll batches processed and saved")


if __name__ == "__main__":
    asyncio.run(main())