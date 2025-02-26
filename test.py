from llm import LLM
import asyncio

async def main():
    run_llm = LLM()
    completions = await run_llm.get_completions_batch([{"model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "user_query": "What is the capital of France?"}])
    print(completions[0])


if __name__ == "__main__":
    asyncio.run(main())