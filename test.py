from llm import LLM
import asyncio

async def main():
    run_llm = LLM()
    completions = await run_llm.get_completions_batch([{"model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "user_query": "4. If a positive integer is equal to 4 times the sum of its digits, then we call this positive integer a quadnumber. The sum of all quadnumbers is $\\qquad$ .\n\n'}, {'model_name': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', 'user_query': 'One, (20 points) Given $t=\\sqrt{2}-1$. If positive integers $a$, $b$, and $m$ satisfy\n$$\n(a t+m)(b t+m)=17 m\n$$\n\nfind the value of $a b$."}])
    print(completions[0])


    msgs = [
        {"model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "user_query": "What is the capital of France?"},
        {"model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "user_query": "What is the capital of USA?"}
    ]
    completions = await run_llm.get_completions_batch(msgs)
    print(completions[0])
    print(completions[1])

if __name__ == "__main__":
    asyncio.run(main())