from openai import AsyncOpenAI
import asyncio
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())

VLLM_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "api_key"

OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


class LLM:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=VLLM_API_KEY, base_url=VLLM_URL)

    async def get_completion(
            self,
        model_name: str, user_query: str
    ) -> str:
        """
        Get a completion from a model for a given provider.
        """
        messages = [
            {"role": "user", "content": user_query},
        ]

        # Add error handling for API requests
        try:
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.6,
                max_tokens=8192,
                timeout=300.0  # Add a timeout to prevent hanging
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during API call: {e}")
            return f"Error occurred: {str(e)}"

    async def get_completions_batch(self, requests):
        tasks = []
        for r in requests:
            tasks.append(
                self.get_completion(
                    r["model_name"],
                    r["user_query"]
                )
            )
        return await asyncio.gather(*tasks)