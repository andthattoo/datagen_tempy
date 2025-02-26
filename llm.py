from openai import AsyncOpenAI
import asyncio
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()
VLLM_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "api_key"

# Initialize OpenAI clients for each provider

class LLM:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=VLLM_API_KEY, base_url=VLLM_URL)

    async def get_completion(
            self,
        model_name: str, user_query: str
    ) -> str:
        """
        Get a completion from a model for a given provider.

        Args:
            model_name: The name of the model to use
            provider: The provider to use
            system_prompt: The system prompt to use
            user_query: The user query to use

        Returns:
            The completion from the model
        """

        # Create the messages for the chat completion

        messages = [
            {"role": "user", "content": user_query},
        ]

        # Make the API call to get the completion
        response = await self.client.chat.completions.create(
            model=model_name, messages=messages, temperature=0.0, max_tokens=1024
        )

        # Extract and return the assistant's reply
        return response.choices[0].message.content

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