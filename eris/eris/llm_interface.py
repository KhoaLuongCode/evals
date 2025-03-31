import logging
from abc import ABC, abstractmethod
from typing import Optional
import requests
from dotenv import load_dotenv
load_dotenv()

class LLMInterface(ABC):
    """
    Abstract class that defines how to interface with an LLM.
    Subclasses must implement 'generate_response' to return a string
    based on a prompt.
    """
    def __init__(self, name: str, api_key: str, model: str):
        self.name = name
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Generates a response to the given prompt.
        """
        pass


class OpenAIInterface(LLMInterface):
    def __init__(self, name: str, api_key: str, model: str):
        super().__init__(name, api_key, model)
        try:
            from openai import OpenAI
            client = OpenAI(api_key = api_key)
            self.client = client
        except ImportError:
            raise ImportError("Please install the openai library to use OpenAIInterface.")

    def generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"OpenAI request failed: {e}")
            return "Error: Could not generate response."


class AnthropicInterface(LLMInterface):
    """
    Example implementation using Anthropic's Claude API.
    """
    def __init__(self, name: str, api_key: str, model: str):
        super().__init__(name, api_key, model)
        try:
            from anthropic import Anthropic
            client = Anthropic(
                api_key = api_key
            )
            self.client = client
        except ImportError:
            raise ImportError("Please install the anthropic library to use AnthropicInterface.")

    def generate_response(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        try:
            response = self.client.messages.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model,
                max_tokens=max_tokens,
            )
            return "\n".join([block.text if hasattr(block, "text") else str(block) for block in response.content])
        except Exception as e:
            logging.error(f"Anthropic request failed: {e}")
            return "Error: Could not generate response."
