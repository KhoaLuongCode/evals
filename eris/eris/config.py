from typing import NamedTuple
import os

class LLMConfig(NamedTuple):
    name: str
    api_key: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 1024

OPENAI_DEFAULT = LLMConfig(
    name="OpenAI-GPT4",
    api_key=os.getenv("OPENAI_API_KEY"), 
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=1024
)

ANTHROPIC_DEFAULT = LLMConfig(
    name="Anthropic-Claude",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    model="claude-3-haiku-20240307",
    temperature=0.7,
    max_tokens=1024
)

