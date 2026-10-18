"""JBFC: Jev-backed function calling.

A model proposes a function call, Jev checks the function and its arguments before
anything runs, and the same model repairs the call with the accumulated feedback.

    import jbfc

    def get_weather(city: str, unit: str = "celsius") -> dict:
        \"\"\"Current weather for a city.\"\"\"
        ...

    agent = jbfc.JBFC("openai:gpt-5-nano", tools=[get_weather])  # OPENAI_API_KEY, TYPESAFE_API_KEY
    result = agent.run("Is it warm in Seoul right now?")
    result.call, result.accepted, result.output
"""

from ._prompt import SYSTEM_PROMPT
from .core import JBFC, METHODS, Result, Verdict, verify
from .jev import Jev, JevError
from .providers import (PROVIDERS, XAI, Anthropic, AzureOpenAI, Gemini, OpenAI, OpenAICompatible, OpenRouter,
                        Provider, ProviderError, provider)
from .tools import invoke, normalize_calls, schema_of, tool, toolset

__version__ = "0.1.0"

__all__ = [
    "JBFC", "METHODS", "PROVIDERS", "SYSTEM_PROMPT", "XAI", "Anthropic", "AzureOpenAI", "Gemini", "Jev",
    "JevError", "OpenAI", "OpenAICompatible", "OpenRouter", "Provider", "ProviderError", "Result", "Verdict",
    "__version__", "invoke", "normalize_calls", "provider", "schema_of", "tool", "toolset", "verify",
]
