"""DM-Call: Decision-Model-Guided Function Calling with Cumulative Verification Feedback.

An LLM proposes a function call; a decision model checks whether the function and the
arguments are right before anything runs; a rejected call goes back to the same LLM with every
earlier candidate and its feedback, until a candidate passes or the repair limit is reached.

    import dmcall

    def get_weather(city: str, unit: str = "celsius") -> dict:
        \"\"\"Current weather for a city.\"\"\"
        ...

    agent = dmcall.DMCall("openai:gpt-5-nano", tools=[get_weather])  # OPENAI_API_KEY, TYPESAFE_API_KEY
    result = agent.run("Is it warm in Seoul right now?")
    result.call, result.accepted, result.output
"""

from .core import CHECKS, HISTORIES, ON_EXHAUSTION, DMCall, Result, Verdict, verify
from .decision import DecisionModel, DecisionModelError, Jev, PPLXDecider
from .feedback import FEEDBACK, label
from .prompts import ARGUMENT_QUESTION, FUNCTION_QUESTION, SYSTEM_PROMPT
from .providers import (PROVIDERS, XAI, Anthropic, AzureOpenAI, Gemini, Meta, OpenAI, OpenAICompatible, OpenRouter,
                        Provider, ProviderError, provider)
from .schema import check_schema
from .tools import invoke, normalize_calls, schema_of, tool, toolset

__version__ = "1.0.0"

__all__ = [
    "ARGUMENT_QUESTION", "CHECKS", "DMCall", "DecisionModel", "DecisionModelError", "FEEDBACK", "FUNCTION_QUESTION",
    "HISTORIES", "Jev", "ON_EXHAUSTION", "PPLXDecider", "PROVIDERS", "Result", "SYSTEM_PROMPT", "Verdict",
    "Anthropic", "AzureOpenAI", "Gemini", "Meta", "OpenAI", "OpenAICompatible", "OpenRouter", "Provider",
    "ProviderError", "XAI", "__version__", "check_schema", "invoke", "label", "normalize_calls", "provider", "schema_of", "tool",
    "toolset", "verify",
]
