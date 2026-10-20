"""Compatibility facade for the former ``jbfc`` package.

Use ``import dmfc`` and ``dmfc.DMFC`` for new code. ``JBFC`` is an alias of that
same class; old method strings continue to select the same frozen repair loop.
"""

from dmfc import (DMFC, METHODS, PROVIDERS, SYSTEM_PROMPT, XAI, Anthropic, AzureOpenAI, Gemini, Jev,
                  JevError, OpenAI, OpenAICompatible, OpenRouter, Provider, ProviderError, Result, Verdict,
                  __version__, invoke, normalize_calls, provider, schema_of, tool, toolset, verify)

JBFC = DMFC

__all__ = [
    "DMFC", "JBFC", "METHODS", "PROVIDERS", "SYSTEM_PROMPT", "XAI", "Anthropic", "AzureOpenAI", "Gemini", "Jev",
    "JevError", "OpenAI", "OpenAICompatible", "OpenRouter", "Provider", "ProviderError", "Result", "Verdict",
    "__version__", "invoke", "normalize_calls", "provider", "schema_of", "tool", "toolset", "verify",
]
