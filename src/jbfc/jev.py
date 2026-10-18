"""Jev, TypeSafe's verifier model, as JBFC uses it: a thin layer over ``jev_fc.jev``.

Jev answers two yes/no questions about a proposed call before it runs, each as a
probability of yes: is this the right function (or is making no call right), and
are the arguments correct and complete? A check passes at or above its threshold.
The argument question is skipped for a no-call decision.
"""

from __future__ import annotations

import os

import httpx
from jev_fc.jev import ARGUMENT_QUESTION, FUNCTION_QUESTION, JevAPIError, JevVerifier, validate_calls

__all__ = ["ARGUMENT_QUESTION", "FUNCTION_QUESTION", "Jev", "JevError", "validate_calls"]

JevError = JevAPIError


def _env(*names):
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


class Jev(JevVerifier):
    """The Jev verifier. Key: ``api_key=`` or TYPESAFE_API_KEY (JEV_API_KEY also works).

    ``threshold`` sets both checks (0.5, as in the paper); ``function_threshold`` and
    ``arguments_threshold`` override one of them. ``gateway="apim"`` routes through an Azure API Management
    gateway with APIM_SUBSCRIPTION_KEY instead.
    """

    def __init__(self, api_key: str | None = None, *, threshold: float = 0.5,
                 function_threshold: float | None = None, arguments_threshold: float | None = None,
                 model: str | None = None, base_url: str | None = None, gateway: str = "direct",
                 timeout: float = 60.0, max_retries: int = 3, client: httpx.Client | None = None):
        if gateway == "direct":
            api_key = api_key or _env("TYPESAFE_API_KEY", "JEV_API_KEY")
            if not api_key:
                raise ValueError("Jev needs an API key: pass api_key= or set TYPESAFE_API_KEY.")
        super().__init__(
            api_key, model=model,
            function_threshold=threshold if function_threshold is None else function_threshold,
            arguments_threshold=threshold if arguments_threshold is None else arguments_threshold,
            base_url=base_url, gateway=gateway, timeout=timeout, max_retries=max_retries, client=client,
        )

    def __repr__(self):
        return (f"Jev(model={self.model!r}, function_threshold={self.function_threshold}, "
                f"arguments_threshold={self.arguments_threshold})")
