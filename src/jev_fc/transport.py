"""Provider routing and authentication, with APIM credentials kept off disk."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit


DEFAULT_APIM_BASE_URL = "https://enhans-aoai-prod-apim.azure-api.net"


@dataclass(frozen=True)
class APIConfig:
    gateway: str
    provider: str
    base_url: str
    headers: dict[str, str] = field(repr=False)


def gateway_mode(value: str | None = None) -> str:
    mode = (value if value is not None else os.getenv("API_GATEWAY", "direct")).strip().lower()
    if mode not in {"direct", "apim"}:
        raise ValueError("API_GATEWAY must be direct or apim.")
    return mode


def _base_url(value: str) -> str:
    url = value.strip().rstrip("/")
    parsed = urlsplit(url)
    if (parsed.scheme not in {"http", "https"} or not parsed.netloc
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise ValueError("API base URL must be an HTTP(S) URL without credentials, query, or fragment.")
    return url


def resolve_api_config(
    provider: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    gateway: str | None = None,
    azure_chat_completions: bool = False,
) -> APIConfig:
    if provider not in {"openai", "anthropic", "typesafe"}:
        raise ValueError("Unknown API provider.")
    mode = gateway_mode(gateway)
    if azure_chat_completions and (provider != "openai" or mode != "apim"):
        raise ValueError("Azure Chat Completions requires the OpenAI provider and API_GATEWAY=apim.")
    if mode == "apim":
        key = api_key or os.getenv("APIM_SUBSCRIPTION_KEY")
        if not key:
            raise ValueError("Set APIM_SUBSCRIPTION_KEY in .env before running through APIM.")
        if azure_chat_completions:
            url = _base_url(base_url or os.getenv("AZURE_ENDPOINT") or os.getenv("APIM_BASE_URL") or DEFAULT_APIM_BASE_URL)
        else:
            root = _base_url(os.getenv("APIM_BASE_URL") or DEFAULT_APIM_BASE_URL)
            suffix = "/llm/typesafe" if provider == "typesafe" else f"/llm/{provider}/v1"
            url = _base_url(base_url) if base_url else root + suffix
        headers = {"api-key": key}
    else:
        if provider == "typesafe":
            key = api_key or os.getenv("TYPESAFE_API_KEY")
            url = base_url or os.getenv("TYPESAFE_BASE_URL") or "https://api.typesafe.ai"
            missing = "Set TYPESAFE_API_KEY in .env before running direct Jev experiments."
        else:
            provider_key = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
            key = api_key or os.getenv("LLM_API_KEY") or os.getenv(provider_key)
            default = "https://api.anthropic.com/v1" if provider == "anthropic" else "https://api.openai.com/v1"
            url = base_url or os.getenv("LLM_BASE_URL") or default
            missing = "Set LLM_API_KEY in .env before running direct generator experiments."
        if not key:
            raise ValueError(missing)
        url = _base_url(url)
        headers = ({"x-api-key": key, "anthropic-version": "2023-06-01"}
                   if provider == "anthropic" else {"Authorization": "Bearer " + key})
    if not isinstance(key, str) or not key.strip() or "\r" in key or "\n" in key:
        raise ValueError("API credential must be a nonempty single-line string.")
    headers["Content-Type"] = "application/json"
    return APIConfig(mode, provider, url, headers)
