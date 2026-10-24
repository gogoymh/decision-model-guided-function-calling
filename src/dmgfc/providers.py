"""Function-calling models over plain HTTP (httpx only, no vendor SDKs).

Each provider turns (messages, tools, repair history) into one request and the
response into a proposal: a list of calls ``[{"name", "arguments"}]`` (``[]`` for
no call), or ``None`` when the output holds no complete proposal (cut off at the
token limit, filtered or refused). Keys come from arguments or environment
variables and never appear in errors, reprs or results.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any
from urllib.parse import quote, urlsplit

import httpx

from ._prompt import SYSTEM_PROMPT, functions_of, with_history

__all__ = ["INCOMPLETE", "PROVIDERS", "Anthropic", "AzureOpenAI", "Gemini", "OpenAI", "OpenAICompatible",
           "OpenRouter", "Provider", "ProviderError", "XAI", "provider"]

# Why an output holds no complete proposal. Such a proposal is rejected and repaired.
INCOMPLETE = {
    "length": "The response reached the output token limit before a complete proposal.",
    "content_filter": "The provider's content filter stopped the response before a complete proposal.",
    "refusal": "The response was a refusal and contains no proposal.",
    "prompt_filter": "The provider's content filter rejected the request; no response was generated.",
    "malformed_call": "The model produced a malformed function call.",
}


class ProviderError(RuntimeError):
    """A sanitized provider failure. It never contains keys or response bodies."""

    def __init__(self, message, *, status_code=None, request_id=None, attempts=None, provider_code=None):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
        self.attempts = attempts
        self.provider_code = provider_code


def _env(names):
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def _base_url(value):
    url = (value or "").strip().rstrip("/")
    parts = urlsplit(url)
    if (parts.scheme not in {"http", "https"} or not parts.netloc or parts.username is not None
            or parts.password is not None or parts.query or parts.fragment):
        return None
    return url


def _count(value):
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise ProviderError("The response contained invalid token usage.")
    return value


def _arguments(value, *, encoded):
    """Malformed arguments stay a rejected, repairable call, never a no-call."""
    if encoded:
        if not isinstance(value, str):
            raise ProviderError("Function arguments were not an encoded JSON string.")
        try:
            value = json.loads(value) if value.strip() else {}
        except ValueError:
            return None, "Function arguments are not valid JSON."
    if value is None:
        value = {}
    if not isinstance(value, dict):
        return None, "Function arguments must be a JSON object."
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return None, "Function arguments must contain finite JSON values."
    return value, None


def _append_call(calls, errors, name, arguments, *, encoded):
    if not isinstance(name, str) or not name:
        raise ProviderError("The response contained an invalid function name.")
    arguments, error = _arguments(arguments, encoded=encoded)
    calls.append({"name": name, "arguments": arguments})
    if error:
        errors.append(error)


def _incomplete(reason):
    return None, [INCOMPLETE[reason]], None, reason


def _complete(calls, errors, text):
    """A complete proposal. A response with neither a call nor text is a decision to make no call."""
    return calls, errors, "\n".join(text) if text else None, None


def _retry_delay(response, attempt):
    try:
        delay = float(response.headers.get("retry-after", 2 ** attempt))
    except ValueError:
        delay = 2 ** attempt
    return max(0.0, min(delay, 30.0)) if math.isfinite(delay) else min(2.0 ** attempt, 30.0)


class Provider:
    """Base class: build a request, send it with bounded retries, decode a proposal.

    Subclasses define ``url``, ``payload()``, ``decode()`` and usually ``auth_headers()``
    and ``usage()``. ``generate(example, history)`` is what the DMGFC loop calls; any
    object with that method can stand in for a provider.
    """

    name = "provider"
    default_base_url: str | None = None
    base_url_env: tuple[str, ...] = ()
    key_env: tuple[str, ...] = ()
    needs_key = True

    def __init__(self, model: str, *, api_key: str | None = None, base_url: str | None = None,
                 max_output_tokens: int | None = 4096, temperature: float | None = None,
                 system_prompt: str = SYSTEM_PROMPT, timeout: float = 120.0, max_retries: int = 3,
                 headers: dict[str, str] | None = None, client: httpx.Client | None = None):
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"{type(self).__name__} needs a model name.")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer.")
        if max_output_tokens is not None and (isinstance(max_output_tokens, bool)
                                              or not isinstance(max_output_tokens, int) or max_output_tokens < 1):
            raise ValueError("max_output_tokens must be a positive integer or None.")
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout)
                or timeout <= 0):
            raise ValueError("timeout must be a positive number of seconds.")
        if temperature is not None and (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                                        or not math.isfinite(temperature)):
            raise ValueError("temperature must be a finite number or None.")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be nonempty text.")
        self.model = model
        self._api_key = api_key or _env(self.key_env)
        if self.needs_key and not self._api_key:
            names = " or ".join(self.key_env) or "api_key"
            raise ValueError(f"{type(self).__name__} needs an API key: pass api_key= or set {names}.")
        if self._api_key is not None and (not isinstance(self._api_key, str) or "\n" in self._api_key
                                          or "\r" in self._api_key):
            raise ValueError("API keys must be single-line strings.")
        self.base_url = _base_url(base_url or _env(self.base_url_env) or self.default_base_url)
        if not self.base_url:
            raise ValueError(f"{type(self).__name__} needs an http(s) base_url without credentials, "
                             "query or fragment.")
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_headers = dict(headers or {})
        self.client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    # Subclass interface ---------------------------------------------------
    @property
    def url(self) -> str:
        raise NotImplementedError

    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def payload(self, messages: list[dict], tools: list[dict], history: list[dict] | None) -> dict:
        raise NotImplementedError

    def decode(self, data: dict):
        raise NotImplementedError

    def usage(self, data: dict) -> dict:
        return {}

    def filtered_request(self, status_code: int, code: str | None) -> bool:
        return False

    # Shared machinery -----------------------------------------------------
    @property
    def headers(self) -> dict[str, str]:
        return {**self.extra_headers, **self.auth_headers(), "Content-Type": "application/json"}

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self):
        return f"{type(self).__name__}(model={self.model!r}, base_url={self.base_url!r})"

    def _request_id(self, response):
        for header in ("x-request-id", "request-id", "apim-request-id", "x-goog-request-id"):
            value = response.headers.get(header, "")
            if re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", value) and not (self._api_key and self._api_key in value):
                return value
        return None

    @staticmethod
    def _error_code(response):
        try:
            error = response.json().get("error")
        except (ValueError, AttributeError):
            return None
        code = None
        if isinstance(error, dict):
            code = error.get("code") if isinstance(error.get("code"), str) else error.get("type") or error.get("status")
        return code if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", code) else None

    def _sleep(self, delay):
        started = time.perf_counter()
        time.sleep(delay)
        return time.perf_counter() - started

    def generate(self, example: dict, history: list[dict] | None = None) -> dict[str, Any]:
        """One proposal for ``example`` ({"messages", "tools"}) after the given repair history."""
        payload = self.payload(example["messages"], example.get("tools") or [], history)
        start, waited, attempts = time.perf_counter(), 0.0, 0
        for attempt in range(self.max_retries + 1):
            attempts += 1
            try:
                response = self.client.post(self.url, headers=self.headers, json=payload, timeout=self.timeout)
            except httpx.TransportError as exc:
                if attempt == self.max_retries:
                    raise ProviderError(f"{self.name} transport failed after {attempts} attempts "
                                        f"({type(exc).__name__}).", attempts=attempts) from None
                waited += self._sleep(min(2 ** attempt, 8))
                continue
            if (response.status_code in {408, 429} or response.status_code >= 500) and attempt < self.max_retries:
                waited += self._sleep(_retry_delay(response, attempt))
                continue
            break
        latency = max(0.0, time.perf_counter() - start - waited)
        data: dict = {}
        usage: dict = {}
        if not response.is_success:
            code = self._error_code(response)
            if not self.filtered_request(response.status_code, code):
                detail = f" ({code})" if code else ""
                raise ProviderError(f"{self.name} HTTP {response.status_code}{detail}; check the model, key "
                                    "and endpoint.", status_code=response.status_code,
                                    request_id=self._request_id(response), attempts=attempts, provider_code=code)
            decoded = _incomplete("prompt_filter")
        else:
            try:
                data = response.json()
                if not isinstance(data, dict):
                    raise ProviderError("The response was not a JSON object.")
                decoded = self.decode(data)
                usage = self.usage(data)
            except ProviderError as exc:
                exc.status_code, exc.request_id, exc.attempts = (response.status_code, self._request_id(response),
                                                                 attempts)
                raise
            except (KeyError, TypeError, ValueError, IndexError, AttributeError):
                raise ProviderError(f"The {self.name} response did not match its protocol.",
                                    status_code=response.status_code, request_id=self._request_id(response),
                                    attempts=attempts) from None
        calls, parse_errors, content, incomplete = decoded
        resolved = data.get("model") or data.get("modelVersion")
        return {
            "calls": calls, "content": content, "parse_errors": parse_errors, "incomplete": incomplete,
            "provider": self.name, "requested_model": self.model,
            "model": resolved if isinstance(resolved, str) else None,
            "usage": usage, "latency_s": latency, "retry_wait_s": waited, "api_calls": attempts,
        }


class OpenAICompatible(Provider):
    """Any Chat Completions endpoint with function tools: vLLM, Ollama, LM Studio, Together, Groq, ...

    ``OpenAICompatible("llama3.1", base_url="http://localhost:11434/v1")`` needs no key.
    """

    name = "openai-compatible"
    needs_key = False
    max_tokens_field = "max_completion_tokens"

    def __init__(self, model: str, *, base_url: str | None = None, api_key: str | None = None,
                 reasoning_effort: str | None = None, parallel_tool_calls: bool | None = False,
                 max_tokens_field: str | None = None, path: str = "/chat/completions", **kwargs):
        super().__init__(model, api_key=api_key, base_url=base_url, **kwargs)
        if not isinstance(path, str) or not path.startswith("/"):
            raise ValueError("path must start with '/'.")
        self.reasoning_effort = reasoning_effort
        self.parallel_tool_calls = parallel_tool_calls
        self.max_tokens_field = max_tokens_field or self.max_tokens_field
        self.path = path

    @property
    def url(self) -> str:
        return self.base_url + self.path

    def payload(self, messages, tools, history):
        payload: dict[str, Any] = {"model": self.model,
                                   "messages": [{"role": "system", "content": self.system_prompt}]
                                   + with_history(messages, history)}
        functions = functions_of(tools)
        if functions:  # a request without functions carries no tool settings
            payload["tools"] = [{"type": "function", "function": function} for function in functions]
            payload["tool_choice"] = "auto"
            if self.parallel_tool_calls is not None:
                payload["parallel_tool_calls"] = self.parallel_tool_calls
        if self.max_output_tokens:
            payload[self.max_tokens_field] = self.max_output_tokens
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        return payload

    def decode(self, data):
        if data.get("error") is not None:
            raise ProviderError("The response contained a provider error.")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ProviderError("The response did not contain a choices list.")
        choice = choices[0]
        finish = choice.get("finish_reason")
        if finish in {"length", "content_filter"}:
            return _incomplete(finish)
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("role", "assistant") != "assistant":
            raise ProviderError("The response did not contain an assistant message.")
        if message.get("refusal"):
            return _incomplete("refusal")
        calls, errors, text = [], [], []
        native = message.get("tool_calls") or []
        if not isinstance(native, list):
            raise ProviderError("The response tool_calls must be a list.")
        for item in native:
            if not isinstance(item, dict) or not isinstance(item.get("function"), dict):
                raise ProviderError("The response contained an invalid function-call object.")
            _append_call(calls, errors, item["function"].get("name"), item["function"].get("arguments", "{}"),
                         encoded=True)
        content = message.get("content")
        if isinstance(content, str) and content:
            text.append(content)
        elif content is not None and not isinstance(content, str):
            raise ProviderError("The message content was not text.")
        return _complete(calls, errors, text)

    def usage(self, data):
        raw = data.get("usage")
        if not isinstance(raw, dict):
            return {}
        details = raw.get("prompt_tokens_details")
        return {key: value for key, value in {
            "input_tokens": _count(raw.get("prompt_tokens")),
            "cached_input_tokens": _count(details.get("cached_tokens")) if isinstance(details, dict) else None,
            "output_tokens": _count(raw.get("completion_tokens")),
        }.items() if value is not None}

    def filtered_request(self, status_code, code):
        return status_code == 400 and code in {"content_filter", "invalid_prompt"}


class OpenAI(OpenAICompatible):
    """OpenAI Chat Completions. Key: OPENAI_API_KEY (base URL: OPENAI_BASE_URL, optional)."""

    name = "openai"
    default_base_url = "https://api.openai.com/v1"
    base_url_env = ("OPENAI_BASE_URL",)
    key_env = ("OPENAI_API_KEY",)
    needs_key = True


class XAI(OpenAICompatible):
    """xAI Grok, OpenAI-compatible. Key: XAI_API_KEY."""

    name = "xai"
    default_base_url = "https://api.x.ai/v1"
    key_env = ("XAI_API_KEY",)
    needs_key = True
    max_tokens_field = "max_tokens"


class OpenRouter(OpenAICompatible):
    """OpenRouter, one OpenAI-compatible endpoint for many models. Key: OPENROUTER_API_KEY."""

    name = "openrouter"
    default_base_url = "https://openrouter.ai/api/v1"
    key_env = ("OPENROUTER_API_KEY",)
    needs_key = True
    max_tokens_field = "max_tokens"


class AzureOpenAI(OpenAICompatible):
    """An Azure OpenAI deployment, directly or behind an API Management gateway.

    Key: AZURE_OPENAI_API_KEY, endpoint: AZURE_OPENAI_ENDPOINT, version: AZURE_OPENAI_API_VERSION.
    """

    name = "azure"
    key_env = ("AZURE_OPENAI_API_KEY", "AZURE_API_KEY")
    needs_key = True

    def __init__(self, deployment: str, *, endpoint: str | None = None, api_version: str | None = None,
                 api_key: str | None = None, **kwargs):
        endpoint = endpoint or _env(("AZURE_OPENAI_ENDPOINT", "AZURE_ENDPOINT"))
        if not endpoint:
            raise ValueError("AzureOpenAI needs endpoint= or AZURE_OPENAI_ENDPOINT.")
        self.api_version = api_version or _env(("AZURE_OPENAI_API_VERSION",)) or "2024-12-01-preview"
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:-preview)?", self.api_version):
            raise ValueError("api_version must look like 2024-12-01-preview.")
        if not isinstance(deployment, str) or deployment in {".", ".."} or any(ord(char) < 32 for char in deployment):
            raise ValueError("The deployment name contains invalid characters.")
        super().__init__(deployment, base_url=endpoint, api_key=api_key, **kwargs)

    @property
    def url(self):
        return (f"{self.base_url}/openai/deployments/{quote(self.model, safe='')}/chat/completions"
                f"?api-version={self.api_version}")

    def auth_headers(self):
        return {"api-key": self._api_key}


class Anthropic(Provider):
    """Anthropic Messages API. Key: ANTHROPIC_API_KEY (base URL: ANTHROPIC_BASE_URL, optional)."""

    name = "anthropic"
    default_base_url = "https://api.anthropic.com/v1"
    base_url_env = ("ANTHROPIC_BASE_URL",)
    key_env = ("ANTHROPIC_API_KEY",)
    version = "2023-06-01"

    @property
    def url(self):
        return self.base_url + "/messages"

    def auth_headers(self):
        return {"x-api-key": self._api_key, "anthropic-version": self.version}

    def payload(self, messages, tools, history):
        system, conversation = [self.system_prompt], []
        for message in with_history(messages, history):
            if message["role"] in {"system", "developer"}:
                if not isinstance(message["content"], str):
                    raise ValueError("Anthropic needs text system messages.")
                system.append(message["content"])
            elif message["role"] in {"user", "assistant"}:
                conversation.append(message)
            else:
                raise ValueError("Anthropic accepts only user and assistant conversation messages.")
        payload: dict[str, Any] = {"model": self.model, "system": "\n\n".join(system), "messages": conversation,
                                   "max_tokens": self.max_output_tokens or 4096}
        functions = functions_of(tools)
        if functions:
            payload["tools"] = [{"name": f["name"], "description": f.get("description", ""),
                                 "input_schema": f.get("parameters", {"type": "object", "properties": {}})}
                                for f in functions]
            payload["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        return payload

    def decode(self, data):
        if data.get("type") == "error" or data.get("error") is not None:
            raise ProviderError("The response contained a provider error.")
        if data.get("role") != "assistant" or not isinstance(data.get("content"), list):
            raise ProviderError("The Anthropic response was not an assistant message.")
        details = data.get("stop_details")
        reason = data.get("stop_reason")
        if reason in {"max_tokens", "model_context_window_exceeded"}:
            return _incomplete("length")
        if reason == "refusal" or (isinstance(details, dict) and details.get("type") == "refusal"):
            return _incomplete("refusal")
        calls, errors, text = [], [], []
        for block in data["content"]:
            if not isinstance(block, dict):
                raise ProviderError("An Anthropic content block was malformed.")
            if block.get("type") == "tool_use":
                _append_call(calls, errors, block.get("name"), block.get("input"), encoded=False)
            elif block.get("type") == "text" and isinstance(block.get("text"), str) and block["text"]:
                text.append(block["text"])
        return _complete(calls, errors, text)

    def usage(self, data):
        raw = data.get("usage")
        if not isinstance(raw, dict):
            return {}
        read = _count(raw.get("cache_read_input_tokens")) or 0
        written = _count(raw.get("cache_creation_input_tokens")) or 0
        base = _count(raw.get("input_tokens"))
        result = {"cached_input_tokens": read, "output_tokens": _count(raw.get("output_tokens"))}
        if base is not None:
            result["input_tokens"] = base + read + written
        return {key: value for key, value in result.items() if value is not None}


class Gemini(Provider):
    """Google Gemini API (generateContent with function declarations). Key: GEMINI_API_KEY or GOOGLE_API_KEY."""

    name = "gemini"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"
    key_env = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    FILTERED = frozenset({"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY",
                          "LANGUAGE"})

    def __init__(self, model: str, *, thinking_budget: int | None = None, **kwargs):
        super().__init__(model.removeprefix("models/") if isinstance(model, str) else model, **kwargs)
        self.thinking_budget = thinking_budget

    @property
    def url(self):
        return f"{self.base_url}/models/{quote(self.model, safe='')}:generateContent"

    def auth_headers(self):
        return {"x-goog-api-key": self._api_key}

    @staticmethod
    def _text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):  # OpenAI-style text parts
            return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
        raise ValueError("Gemini messages need text content.")

    def payload(self, messages, tools, history):
        system, contents = [self.system_prompt], []
        for message in with_history(messages, history):
            role = message["role"]
            if role in {"system", "developer"}:
                system.append(self._text(message["content"]))
            elif role in {"user", "assistant"}:
                contents.append({"role": "model" if role == "assistant" else "user",
                                 "parts": [{"text": self._text(message["content"])}]})
            else:
                raise ValueError("Gemini accepts only user and assistant conversation messages.")
        payload: dict[str, Any] = {"systemInstruction": {"parts": [{"text": "\n\n".join(system)}]},
                                   "contents": contents}
        functions = functions_of(tools)
        if functions:
            payload["tools"] = [{"functionDeclarations": [
                {"name": f["name"], "description": f.get("description", ""),
                 "parametersJsonSchema": f.get("parameters", {"type": "object", "properties": {}})}
                for f in functions]}]
            payload["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
        config: dict[str, Any] = {}
        if self.max_output_tokens:
            config["maxOutputTokens"] = self.max_output_tokens
        if self.temperature is not None:
            config["temperature"] = self.temperature
        if self.thinking_budget is not None:
            config["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}
        if config:
            payload["generationConfig"] = config
        return payload

    def decode(self, data):
        if data.get("error") is not None:
            raise ProviderError("The response contained a provider error.")
        candidates = data.get("candidates")
        if not candidates:
            if isinstance(data.get("promptFeedback"), dict) and data["promptFeedback"].get("blockReason"):
                return _incomplete("prompt_filter")
            raise ProviderError("The Gemini response had no candidates.")
        candidate = candidates[0]
        reason = candidate.get("finishReason")
        if reason == "MAX_TOKENS":
            return _incomplete("length")
        if reason in self.FILTERED:
            return _incomplete("content_filter")
        if reason in {"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"}:
            return _incomplete("malformed_call")
        parts = (candidate.get("content") or {}).get("parts") or []
        calls, errors, text = [], [], []
        for part in parts:
            if not isinstance(part, dict):
                raise ProviderError("A Gemini content part was malformed.")
            if isinstance(part.get("functionCall"), dict):
                call = part["functionCall"]
                _append_call(calls, errors, call.get("name"), call.get("args"), encoded=False)
            elif isinstance(part.get("text"), str) and part["text"] and not part.get("thought"):
                text.append(part["text"])
        return _complete(calls, errors, text)

    def usage(self, data):
        raw = data.get("usageMetadata")
        if not isinstance(raw, dict):
            return {}
        output = _count(raw.get("candidatesTokenCount"))
        thoughts = _count(raw.get("thoughtsTokenCount")) or 0
        return {key: value for key, value in {
            "input_tokens": _count(raw.get("promptTokenCount")),
            "cached_input_tokens": _count(raw.get("cachedContentTokenCount")),
            "output_tokens": None if output is None and not thoughts else (output or 0) + thoughts,
        }.items() if value is not None}


PROVIDERS = {"openai": OpenAI, "azure": AzureOpenAI, "anthropic": Anthropic, "claude": Anthropic,
             "gemini": Gemini, "google": Gemini, "xai": XAI, "grok": XAI, "openrouter": OpenRouter}


def provider(spec, **options) -> Provider:
    """A provider from ``"provider:model"``, e.g. ``"openai:gpt-5-nano"`` or ``"azure:<deployment>"``.

    Keyword options go to the provider (api_key=, base_url=, endpoint=, ...). A provider
    instance, or any object with ``generate(example, history)``, passes through unchanged.
    """
    if not isinstance(spec, str):
        if callable(getattr(spec, "generate", None)):
            if options:
                raise ValueError("Pass options to the provider itself, not with an instance.")
            return spec
        raise TypeError('A model is a "provider:model" string or a provider object.')
    name, separator, model = spec.partition(":")
    if not separator or not model:
        raise ValueError('Use "provider:model", e.g. "openai:gpt-5-nano", "anthropic:claude-sonnet-4-5", '
                         '"gemini:gemini-2.5-flash", "xai:grok-4" or "azure:<deployment>".')
    if name.lower() not in PROVIDERS:
        raise ValueError(f"Unknown provider {name!r}; choose from {', '.join(sorted(PROVIDERS))}, or pass "
                         "dmgfc.OpenAICompatible(model, base_url=...) for any other endpoint.")
    return PROVIDERS[name.lower()](model, **options)
