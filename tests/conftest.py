"""Shared test helpers. Every test is offline: HTTP goes through httpx.MockTransport."""

import json

import httpx
import pytest

import dmcall
from dmcall import providers as providers_module

KEY_NAMES = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "GEMINI_API_KEY",
             "GOOGLE_API_KEY", "XAI_API_KEY", "OPENROUTER_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_API_KEY",
             "AZURE_OPENAI_ENDPOINT", "AZURE_ENDPOINT", "AZURE_OPENAI_API_VERSION", "TYPESAFE_API_KEY",
             "TYPESAFE_BASE_URL", "TYPESAFE_MODEL", "MODEL_API_KEY")
SECRET = "sk-test-SECRET-123"

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                   "required": ["city"]}}}
EXAMPLE = {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather in Seoul?"}],
           "tools": [WEATHER]}
HISTORY = [{"calls": [{"name": "get_weather", "arguments": {"city": "Busan"}}],
            "feedback": {"failed_checks": ["arguments"], "instruction": "Reconsider."}},
           {"round": 2, "calls": [], "feedback": {"failed_checks": ["function_selection"]}}]


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for name in KEY_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(providers_module.Provider, "_sleep", lambda self, delay: 0.0)


def client(handler, seen=None):
    def wrapped(request):
        if seen is not None:
            seen.append(request)
        return handler(request)
    return httpx.Client(transport=httpx.MockTransport(wrapped))


def chat(calls=(), content=None, finish=None, usage=None, **message):
    """A Chat Completions response body."""
    tool_calls = [{"id": f"call_{i}", "type": "function",
                   "function": {"name": name, "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments)}}
                  for i, (name, arguments) in enumerate(calls)]
    body = {"model": "resolved-model", "choices": [{
        "finish_reason": finish or ("tool_calls" if tool_calls else "stop"),
        "message": {"role": "assistant", "content": content, **({"tool_calls": tool_calls} if tool_calls else {}),
                    **message}}]}
    if usage is not None:
        body["usage"] = usage
    return body


def answer(function=0.95, arguments=0.95, model="jev-1.13.0"):
    """A /v1/systemone response body."""
    answers = {"function_correct": {"type": "noul", "noul": function}}
    if arguments is not None:
        answers["arguments_correct"] = {"type": "noul", "noul": arguments}
    return {"model": model, "answers": answers, "usage": {"input_tokens": 300, "output_tokens": 0}}


def seoul():
    return [{"name": "get_weather", "arguments": {"city": "Seoul"}}]


def busan():
    return [{"name": "get_weather", "arguments": {"city": "Busan"}}]


class ScriptedModel:
    """A provider stand-in that returns scripted candidates and records the history it saw."""

    def __init__(self, *proposals):
        self.proposals = list(proposals)
        self.histories = []
        self.closed = False

    def generate(self, example, history=None):
        self.histories.append(json.loads(json.dumps(history or [])))
        calls = self.proposals.pop(0)
        incomplete = "length" if calls is None else None
        return {"calls": calls, "content": None if calls else "No tool fits.", "incomplete": incomplete,
                "parse_errors": ["cut off"] if calls is None else [],
                "usage": {"input_tokens": 10, "output_tokens": 2}, "latency_s": 0.01}

    def close(self):
        self.closed = True


class ScriptedJev:
    """Jev over mocked HTTP: arguments are right only for ``good_city``; a no-call scores ``no_call``."""

    def __init__(self, good_city="Seoul", function=0.95, no_call=0.05, **options):
        self.good_city, self.function, self.no_call, self.requests = good_city, function, no_call, []
        self.jev = dmcall.Jev("jev-key", client=client(self.reply), **options)

    def reply(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        calls = body["state"]["proposed_calls"]
        if not calls:
            return httpx.Response(200, json=answer(self.no_call, None))
        arguments = 0.95 if calls[0]["arguments"].get("city") == self.good_city else 0.05
        return httpx.Response(200, json=answer(self.function, arguments))
