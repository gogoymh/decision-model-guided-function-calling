"""Record loop scenarios from the Python reference for ``spec/conformance/loop_edges.json``.

Each scenario keeps the exact response bodies served to the library and every request it sent, so a
port can replay the responses in order and compare its own requests. Used by ``spec/generate.py``.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import httpx

import dmcall

ROOT = Path(__file__).resolve().parent
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name."},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}},
        "required": ["city"]}}}
CLOCK = {"type": "function", "function": {
    "name": "get_time", "description": "Current local time in a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
SEARCH = {"type": "function", "function": {
    "name": "restaurant_search", "description": "Find restaurants.",
    "parameters": {"type": "object", "properties": {"location": {"type": "string"}, "rating": {"type": "number"},
                                                    "cuisine": {"type": "string"}},
                   "required": ["location"]}}}
TOOLS = [WEATHER, CLOCK, SEARCH]
MESSAGES = [{"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "Find Italian places in Seoul rated above 4.5, or tell me the weather there."}]
LENGTH = "length"
HEADERS = {"authorization", "x-api-key", "api-key", "x-goog-api-key", "anthropic-version", "content-type"}


def c(name, **arguments):
    return {"name": name, "arguments": arguments}


def openai_body(calls, model="gpt-5-nano-2025-08-07"):
    if calls == LENGTH:
        return {"model": model, "choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": ""}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 4096}}
    tool_calls = [{"id": f"call_{i}", "type": "function",
                   "function": {"name": x["name"], "arguments": json.dumps(x["arguments"], ensure_ascii=False)}}
                  for i, x in enumerate(calls)]
    message = {"role": "assistant", "content": None if tool_calls else "It is mild there today."}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"model": model, "choices": [{"finish_reason": "tool_calls" if tool_calls else "stop", "message": message}],
            "usage": {"prompt_tokens": 150, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 32}}}


def anthropic_body(calls):
    content = [{"type": "tool_use", "id": f"t{i}", "name": x["name"], "input": x["arguments"]}
               for i, x in enumerate(calls)] or [{"type": "text", "text": "It is mild there today."}]
    return {"type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
            "stop_reason": "tool_use" if calls else "end_turn", "content": content,
            "usage": {"input_tokens": 100, "cache_read_input_tokens": 20, "output_tokens": 10}}


def gemini_body(calls):
    parts = [{"functionCall": {"name": x["name"], "args": x["arguments"]}} for x in calls] or [{"text": "It is mild."}]
    return {"modelVersion": "gemini-2.5-flash-001", "candidates": [{"finishReason": "STOP",
            "content": {"role": "model", "parts": parts}}],
            "usageMetadata": {"promptTokenCount": 90, "candidatesTokenCount": 8, "thoughtsTokenCount": 4}}


def decision_body(request, good, model="jev-1.13.0", usage=True):
    """Function: right only for a function in ``good``. Arguments: right only for ``good``'s exact arguments."""
    calls = request["state"]["proposed_calls"]
    if not calls:
        function, arguments = 0.04, None
    else:
        name = calls[0]["name"]
        function = 0.96 if name in good else 0.11
        arguments = 0.91 if good.get(name) == calls[0]["arguments"] else 0.08
    answers = {"function_correct": {"type": "noul", "noul": function}}
    if "arguments_correct" in request["questions"]:
        answers["arguments_correct"] = {"type": "noul", "noul": arguments}
    body = {"model": model, "answers": answers}
    if usage:
        body["usage"] = {"input_tokens": 420, "output_tokens": 0}
    return body


GENERATORS = {"openai": openai_body, "openai-compatible": openai_body, "azure": openai_body, "xai": openai_body,
              "anthropic": anthropic_body, "gemini": gemini_body}


def make_generator(spec, client):
    kind, model, options = spec["kind"], spec["model"], dict(spec.get("options", {}))
    cls = {"openai": dmcall.OpenAI, "openai-compatible": dmcall.OpenAICompatible, "azure": dmcall.AzureOpenAI,
           "xai": dmcall.XAI, "anthropic": dmcall.Anthropic, "gemini": dmcall.Gemini}[kind]
    return cls(model, client=client, **options)


def make_decision(spec, client):
    if spec["kind"] == "pplx":
        return dmcall.PPLXDecider(spec["base_url"], client=client)
    return dmcall.Jev("JEV-KEY", client=client)


SCENARIOS = {
    "unicode_and_float_arguments": {
        "generator": {"kind": "openai", "model": "gpt-5-nano", "options": {"api_key": "KEY", "reasoning_effort": "none"}},
        "decision_model": {"kind": "jev"}, "options": {},
        "good": {"restaurant_search": {"location": "Seoul", "rating": 4.5, "cuisine": "Italian"}},
        "proposals": [[c("restaurant_search", location="서울", rating=4.0, cuisine="Italian")],
                      [c("restaurant_search", location="Seoul", rating=4.5, cuisine="Italian")]]},
    "anthropic_text_latest_function_only": {
        "generator": {"kind": "anthropic", "model": "claude-sonnet-5-5", "options": {"api_key": "KEY"}},
        "decision_model": {"kind": "jev"}, "options": {"feedback": "text", "history": "latest", "checks": "function"},
        "good": {"get_weather": {"city": "Seoul"}},
        "proposals": [[c("get_time", city="Seoul")], [c("restaurant_search", location="Seoul")],
                      [c("get_weather", city="Busan")]]},
    "gemini_numeric": {
        "generator": {"kind": "gemini", "model": "gemini-2.5-flash", "options": {"api_key": "KEY", "temperature": 0}},
        "decision_model": {"kind": "jev"}, "options": {"feedback": "numeric"},
        "good": {"get_weather": {"city": "Seoul"}},
        "proposals": [[c("get_weather", city="Busan")], [c("get_weather", city="Seoul")]]},
    "vllm_with_pplx_decider": {
        "generator": {"kind": "openai-compatible", "model": "gemma-4-31b-it",
                      "options": {"base_url": "http://localhost:8000/v1"}},
        "decision_model": {"kind": "pplx", "base_url": "http://127.0.0.1:18081"}, "options": {},
        "good": {"get_weather": {"city": "Seoul"}},
        "proposals": [[c("get_weather", city="Busan")], [], [c("get_weather", city="Seoul")]]},
    "azure_arguments_only_exhausted": {
        "generator": {"kind": "azure", "model": "gpt-5-nano",
                      "options": {"endpoint": "https://res.openai.azure.com", "api_key": "KEY"}},
        "decision_model": {"kind": "jev"}, "options": {"checks": "arguments", "max_repairs": 2},
        "good": {"get_weather": {"city": "Seoul"}},
        "proposals": [[c("get_time", city="Seoul")], [c("get_time", city="Busan")], [c("get_weather", city="Busan")]]},
    "xai_unreadable_then_schema_failure": {
        "generator": {"kind": "xai", "model": "grok-4", "options": {"api_key": "KEY"}},
        "decision_model": {"kind": "jev"}, "options": {"on_exhaustion": "last"},
        "good": {"get_weather": {"city": "Seoul"}},
        "proposals": [LENGTH, [c("get_weather")], [c("get_weather", city="Seoul")]]},
    "anthropic_initial_no_call_kept": {
        "generator": {"kind": "anthropic", "model": "claude-sonnet-5-5", "options": {"api_key": "KEY"}},
        "decision_model": {"kind": "jev"}, "options": {},
        "good": {"get_weather": {"city": "Seoul"}}, "proposals": [[]]},
}


def summary(result):
    return {"calls": result.calls, "accepted": result.accepted, "exhausted": result.exhausted,
            "repairs": result.repairs, "final_attempt": result.final_attempt, "text": result.text,
            "initial_no_call_preserved": result.initial_no_call_preserved, "attempts": result.attempts,
            "probabilities": result.probabilities, "usage": result.usage}


def record(scenario):
    kind = scenario["generator"]["kind"]
    body_of = GENERATORS[kind]
    generator_requests, generator_responses, decision_requests, decision_responses = [], [], [], []
    proposals = list(scenario["proposals"])
    pplx = scenario["decision_model"]["kind"] == "pplx"

    def generator_handler(request):
        generator_requests.append({"url": str(request.url),
                                   "headers": {k: v for k, v in request.headers.items() if k.lower() in HEADERS},
                                   "body": json.loads(request.content)})
        body = body_of(proposals.pop(0))
        generator_responses.append(body)
        return httpx.Response(200, json=body)

    def decision_handler(request):
        payload = json.loads(request.content)
        decision_requests.append({"url": str(request.url),
                                  "headers": {k: v for k, v in request.headers.items() if k.lower() in HEADERS},
                                  "body": payload})
        body = decision_body(payload, scenario["good"], model="autojev-qwen3.8-27b" if pplx else "jev-1.13.0",
                             usage=not pplx)
        decision_responses.append(body)
        return httpx.Response(200, json=body)

    generator = make_generator(scenario["generator"], httpx.Client(transport=httpx.MockTransport(generator_handler)))
    decision = make_decision(scenario["decision_model"], httpx.Client(transport=httpx.MockTransport(decision_handler)))
    agent = dmcall.DMCall(generator, decision_model=decision, tools=TOOLS, **scenario["options"])
    result = agent.call(deepcopy(MESSAGES))
    return {"generator_responses": generator_responses, "decision_responses": decision_responses,
            "generator_requests": generator_requests, "decision_requests": decision_requests,
            "result": summary(result)}


def run_scenarios() -> dict:
    scenarios = {}
    for name, scenario in SCENARIOS.items():
        scenarios[name] = {key: value for key, value in scenario.items() if key not in {"good", "proposals"}}
        scenarios[name].update(record(scenario))
    return {"note": "Replay generator_responses and decision_responses in order; the requests the port sends must "
                    "equal generator_requests and decision_requests (url, the listed headers, and the JSON body; the "
                    "history message text must be equal character for character), and the result must equal 'result'. "
                    "Generator API keys are 'KEY' and the Jev key is 'JEV-KEY'.",
            "tools": TOOLS, "messages": MESSAGES, "scenarios": scenarios}


def add_responses_to_paper_parity() -> None:
    """Store the response bodies of each paper_parity scenario so ports can replay them without builders."""
    path = ROOT / "conformance" / "paper_parity.json"
    golden = json.loads(path.read_text())
    import importlib.util
    spec = importlib.util.spec_from_file_location("parity", ROOT.parent / "tests" / "test_paper_parity.py")
    parity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parity)
    for name, scenario in golden["scenarios"].items():
        served_g, served_d, sent_g, sent_d = [], [], [], []
        respond = parity.anthropic_response if scenario["provider"] == "anthropic" else parity.chat_response
        queue = list(scenario["proposals"])

        def g(request, queue=queue, respond=respond, served_g=served_g, sent_g=sent_g):
            sent_g.append(json.loads(request.content))
            body = respond(queue.pop(0))
            served_g.append(body)
            return httpx.Response(200, json=body)

        def d(request, served_d=served_d, sent_d=sent_d):
            body = json.loads(request.content)
            sent_d.append(body)
            answer = parity.decision_answer(body)
            served_d.append(answer)
            return httpx.Response(200, json=answer)

        jev = dmcall.Jev("jev-key", client=httpx.Client(transport=httpx.MockTransport(d)))
        client = httpx.Client(transport=httpx.MockTransport(g))
        model = (dmcall.Anthropic("claude-sonnet-5-5", api_key="k", client=client) if scenario["provider"] == "anthropic"
                 else dmcall.OpenAI("gpt-5-nano", api_key="k", reasoning_effort="none", client=client))
        dmcall.DMCall(model, decision_model=jev, tools=golden["tools"], **scenario["options"]).call(deepcopy(golden["messages"]))
        assert sent_g == scenario["generator_requests"] and sent_d == scenario["decision_requests"], name
        scenario["generator_responses"], scenario["decision_responses"] = served_g, served_d
    served = []

    def v(request):
        body = json.loads(request.content)
        answer = parity.decision_answer(body)
        served.append(answer)
        return httpx.Response(200, json=answer)

    jev = dmcall.Jev("jev-key", client=httpx.Client(transport=httpx.MockTransport(v)))
    seoul = [{"name": "get_weather", "arguments": {"city": "Seoul"}}]
    dmcall.verify(golden["verify"]["messages"], seoul, golden["tools"], decision_model=jev)
    dmcall.verify("Tell me a joke.", [], golden["tools"], decision_model=jev)
    golden["verify"]["decision_responses"] = served
    golden["verify"]["calls"] = [seoul, []]
    golden["verify"]["second_messages"] = "Tell me a joke."
    golden["generator_api_key"], golden["jev_api_key"] = "k", "jev-key"
    path.write_text(json.dumps(golden, ensure_ascii=False, separators=(",", ":")))
