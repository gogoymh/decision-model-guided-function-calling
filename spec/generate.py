"""Regenerate the language-neutral spec files from the Python reference implementation.

    uv run python spec/generate.py

Writes ``spec/prompts.json`` and every file in ``spec/conformance/`` except ``paper_parity.json``,
which was recorded once from the code behind the paper's tables and must not be regenerated.
Every port of DM-Call (TypeScript, Go, Rust, Java, ...) reads these files in its tests.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import httpx

import dmcall
from dmcall import feedback as fb
from dmcall import prompts

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "conformance"

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name."},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}},
        "required": ["city"]}}}
CLOCK = {"type": "function", "function": {
    "name": "get_time", "description": "Current local time in a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
TOOLS = [WEATHER, CLOCK]
MESSAGES = [{"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "Is it warm in Seoul right now?"}]
HISTORY = [{"calls": [{"name": "get_weather", "arguments": {"city": "Busan"}}],
            "feedback": {"failed_checks": ["arguments"], "instruction": "Reconsider."}},
           {"round": 2, "calls": [], "feedback": {"failed_checks": ["function_selection"]}}]


def write(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")


def mock(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# prompts.json --------------------------------------------------------------------------------
def prompts_file() -> dict:
    return {
        "system_prompt": prompts.SYSTEM_PROMPT,
        "history_header": prompts.HISTORY_HEADER,
        "history_footer": prompts.HISTORY_FOOTER,
        "function_question": prompts.FUNCTION_QUESTION,
        "argument_question": prompts.ARGUMENT_QUESTION,
        "repair_instruction": prompts.REPAIR_INSTRUCTION,
        "labels": list(fb.LABELS),
        "cut_points": list(fb.CUT_POINTS),
        "label_fields": dict(fb._LABEL_FIELDS),
        "threshold_fields": sorted(fb._THRESHOLDS),
        "defaults": {"threshold": 0.5, "max_repairs": 3, "checks": "both", "history": "cumulative",
                     "feedback": "hybrid", "on_exhaustion": "first", "preserve_initial_no_call": True,
                     "execute_unverified": False, "max_output_tokens": 4096},
        "jev": {"model": dmcall.Jev.DEFAULT_MODEL, "base_url": dmcall.Jev.DEFAULT_BASE_URL, "path": "/v1/systemone",
                "feedback_source": dmcall.Jev.feedback_source, "key_env": "TYPESAFE_API_KEY",
                "model_env": "TYPESAFE_MODEL", "base_url_env": "TYPESAFE_BASE_URL"},
        "pplx_decider": {"model_id": dmcall.PPLXDecider.MODEL_ID, "revision": dmcall.PPLXDecider.MODEL_REVISION,
                         "calibration_temperature": dmcall.PPLXDecider.CALIBRATION_TEMPERATURE,
                         "max_input_tokens": dmcall.PPLXDecider.MAX_INPUT_TOKENS,
                         "server_model": dmcall.PPLXDecider.SERVER_MODEL, "wire_model": dmcall.PPLXDecider.WIRE_MODEL,
                         "feedback_source": dmcall.PPLXDecider.feedback_source, "timeout_s": 180, "max_retries": 0},
        "retry_statuses": sorted({408, 429, 500, 502, 503, 504, 529}),
        "incomplete_reasons": dict(dmcall.providers.INCOMPLETE),
    }


# json_text.json: how the history message serializes values --------------------------------
JSON_INPUTS = [
    "0", "1", "-7", "100000000000000000000", "1.0", "0.5", "0.95", "-0.0", "0.1", "1e16", "1.5e16", "1e-05",
    "0.0001", "0.00001234", "123456789.125", "1e22", "3.0e2", "0.30000000000000004", "true", "false", "null",
    '""', '"Seoul"', '"서울 날씨"', '"quote \\" backslash \\\\ slash /"', '"tab\\tnewline\\nreturn\\r"',
    '"\\u0001\\u001f\\u007f"', '"emoji 🌤"', '"\\u2028\\u2029"', "[]", "{}", "[1, 2.0, \"a\", null, true]",
    '{"b": 1, "a": {"z": [], "y": {}}, "10": "x", "2": "y"}',
    '{"rating": 4.0, "amount": 10800, "ratio": 0.57, "nested": [{"k": 1e-7}]}',
    # Ties between two shortest round-trip strings resolve to the nearest, half to even.
    "834400034008635.25", "5e-324", "1.7976931348623157e308", "2.5e-05", "9007199254740993.0", "0.1e1",
]


def json_text_file() -> dict:
    return {"rule": "json.dumps(value, ensure_ascii=False) of Python's json.loads(input): separators ', ' and ': ', "
                    "insertion key order, floats in Python repr, non-ASCII kept, control characters escaped.",
            "cases": [{"input": text, "output": json.dumps(json.loads(text), ensure_ascii=False)} for text in JSON_INPUTS]}


# schema_check.json -----------------------------------------------------------------------------
BOOK = {"type": "function", "function": {"name": "book", "description": "", "parameters": {
    "type": "object",
    "properties": {
        "genre": {"type": "string", "enum": ["Horror", "Comedy"]},
        "rooms": {"type": "integer", "minimum": 1},
        "metrics": {"type": "array", "enum": ["view", "trust"], "items": {"type": "string"}},
        "start": {"type": "string", "default": None},
        "end": {"type": "string"},
        "filters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        "tags": {"type": "array", "items": {"type": "string"}},
        "pair": {"type": "array", "prefixItems": [{"type": "integer"}, {"type": "string"}]},
        "price": {"type": "number"},
        "flag": {"type": "boolean"},
        "either": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
        "one": {"oneOf": [{"type": "string", "enum": ["a"]}, {"type": "string", "maxLength": 1}]},
        "maybe": {"type": ["string", "null"]},
        "ref": {"$ref": "#/$defs/point"},
    },
    "$defs": {"point": {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"],
                        "additionalProperties": False}},
    "required": ["genre"],
}}}
CLOSED = {"type": "function", "function": {"name": "closed", "description": "", "parameters": {
    "type": "object", "additionalProperties": False, "properties": {"q": {"type": "string"}}, "required": ["q"]}}}
OPEN = {"type": "function", "function": {"name": "open", "description": "", "parameters": {
    "type": "object", "additionalProperties": True, "properties": {"q": {"type": "string"}}}}}
TYPED_EXTRA = {"type": "function", "function": {"name": "typed_extra", "description": "", "parameters": {
    "type": "object", "properties": {"q": {"type": "string"}}, "additionalProperties": {"type": "integer"}}}}
NO_PARAMS = {"type": "function", "function": {"name": "ping", "description": "Ping."}}
# A false subschema reports its error at the parent path (jsonschema's behavior).
FALSE_SUB = {"type": "function", "function": {"name": "f", "description": "", "parameters": {
    "type": "object", "properties": {"a": False, "b": {"type": "array", "items": False}}}}}


def call(name, arguments):
    return {"name": name, "arguments": arguments}


SCHEMA_CASES = [
    ("valid call", [WEATHER], [call("get_weather", {"city": "Seoul"})]),
    ("no call", [WEATHER], []),
    ("missing required", [WEATHER], [call("get_weather", {})]),
    ("wrong type", [WEATHER], [call("get_weather", {"city": 3})]),
    ("unknown parameter", [WEATHER], [call("get_weather", {"city": "Seoul", "invented": 1, "also": True})]),
    ("enum not enforced", [WEATHER], [call("get_weather", {"city": "Seoul", "unit": "kelvin"})]),
    ("unknown function", [WEATHER], [call("invented", {})]),
    ("two calls", [WEATHER, CLOCK], [call("get_weather", {"city": "Seoul"}), call("get_time", {"city": "Seoul"})]),
    ("not a list", [WEATHER], None),
    ("call not an object", [WEATHER], ["get_weather"]),
    ("empty name", [WEATHER], [{"name": "", "arguments": {}}]),
    ("missing name", [WEATHER], [{"arguments": {}}]),
    ("arguments not an object", [WEATHER], [{"name": "get_weather", "arguments": "{}"}]),
    ("arguments null", [WEATHER], [{"name": "get_weather", "arguments": None}]),
    ("book valid", [BOOK], [call("book", {"genre": "Any", "rooms": 0, "metrics": ["view"], "start": None,
                                          "filters": {"city": "Seoul"}, "tags": ["a"], "pair": [1, "x"], "price": 2,
                                          "flag": False, "either": 3, "one": "abc", "maybe": None,
                                          "ref": {"x": 1.5}})]),
    ("book many errors", [BOOK], [call("book", {"genre": True, "rooms": "2", "metrics": "view", "end": None,
                                                "filters": {}, "tags": ["a", 2], "pair": ["x", 1], "price": "1",
                                                "flag": 1, "either": 1.5, "one": 3, "maybe": 1,
                                                "ref": {"y": 1}, "extra": [1, {"a": None}]})]),
    ("book missing required only", [BOOK], [call("book", {"rooms": 1})]),
    ("integer-valued float for integer", [BOOK], [call("book", {"genre": "x", "rooms": 2.0})]),
    ("closed object", [CLOSED], [call("closed", {"q": "x", "z": 1, "a": 2})]),
    ("open object", [OPEN], [call("open", {"q": "x", "z": 1})]),
    ("typed additional properties", [TYPED_EXTRA], [call("typed_extra", {"q": "x", "n": 1, "s": "no"})]),
    ("no parameters declared", [NO_PARAMS], [call("ping", {})]),
    ("no parameters declared with argument", [NO_PARAMS], [call("ping", {"x": 1})]),
    ("nested error path", [BOOK], [call("book", {"genre": "x", "filters": {"city": ["Seoul"]}})]),
    ("false subschema property", [FALSE_SUB], [call("f", {"a": 1})]),
    ("false items subschema", [FALSE_SUB], [call("f", {"b": [1]})]),
    ("false items subschema, empty array", [FALSE_SUB], [call("f", {"b": []})]),
]


def schema_check_file() -> dict:
    cases = []
    for name, tools, calls in SCHEMA_CASES:
        cases.append({"name": name, "tools": tools, "calls": calls, "expected": dmcall.check_schema(tools, deepcopy(calls))})
    invalid = []
    for name, tools in (("invalid parameter schema", [{"type": "function", "function": {"name": "f", "parameters": {
            "type": "object", "properties": {"a": {"type": "nonsense"}}}}}]),
                        ("duplicate names", [WEATHER, WEATHER]),
                        ("parameters not an object", [{"type": "function", "function": {"name": "f", "parameters": []}}])):
        try:
            dmcall.check_schema(tools, [])
        except ValueError:
            invalid.append({"name": name, "tools": tools})
        else:
            raise AssertionError(name)
    return {"note": "check_schema(tools, calls). Error messages follow Python jsonschema's wording; errors are sorted "
                    "by str(list(path)) of the jsonschema error, then by the order jsonschema reports them.",
            "cases": cases, "configuration_errors": invalid}


# feedback.json ---------------------------------------------------------------------------------
FEEDBACK = {
    "source": "jev_noul_and_deterministic_schema_validation",
    "checks": {"function_selection": {"pass": True, "probability_yes": 0.95, "threshold": 0.5, "schema_pass": True},
               "arguments": {"skipped": False, "pass": False, "probability_yes": 0.12, "threshold": 0.5,
                             "schema_pass": False}},
    "failed_checks": ["arguments"],
    "schema_errors": [{"check": "arguments", "path": "calls[0].arguments", "message": "'city' is a required property"}],
    "instruction": prompts.REPAIR_INSTRUCTION,
}
SKIPPED = {
    "source": "jev_noul_and_deterministic_schema_validation",
    "checks": {"function_selection": {"pass": False, "probability_yes": 0.05, "threshold": 0.5, "schema_pass": True},
               "arguments": {"skipped": True, "pass": None, "probability_yes": None, "threshold": 0.5,
                             "schema_pass": True}},
    "failed_checks": ["function_selection"], "schema_errors": [], "instruction": prompts.REPAIR_INSTRUCTION,
}
CUSTOM = {"function_probability": 0.41, "arguments_probability": 1, "function_threshold": 0.7,
          "nested": [{"probability_yes": 0.2}, {"p_f": 0, "p_a": 0.6}], "instruction": "The probabilities differ."}
# Two scores that would get the same label field are an error in hybrid and text.
CONFLICTS = [{"function_probability": 0.4, "p_f": 0.4}, {"probability_yes": 0.5, "feedback_label": "x"}]


def feedback_file() -> dict:
    labels = [{"p": p, "label": dmcall.label(p)} for p in
              (0, 0.1, 0.19999, 0.2, 0.39, 0.4, 0.48, 0.5, 0.52, 0.59, 0.6, 0.79, 0.8, 0.95, 1, None)]
    represent = [{"representation": r, "feedback": f, "expected": fb.represent(f, r)}
                 for f in (FEEDBACK, SKIPPED, CUSTOM) for r in fb.FEEDBACK]
    single = [{"keep": k, "feedback": f, "expected": fb.single_check(f, k)}
              for f in (FEEDBACK, SKIPPED) for k in ("function", "arguments")]
    for conflict in CONFLICTS:
        for representation in ("hybrid", "text"):
            try:
                fb.represent(conflict, representation)
            except ValueError:
                continue
            raise AssertionError(conflict)
    return {"labels": labels, "invalid_probabilities": [-0.1, 1.1, "0.5", True],
            "represent": represent, "single_check": single,
            "conflicts": {"representations": ["hybrid", "text"], "feedback": CONFLICTS}}


# normalize_calls.json -------------------------------------------------------------------------------
SHAPES = [
    {"name": "get_weather", "arguments": {"city": "Seoul"}},
    {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Seoul"}'}},
    {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": '{"city": "Seoul"}'},
    {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Seoul"}},
    {"functionCall": {"name": "get_weather", "args": {"city": "Seoul"}}},
    {"function_call": {"name": "get_weather", "args": {"city": "Seoul"}, "id": None}},
    {"name": "ping", "arguments": ""},
    {"name": "ping"},
    {"name": "ping", "arguments": None},
    {"name": "get_weather", "arguments": "{oops"},
]


def normalize_file() -> dict:
    cases = [{"input": [shape], "expected": dmcall.normalize_calls([deepcopy(shape)])} for shape in SHAPES]
    cases.append({"input": None, "expected": dmcall.normalize_calls(None)})
    cases.append({"input": SHAPES[0], "expected": dmcall.normalize_calls(deepcopy(SHAPES[0]))})
    return {"cases": cases}


# providers.json -----------------------------------------------------------------------------------
def providers_file() -> dict:
    example = {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather in Seoul?"},
                            {"role": "assistant", "content": "Which unit?"}, {"role": "user", "content": "Celsius."}],
               "tools": [WEATHER]}
    requests = []

    def capture(name, provider, history, example=example):
        seen = []
        provider.client = mock(lambda r: (seen.append(r), httpx.Response(200, json={}))[1])
        try:
            provider.generate(example, history)
        except dmcall.ProviderError:
            pass  # an empty response body is not a valid answer; only the request is recorded here
        request = seen[0]
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() in {"authorization", "x-api-key", "api-key", "x-goog-api-key", "anthropic-version",
                                    "content-type"}}
        requests.append({"name": name, "url": str(request.url), "headers": headers,
                         "body": json.loads(request.content)})

    capture("openai with history", dmcall.OpenAI("gpt-5-nano", api_key="KEY", reasoning_effort="none"), HISTORY)
    capture("openai no tools", dmcall.OpenAI("gpt-5-nano", api_key="KEY"), None,
            {"messages": [{"role": "user", "content": "hi"}], "tools": []})
    capture("openai-compatible local", dmcall.OpenAICompatible("gemma-4-31b-it", base_url="http://localhost:8000/v1"), None)
    capture("xai", dmcall.XAI("grok-4", api_key="KEY"), None)
    capture("openrouter", dmcall.OpenRouter("meta/llama", api_key="KEY", temperature=0.2), None)
    capture("meta muse", dmcall.Meta("muse-spark-1.3", api_key="KEY"), HISTORY)
    capture("azure", dmcall.AzureOpenAI("my deployment", endpoint="https://res.openai.azure.com/", api_key="KEY"), HISTORY)
    capture("anthropic with history", dmcall.Anthropic("claude-sonnet-5-5", api_key="KEY"), HISTORY)
    capture("gemini", dmcall.Gemini("models/gemini-2.5-flash", api_key="KEY", temperature=0, thinking_budget=0), HISTORY)

    def decoded(provider_cls, body, **options):
        provider = provider_cls("m", api_key="KEY", **options)
        provider.client = mock(lambda r: httpx.Response(200, json=body))
        proposal = provider.generate(example)
        return {key: proposal[key] for key in ("calls", "content", "parse_errors", "incomplete", "usage", "model")}

    def chat(calls=(), content=None, finish=None, **message):
        tool_calls = [{"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": a}}
                      for i, (n, a) in enumerate(calls)]
        return {"model": "resolved", "choices": [{"finish_reason": finish or ("tool_calls" if tool_calls else "stop"),
                "message": {"role": "assistant", "content": content,
                            **({"tool_calls": tool_calls} if tool_calls else {}), **message}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 9, "prompt_tokens_details": {"cached_tokens": 64}}}

    anthropic_ok = {"type": "message", "role": "assistant", "model": "claude-x", "stop_reason": "tool_use",
                    "content": [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "Checking."},
                                {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Seoul"}}],
                    "usage": {"input_tokens": 50, "cache_read_input_tokens": 30, "cache_creation_input_tokens": 20,
                              "output_tokens": 7}}
    gemini_ok = {"modelVersion": "gemini-x-001", "candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": [
        {"text": "thinking", "thought": True}, {"functionCall": {"name": "get_weather", "args": {"city": "Seoul"}}}]}}],
        "usageMetadata": {"promptTokenCount": 80, "candidatesTokenCount": 6, "thoughtsTokenCount": 40}}
    decode_cases = [
        ("openai call", "openai", chat([("get_weather", '{"city": "Seoul", "days": 2.0}')])),
        ("openai text", "openai", chat(content="No function applies.")),
        ("openai empty", "openai", chat()),
        ("openai length", "openai", chat(finish="length")),
        ("openai content filter", "openai", chat(finish="content_filter")),
        ("openai refusal", "openai", chat(refusal="I can't help with that.")),
        ("openai bad json", "openai", chat([("get_weather", "{not json")])),
        ("openai empty arguments", "openai", chat([("get_weather", "")])),
        ("openai array arguments", "openai", chat([("get_weather", "[1]")])),
        ("anthropic call", "anthropic", anthropic_ok),
        ("anthropic max tokens", "anthropic", {**anthropic_ok, "stop_reason": "max_tokens", "content": []}),
        ("anthropic refusal", "anthropic", {**anthropic_ok, "stop_reason": "refusal", "content": []}),
        ("anthropic text", "anthropic", {**anthropic_ok, "stop_reason": "end_turn",
                                         "content": [{"type": "text", "text": "Hello."}]}),
        ("gemini call", "gemini", gemini_ok),
        ("gemini max tokens", "gemini", {"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": []}}]}),
        ("gemini safety", "gemini", {"candidates": [{"finishReason": "SAFETY"}]}),
        ("gemini malformed", "gemini", {"candidates": [{"finishReason": "MALFORMED_FUNCTION_CALL"}]}),
        ("gemini blocked prompt", "gemini", {"promptFeedback": {"blockReason": "SAFETY"}}),
    ]
    classes = {"openai": dmcall.OpenAI, "anthropic": dmcall.Anthropic, "gemini": dmcall.Gemini}
    decodes = [{"name": name, "provider": kind, "response": body, "expected": decoded(classes[kind], body)}
               for name, kind, body in decode_cases]
    return {"example": example, "history": HISTORY, "requests": requests, "decode": decodes,
            "filtered_prompt": {"status": 400, "response": {"error": {"code": "content_filter", "message": "filtered"}},
                                "expected_incomplete": "prompt_filter"}}


# loop_edges.json: more scenarios, recorded from the Python reference ---------------------------------
def loop_edges_file() -> dict:
    from tests_helpers import run_scenarios  # noqa: E402  (spec/tests_helpers.py)
    return run_scenarios()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    (ROOT / "prompts.json").write_text(json.dumps(prompts_file(), ensure_ascii=False, indent=1) + "\n")
    write("json_text.json", json_text_file())
    write("schema_check.json", schema_check_file())
    write("feedback.json", feedback_file())
    write("normalize_calls.json", normalize_file())
    write("providers.json", providers_file())
    write("loop_edges.json", loop_edges_file())
    from tests_helpers import add_responses_to_paper_parity  # noqa: E402
    add_responses_to_paper_parity()  # adds the served responses; the recorded requests are checked, not changed
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
