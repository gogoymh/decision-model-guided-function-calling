"""Offline checks for the opt-in categorical planner; no real model calls."""

import copy
import importlib
import json
import math

import httpx
import pytest

from dmgfc.decision_plan import (
    ChoiceClient, DecisionPlanError, TOKENIZATION_VERSION, argument_categories,
    build_hint_example, build_plan, function_question, observable_example,
    parse_choice_response, question_batches, source_spans, token_questions,
)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for name in ("API_GATEWAY", "TYPESAFE_API_KEY", "TYPESAFE_BASE_URL", "TYPESAFE_MODEL",
                 "APIM_SUBSCRIPTION_KEY", "APIM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def example():
    return {
        "id": "DO_NOT_SEND", "category": "DO_NOT_SEND", "ground_truth": "DO_NOT_SEND",
        "name_map": {"weather": "DO_NOT_SEND"},
        "messages": [
            {"role": "system", "content": "Use the user's full conversation.", "label": "DO_NOT_SEND"},
            {"role": "user", "content": "서울, New York?", "private_label": "DO_NOT_SEND"},
            {"role": "assistant", "content": "For how long?"},
            {"role": "user", "content": "2 days.\nPlease!", "category": "DO_NOT_SEND"},
        ],
        "tools": [
            {"type": "function", "private": "DO_NOT_SEND", "function": {
                "name": "weather", "description": "Weather for a city", "reference": "DO_NOT_SEND",
                "parameters": {"type": "object", "properties": {
                    "city/name~": {"type": "string", "description": "Destination name"},
                    "days": {"type": "integer", "minimum": 1},
                    "filters": {"type": "object", "properties": {"sun": {"type": "boolean"}}},
                }, "required": ["city/name~"], "additionalProperties": False},
            }},
            {"type": "function", "function": {"name": "travel", "description": "Reserve travel",
                "parameters": {"type": "object", "properties": {"destination": {"type": "string"}}}}},
        ],
    }


def reply(questions, *, selected=None, usage=True):
    answers = {}
    for qid, question in questions.items():
        keys = list(question["criteria"])
        key = selected.get(qid, keys[0]) if isinstance(selected, dict) else (selected or keys[0])
        probabilities = {option: (1.0 if option == key else 0.0) for option in keys}
        answers[qid] = {"type": "choice", "choice": key, "probabilities": probabilities, "confidence": 1.0}
    body = {"model": "jev-1.13.0", "answers": answers}
    if usage:
        body["usage"] = {"input_tokens": 123, "output_tokens": 0}
    return body


def trace(state, questions, *, selected=None):
    parsed = parse_choice_response(reply(questions, selected=selected), questions, requested_model="jev-1.13.0")
    return {**parsed, "request_payload": {"model": "jev-1.13.0", "state": copy.deepcopy(state),
                                           "questions": copy.deepcopy(questions)},
            "api_calls": 1, "latency_s": 0.1}


def token_plan(example, *, selected="f0"):
    state, questions = function_question(example)
    function_trace = trace(state, questions, selected=selected)
    state, questions, _ = token_questions(example, selected)
    tokens = [trace(state, batch, selected="none") for batch in question_batches(questions, 8)]
    return build_plan(example, function_trace, tokens), function_trace, tokens


def test_import_and_constructor_are_network_free(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Import and configuration must not create a client or send a request")
    monkeypatch.setattr(httpx, "Client", fail)
    original = importlib.import_module("dmgfc.decision_plan")
    spec = importlib.util.spec_from_file_location("offline_choice_import", original.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client = module.ChoiceClient("SECRET")
    assert client.configuration()["endpoint"] == "https://api.typesafe.ai/v1/systemone"
    assert "SECRET" not in repr(client) + json.dumps(client.configuration())
    client.close()


def test_public_context_strips_only_private_metadata_and_keeps_full_schema(example):
    original = copy.deepcopy(example)
    public = observable_example(example)
    assert set(public) == {"messages", "tools"}
    assert "DO_NOT_SEND" not in json.dumps(public)
    assert [m["role"] for m in public["messages"]] == [m["role"] for m in original["messages"]]
    assert [m["content"] for m in public["messages"]] == [m["content"] for m in original["messages"]]
    assert public["tools"][0]["function"]["parameters"] == original["tools"][0]["function"]["parameters"]
    public["tools"][0]["function"]["parameters"]["properties"].clear()
    assert example == original


def test_flat_tool_definitions_and_public_tool_calls(example):
    example["tools"] = [example["tools"][0]["function"]]
    example["messages"].append({"role": "assistant", "content": "", "tool_calls": [{
        "id": "call-a", "type": "function", "reference": "DO_NOT_SEND",
        "function": {"name": "weather", "arguments": '{"city":"Seoul"}', "label": "DO_NOT_SEND"}}]})
    public = observable_example(example)
    assert public["tools"][0]["function"]["name"] == "weather"
    assert public["messages"][-1]["tool_calls"][0]["function"] == {
        "name": "weather", "arguments": '{"city":"Seoul"}'}
    assert "DO_NOT_SEND" not in json.dumps(public)


def test_unicode_offsets_all_user_messages_and_punctuation_are_reversible(example):
    spans = source_spans(example)
    assert {s["message_index"] for s in spans} == {1, 3}
    assert [s["text"] for s in spans] == ["서울", ",", "New", "York", "?", "2", "days", ".", "Please", "!"]
    assert spans[0] == {"message_index": 1, "token_index": 0, "char_start": 0, "char_end": 2, "text": "서울"}
    for span in spans:
        original = example["messages"][span["message_index"]]["content"]
        assert original[span["char_start"]:span["char_end"]] == span["text"]
    for index in (1, 3):
        content = example["messages"][index]["content"]
        covered = {i for span in spans if span["message_index"] == index
                   for i in range(span["char_start"], span["char_end"])}
        assert covered == {i for i, char in enumerate(content) if not char.isspace()}
    assert "unicode-lexical" in TOKENIZATION_VERSION


def test_function_choice_has_opaque_ordered_keys_and_no_call(example):
    state, questions = function_question(example)
    question = questions["function_selection"]
    assert list(question["criteria"]) == ["f0", "f1", "no_call"]
    assert question["type"] == "choice"
    assert question["criteria"]["f0"]["parameters"] == example["tools"][0]["function"]["parameters"]
    assert "DO_NOT_SEND" not in json.dumps({"state": state, "questions": questions})
    assert len(state["tools"]) == 2


def test_top_level_roles_use_schema_json_pointers_and_preserve_nesting(example):
    roles = argument_categories(example, "f0")
    assert list(roles) == ["a0", "a1", "a2", "none"]
    assert roles["a0"]["pointer"] == "/city~1name~0"
    assert roles["a0"]["required"] is True
    assert roles["a2"]["schema"] == example["tools"][0]["function"]["parameters"]["properties"]["filters"]
    assert roles["a1"]["required"] is False


def test_token_instructions_contain_source_span_evidence_and_every_token(example):
    state, questions, spans = token_questions(example, "f0")
    assert len(questions) == len(spans) == 10
    assert list(questions) == [f"t{s['message_index']}_{s['token_index']}" for s in spans]
    assert state["messages"] == observable_example(example)["messages"]
    assert len(state["tools"]) == 2
    assert state["selected_function"]["name"] == "weather"
    for span, question in zip(spans, questions.values(), strict=True):
        text = question["instructions"]
        assert json.dumps(span["text"], ensure_ascii=False) in text
        assert f"message index {span['message_index']}" in text
        assert f"[{span['char_start']}, {span['char_end']})" in text
    assert "DO_NOT_SEND" not in json.dumps({"state": state, "questions": questions})


def test_no_call_and_empty_tools_skip_tokens_without_omitting_source_spans(example):
    state, questions, spans = token_questions(example, "no_call")
    assert questions == {} and len(spans) == 10 and state["selected_function"] is None
    assert argument_categories(example, "no_call") == {}
    plan, _, _ = token_plan(example, selected="no_call")
    assert plan["function"]["name"] is None and plan["token_labels"] == []
    hinted = build_hint_example(example, plan)
    assert len(hinted["tools"]) == 2
    example["tools"] = []
    _, function_questions = function_question(example)
    assert list(function_questions["function_selection"]["criteria"]) == ["no_call"]
    plan, _, _ = token_plan(example, selected="no_call")
    assert build_hint_example(example, plan)["tools"] == []


def test_zero_parameter_function_keeps_all_tokens_as_none(example):
    example["tools"][0]["function"]["parameters"] = {"type": "object", "properties": {}}
    _, questions, spans = token_questions(example, "f0")
    assert len(questions) == len(spans) and all(list(q["criteria"]) == ["none"] for q in questions.values())
    plan, _, _ = token_plan(example)
    assert len(plan["token_labels"]) == 10
    assert all(label["parameter"] is None for label in plan["token_labels"])


def test_large_context_is_not_truncated_and_batches_cover_every_question(example):
    example["messages"] = [{"role": "user", "content": " ".join(f"word{i}!" for i in range(300))}]
    _, questions, spans = token_questions(example, "f0")
    batches = question_batches(questions, 8)
    assert len(spans) == len(questions) == 600 and len(batches) == 75
    assert [key for batch in batches for key in batch] == list(questions)
    assert spans[-1]["text"] == "!"


@pytest.mark.parametrize("count,ok", [(254, True), (255, False)])
def test_function_option_limit_includes_no_call(example, count, ok):
    example["tools"] = [{"function": {"name": f"function{i}", "parameters": {}}} for i in range(count)]
    if ok:
        assert len(function_question(example)[1]["function_selection"]["criteria"]) == 255
    else:
        with pytest.raises(ValueError, match="not truncated"):
            function_question(example)


def test_argument_option_limit_includes_none(example):
    example["tools"][0]["function"]["parameters"]["properties"] = {
        f"argument{i}": {"type": "string"} for i in range(255)}
    with pytest.raises(ValueError, match="not truncated"):
        token_questions(example, "f0")


@pytest.mark.parametrize("mutate", [
    lambda body: body["answers"].clear(),
    lambda body: body["answers"].update({"extra": {}}),
    lambda body: body["answers"]["function_selection"].update(type="noul"),
    lambda body: body["answers"]["function_selection"].update(choice="unknown"),
    lambda body: body["answers"]["function_selection"]["probabilities"].pop("no_call"),
    lambda body: body["answers"]["function_selection"]["probabilities"].update(f0=True),
    lambda body: body["answers"]["function_selection"]["probabilities"].update(f0=math.nan),
    lambda body: body["answers"]["function_selection"]["probabilities"].update(f0=1.1),
    lambda body: body["answers"]["function_selection"]["probabilities"].update(f0=0.9),
    lambda body: body["answers"]["function_selection"].update(confidence=-0.1),
    lambda body: body.update(model=None),
    lambda body: body.update(usage={"input_tokens": -1}),
    lambda body: body.update(usage={"output_tokens": True}),
    lambda body: body.update(extra=math.nan),
])
def test_invalid_categorical_reply_is_not_a_semantic_plan(example, mutate):
    _, questions = function_question(example)
    body = reply(questions)
    mutate(body)
    with pytest.raises(DecisionPlanError):
        parse_choice_response(body, questions)


def test_nonmaximal_choice_rejected_and_low_confidence_retained(example):
    _, questions = function_question(example)
    body = reply(questions)
    answer = body["answers"]["function_selection"]
    answer.update(choice="f1", probabilities={"f0": 0.6, "f1": 0.4, "no_call": 0.0})
    with pytest.raises(DecisionPlanError, match="non-maximal"):
        parse_choice_response(body, questions)
    answer.update(choice="f0", probabilities={"f0": 0.334, "f1": 0.333, "no_call": 0.333}, confidence=0.001)
    parsed = parse_choice_response(body, questions)
    assert parsed["answers"]["function_selection"]["choice"] == "f0"
    assert parsed["answers"]["function_selection"]["confidence"] == 0.001


def test_missing_usage_is_unknown_not_zero(example):
    _, questions = function_question(example)
    parsed = parse_choice_response(reply(questions, usage=False), questions)
    assert parsed["usage"] == {"input_tokens": None, "output_tokens": None}
    parsed = parse_choice_response(reply(questions), questions)
    assert parsed["usage"] == {"input_tokens": 123, "output_tokens": 0}


@pytest.mark.parametrize("probabilities", [
    {"f0": .81, "f1": .10, "no_call": .08},
    {"f0": .81, "f1": .12, "no_call": .08},
])
def test_hosted_hundredth_rounding_retains_values_without_renormalization(example, probabilities):
    _, questions = function_question(example)
    body = reply(questions)
    body["answers"]["function_selection"].update(probabilities=probabilities)
    parsed = parse_choice_response(body, questions)
    assert parsed["answers"]["function_selection"]["probabilities"] == probabilities
    assert parsed["answers"]["function_selection"]["choice"] == "f0"


@pytest.mark.parametrize("probabilities", [
    {"f0": .80, "f1": .09, "no_call": .08},
    {"f0": .8101, "f1": .10, "no_call": .08},
])
def test_rounding_allowance_rejects_excess_or_nonhundredth_sum_error(example, probabilities):
    _, questions = function_question(example)
    body = reply(questions)
    body["answers"]["function_selection"].update(probabilities=probabilities)
    with pytest.raises(DecisionPlanError, match="sum to one"):
        parse_choice_response(body, questions)


def test_vendor_choice_one_display_tick_discrepancy_is_preserved(example):
    _, questions = function_question(example)
    body = reply(questions)
    body["answers"]["function_selection"].update(choice="f0",
        probabilities={"f0": .45, "f1": .46, "no_call": .09})
    parsed = parse_choice_response(body, questions)
    assert parsed["answers"]["function_selection"]["choice"] == "f0"
    assert parsed["answers"]["function_selection"]["probabilities"] == body["answers"]["function_selection"]["probabilities"]


@pytest.mark.parametrize("probabilities", [
    {"f0": .44, "f1": .46, "no_call": .10},
    {"f0": .454, "f1": .455, "no_call": .091},
])
def test_display_consistency_allowance_rejects_larger_or_nonhundredth_mismatch(example, probabilities):
    _, questions = function_question(example)
    body = reply(questions)
    body["answers"]["function_selection"].update(choice="f0", probabilities=probabilities)
    with pytest.raises(DecisionPlanError, match="non-maximal"):
        parse_choice_response(body, questions)


def test_real_mock_wire_uses_choice_full_context_and_safe_journal(example):
    state, questions = function_question(example)
    events, observed = [], []
    def handler(request):
        payload = json.loads(request.content)
        observed.append(payload)
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        assert request.headers["Authorization"] == "Bearer SECRET"
        assert "DO_NOT_SEND" not in request.content.decode()
        return httpx.Response(200, json=reply(questions))
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        choice = ChoiceClient("SECRET", client=http, journal_hook=events.append)
        result = choice.ask(state, questions)
        choice.close()
        assert not http.is_closed
    assert len(observed) == 1 and result["api_calls"] == 1
    assert result["request_payload"] == observed[0]
    assert [event["event"] for event in events] == ["request_started", "request_finished"]
    assert events[-1]["status"] == "completed"
    assert events[0]["request_id"] == events[-1]["request_id"]
    assert "SECRET" not in json.dumps(events) + repr(choice)
    assert result["requested_model"] == result["resolved_model"] == "jev-1.13.0"


def test_same_frozen_apim_routing_without_credentials_in_config(example, monkeypatch):
    monkeypatch.setenv("APIM_SUBSCRIPTION_KEY", "APIMSECRET")
    monkeypatch.setenv("APIM_BASE_URL", "https://gateway.example")
    observed = []
    state, questions = function_question(example)
    def handler(request):
        observed.append(request)
        assert str(request.url) == "https://gateway.example/llm/typesafe/v1/systemone"
        assert request.headers["api-key"] == "APIMSECRET"
        return httpx.Response(200, json=reply(questions))
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = ChoiceClient(gateway="apim", client=http)
        assert "APIMSECRET" not in json.dumps(client.configuration())
        client.ask(state, questions)
    assert len(observed) == 1


@pytest.mark.parametrize("kind", ["timeout", "http", "malformed", "contract"])
def test_failures_close_journal_without_retry_body_leak_or_fallback(example, kind):
    state, questions = function_question(example)
    events, calls = [], []
    def handler(request):
        calls.append(request)
        if kind == "timeout":
            raise httpx.ReadTimeout("SECRET HTTP details", request=request)
        if kind == "http":
            return httpx.Response(503, text="SECRET HTTP body")
        if kind == "malformed":
            return httpx.Response(200, text="SECRET not json")
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(DecisionPlanError) as error:
            ChoiceClient("SECRET", client=http, journal_hook=events.append).ask(state, questions)
    assert len(calls) == 1 and len(events) == 2
    assert events[-1]["status"] in {"transport_error", "response_error"}
    assert events[-1]["usage"] is None
    assert events[-1]["outcome_unknown"] is (kind == "timeout")
    assert "SECRET" not in str(error.value) + json.dumps(events)


def test_journal_start_failure_prevents_any_model_request(example):
    state, questions = function_question(example)
    def handler(request):
        pytest.fail("A start event must be persisted before POST")
    def journal(event):
        raise OSError("Journal unavailable")
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(OSError, match="Journal unavailable"):
            ChoiceClient("SECRET", client=http, journal_hook=journal).ask(state, questions)


@pytest.mark.parametrize("arguments", [{"max_retries": 1}, {"max_retries": True}, {"timeout": 0},
                                        {"timeout": math.inf}, {"base_url": "https://secret@host"}])
def test_configuration_errors_are_local(arguments):
    with pytest.raises(ValueError):
        ChoiceClient("SECRET", **arguments)


def test_function_only_and_token_hints_share_function_and_keep_generator_inputs(example):
    original = copy.deepcopy(example)
    plan, function_trace, _ = token_plan(example)
    function_only = build_plan(example, function_trace)
    with_tokens = build_hint_example(example, plan, include_tokens=True)
    without_tokens = build_hint_example(example, function_only, include_tokens=False)
    public = observable_example(example)
    assert with_tokens["messages"][1:] == without_tokens["messages"][1:] == public["messages"]
    assert with_tokens["tools"] == without_tokens["tools"] == public["tools"]
    assert len(with_tokens["messages"]) == len(original["messages"]) + 1
    assert "NOT executed" in with_tokens["messages"][0]["content"]
    assert "may be overridden" in with_tokens["messages"][0]["content"]
    assert '"token_labels"' not in without_tokens["messages"][0]["content"]
    assert '"token_labels"' in with_tokens["messages"][0]["content"]
    assert plan["function"] == function_only["function"]
    assert not {"tool_choice", "accepted", "threshold", "pass"}.intersection(with_tokens)
    assert example == original
    assert "DO_NOT_SEND" not in json.dumps(with_tokens)


def test_unknown_private_plan_fields_are_not_exposed(example):
    plan, _, _ = token_plan(example)
    plan["ground_truth"] = "DO_NOT_SEND"
    plan["function"]["verdict"] = "DO_NOT_SEND"
    for label in plan["token_labels"]:
        label["private_score"] = "DO_NOT_SEND"
    assert "DO_NOT_SEND" not in json.dumps(build_hint_example(example, plan))


def test_plan_rejects_missing_duplicate_wrong_context_or_altered_token_traces(example):
    plan, function_trace, tokens = token_plan(example)
    assert len(plan["token_labels"]) == len(source_spans(example))
    with pytest.raises(ValueError, match="every source token"):
        build_plan(example, function_trace, tokens[:-1])
    with pytest.raises(ValueError, match="more than once"):
        build_plan(example, function_trace, tokens + tokens[:1])
    bad = copy.deepcopy(tokens)
    bad[0]["request_payload"]["state"]["messages"][1]["content"] = "different context"
    with pytest.raises(ValueError, match="complete source context"):
        build_plan(example, function_trace, bad)
    bad = copy.deepcopy(tokens)
    first_qid = next(iter(bad[0]["request_payload"]["questions"]))
    bad[0]["request_payload"]["questions"][first_qid]["instructions"] += "changed"
    with pytest.raises(ValueError, match="differs"):
        build_plan(example, function_trace, bad)
    bad_function = copy.deepcopy(function_trace)
    bad_function["request_payload"]["state"]["messages"].clear()
    with pytest.raises(ValueError, match="exact source context"):
        build_plan(example, bad_function)


def test_hint_rejects_missing_or_fabricated_source_tokens(example):
    plan, _, _ = token_plan(example)
    altered = copy.deepcopy(plan)
    altered["token_labels"].pop()
    with pytest.raises(ValueError, match="every source token"):
        build_hint_example(example, altered)
    altered = copy.deepcopy(plan)
    altered["token_labels"][0]["text"] = "invented"
    with pytest.raises(ValueError, match="exact source spans"):
        build_hint_example(example, altered)
    altered = copy.deepcopy(plan)
    altered["token_labels"][0]["category"] = "missing"
    with pytest.raises(ValueError, match="unavailable argument category"):
        build_hint_example(example, altered)
    altered = copy.deepcopy(plan)
    altered["token_labels"][0]["parameter"] = "invented"
    with pytest.raises(ValueError, match="does not match"):
        build_hint_example(example, altered)


def test_nontext_user_and_malformed_tools_fail_without_omission(example):
    example["messages"][1]["content"] = [{"type": "text", "text": "partial"}]
    with pytest.raises(ValueError, match="every user message"):
        source_spans(example)
    example["messages"][1]["content"] = "text"
    example["tools"].append(copy.deepcopy(example["tools"][0]))
    with pytest.raises(ValueError, match="unique"):
        function_question(example)
