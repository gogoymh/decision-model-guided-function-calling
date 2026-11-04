"""Offline checks of argument→source-token selection, with no model requests."""

import copy
import importlib
import json

import httpx
import pytest

from dmgfc.token_choice_plan import (
    ChoiceClient, DecisionPlanError, PLAN_FORMAT, TOKEN_PATTERN, TOKENIZATION_VERSION,
    argument_parameters, argument_questions, build_hint_example, build_plan, function_question,
    observable_example, parse_choice_response, question_batches, source_spans, token_candidates,
)


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch):
    for name in ("API_GATEWAY", "TYPESAFE_API_KEY", "TYPESAFE_BASE_URL", "TYPESAFE_MODEL",
                 "APIM_SUBSCRIPTION_KEY", "APIM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def example():
    return {"id": "PRIVATE_GOLD", "category": "PRIVATE_GOLD", "ground_truth": "PRIVATE_GOLD",
        "messages": [
            {"role": "system", "content": "Keep the full conversation", "label": "PRIVATE_GOLD"},
            {"role": "user", "content": "서울, New York? -2.5 2026-10-26 a.b(x)", "label": "PRIVATE_GOLD"},
            {"role": "assistant", "content": "For how long?"},
            {"role": "user", "content": "2 days.\nNew York?", "label": "PRIVATE_GOLD"},
        ],
        "tools": [{"private": "PRIVATE_GOLD", "type": "function", "function": {
            "name": "weather", "description": "A weather request", "label": "PRIVATE_GOLD",
            "parameters": {"type": "object", "properties": {
                "city/name~": {"type": "string", "description": "Destination city"},
                "days": {"type": "integer", "minimum": 1},
                "filters": {"type": "object", "properties": {"sun": {"type": "boolean"}}},
            }, "required": ["city/name~"], "additionalProperties": False}}},
            {"function": {"name": "travel", "description": "Book a trip",
                          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
    }


def response(questions, selected=None):
    answers = {}
    for qid, question in questions.items():
        keys = list(question["criteria"])
        key = selected.get(qid, keys[0]) if isinstance(selected, dict) else (selected or keys[0])
        answers[qid] = {"type": "choice", "choice": key,
                        "probabilities": {option: 1.0 if option == key else 0.0 for option in keys},
                        "confidence": 1.0}
    return {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 51, "output_tokens": 0}}


def trace(state, questions, selected=None):
    parsed = parse_choice_response(response(questions, selected), questions, requested_model="jev-1.13.0")
    return {**parsed, "request_payload": {"model": "jev-1.13.0", "state": copy.deepcopy(state),
                                           "questions": copy.deepcopy(questions)},
            "api_calls": 1, "latency_s": 0.1}


def plans(example, selected="f0", tokens=None):
    state, questions = function_question(example)
    function_trace = trace(state, questions, selected)
    state, questions, _ = argument_questions(example, selected)
    argument_traces = [trace(state, batch, tokens or "none") for batch in question_batches(questions, 8)]
    return build_plan(example, function_trace), build_plan(example, function_trace, argument_traces), argument_traces


def test_import_constructor_metadata_are_network_free(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No client construction or HTTP request is allowed here")
    monkeypatch.setattr(httpx, "Client", forbidden)
    original = importlib.import_module("dmgfc.token_choice_plan")
    spec = importlib.util.spec_from_file_location("dmgfc.offline_token_choice", original.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client = module.ChoiceClient("SECRET")
    config = client.configuration()
    assert config["token_pattern"] == r"\S+"
    assert config["tokenization_version"] == TOKENIZATION_VERSION
    assert config["argument_selection"] == "one-source-token-per-argument"
    assert "SECRET" not in json.dumps(config) + repr(client)
    client.close()


def test_exact_nonwhitespace_offsets_and_all_user_messages(example):
    original = copy.deepcopy(example)
    spans = source_spans(example)
    assert TOKEN_PATTERN == r"\S+"
    assert [span["text"] for span in spans] == [
        "서울,", "New", "York?", "-2.5", "2026-10-26", "a.b(x)", "2", "days.", "New", "York?"]
    assert spans[0] == {"message_index": 1, "token_index": 0, "char_start": 0, "char_end": 3, "text": "서울,"}
    assert {span["message_index"] for span in spans} == {1, 3}
    for index in (1, 3):
        content = example["messages"][index]["content"]
        covered = set()
        for span in spans:
            if span["message_index"] == index:
                assert content[span["char_start"]:span["char_end"]] == span["text"]
                covered.update(range(span["char_start"], span["char_end"]))
        assert covered == {i for i, char in enumerate(content) if not char.isspace()}
    assert example == original


def test_arguments_are_questions_and_tokens_are_options_with_full_context(example):
    state, questions, spans = argument_questions(example, "f0")
    assert list(questions) == ["a0", "a1", "a2"]
    assert len(questions) != len(spans)
    assert state["messages"] == observable_example(example)["messages"]
    assert len(state["tools"]) == 2
    assert state["selected_function"]["name"] == "weather"
    assert state["selected_function"]["parameters"] == example["tools"][0]["function"]["parameters"]
    parameters = argument_parameters(example, "f0")
    assert parameters["a0"]["pointer"] == "/city~1name~0"
    assert parameters["a0"]["required"] is True
    assert parameters["a1"]["required"] is False
    assert parameters["a2"]["schema"] == example["tools"][0]["function"]["parameters"]["properties"]["filters"]
    for qid, question in questions.items():
        assert question["type"] == "choice"
        assert list(question["criteria"]) == [f"t{i}" for i in range(10)] + ["none"]
        assert json.dumps(parameters[qid], ensure_ascii=False) in question["instructions"]
        assert question["criteria"]["t3"]["text"] == "-2.5"
        assert "does not force" in question["criteria"]["none"]["description"]
    assert "PRIVATE_GOLD" not in json.dumps({"state": state, "questions": questions})


def test_repeated_tokens_are_distinct_options_and_can_anchor_multiple_arguments(example):
    candidates = token_candidates(example)
    assert candidates["t1"]["text"] == candidates["t8"]["text"] == "New"
    assert candidates["t1"]["message_index"] != candidates["t8"]["message_index"]
    _, plan, _ = plans(example, tokens={"a0": "t8", "a1": "t6", "a2": "t8"})
    assert [item["token_key"] for item in plan["argument_tokens"]] == ["t8", "t6", "t8"]
    assert plan["argument_tokens"][0]["token"] == plan["argument_tokens"][2]["token"] == candidates["t8"]


def test_argument_batches_preserve_all_options_and_are_not_per_token(example):
    example["tools"][0]["function"]["parameters"] = {"type": "object", "properties": {
        f"param{i}": {"type": "string"} for i in range(21)}}
    state, questions, spans = argument_questions(example, "f0")
    batches = question_batches(questions, 8)
    assert [len(batch) for batch in batches] == [8, 8, 5]
    assert len(spans) == 10 and len(questions) == 21
    assert [key for batch in batches for key in batch] == list(questions)
    for batch in batches:
        assert all(question["criteria"] == token_candidates(example) for question in batch.values())
    function_trace = trace(*function_question(example), selected="f0")
    plan = build_plan(example, function_trace, [trace(state, batch, "t1") for batch in batches])
    assert len(plan["argument_tokens"]) == 21


@pytest.mark.parametrize("count,allowed", [(254, True), (255, False)])
def test_token_option_cap_never_truncates(example, count, allowed):
    example["messages"] = [{"role": "user", "content": " ".join(f"token{i}" for i in range(count))}]
    assert len(source_spans(example)) == count
    if allowed:
        _, questions, _ = argument_questions(example, "f0")
        assert all(len(question["criteria"]) == 255 for question in questions.values())
        assert "t253" in questions["a0"]["criteria"]
    else:
        with pytest.raises(ValueError, match="not truncated"):
            argument_questions(example, "f0")


def test_empty_user_text_has_none_only_not_fabricated_tokens(example):
    example["messages"] = [{"role": "user", "content": " \n\t "}]
    state, questions, spans = argument_questions(example, "f0")
    assert spans == [] and all(list(q["criteria"]) == ["none"] for q in questions.values())
    _, plan, _ = plans(example)
    assert all(item["token"] is None and item["token_key"] == "none" for item in plan["argument_tokens"])
    hint = build_hint_example(example, plan)
    assert hint["messages"][1:] == observable_example(example)["messages"]
    assert "does not force argument omission" in hint["messages"][0]["content"]


def test_no_call_and_zero_argument_functions_skip_all_argument_requests(example):
    state, questions, spans = argument_questions(example, "no_call")
    assert state["selected_function"] is None and questions == {} and len(spans) == 10
    _, plan, traces = plans(example, selected="no_call")
    assert traces == [] and plan["argument_tokens"] == []
    assert plan["function"]["key"] == "no_call"
    assert build_hint_example(example, plan)["tools"] == observable_example(example)["tools"]
    example["tools"][0]["function"]["parameters"] = {"type": "object", "properties": {}}
    _, plan, traces = plans(example)
    assert plan["function"]["name"] == "weather" and traces == [] and plan["argument_tokens"] == []


def test_empty_catalog_is_local_no_call_without_fabricating_trace_or_http(example, monkeypatch):
    monkeypatch.setattr(ChoiceClient, "ask", lambda *args: pytest.fail("Empty catalog needs no model request"))
    example["tools"] = []
    plan = build_plan(example, None, [])
    assert plan["function"] == {"key": "no_call", "name": None, "decision": None, "local_reason": "empty_catalog"}
    assert build_hint_example(example, plan)["tools"] == []
    example["tools"] = [{"function": {"name": "tool", "parameters": {}}}]
    with pytest.raises(ValueError, match="actual function"):
        build_plan(example, None, [])


def test_function_only_plan_and_full_token_plan_share_exact_function_choice(example):
    function, token, _ = plans(example, tokens={"a0": "t1", "a1": "t6", "a2": "none"})
    assert function["function"] == token["function"]
    assert function["format"] == token["format"] == PLAN_FORMAT
    assert function["argument_tokens_included"] is False
    assert token["argument_tokens_included"] is True
    assert build_hint_example(example, function, include_tokens=False)["messages"][1:] == observable_example(example)["messages"]
    with pytest.raises(ValueError, match="cover every"):
        build_hint_example(example, function)


def test_hint_preserves_inputs_and_only_selected_anchors_not_probabilities(example):
    original = copy.deepcopy(example)
    _, plan, _ = plans(example, tokens={"a0": "t1", "a1": "t6", "a2": "none"})
    plan["private"] = "PRIVATE_GOLD"
    plan["argument_tokens"][0]["private"] = "PRIVATE_GOLD"
    hinted = build_hint_example(example, plan)
    assert set(hinted) == {"messages", "tools"}
    assert hinted["messages"][1:] == observable_example(example)["messages"]
    assert hinted["tools"] == observable_example(example)["tools"]
    content = hinted["messages"][0]["content"]
    assert "PRIVATE_GOLD" not in content and "probabilities" not in content and "confidence" not in content
    assert "threshold" not in content and "tool_choice" not in content and '"pass"' not in content
    assert "NOT executed" in content and "may be overridden" in content
    assert "only a source anchor" in content
    hint = json.loads(content.split("\n")[1])
    assert hint["argument_tokens"][0]["token"]["text"] == "New"
    assert "schema" not in hint["argument_tokens"][0]
    assert example == original


@pytest.mark.parametrize("mutator", [
    lambda t: t["request_payload"]["state"]["messages"][1].update(content="changed"),
    lambda t: t["request_payload"]["questions"]["a0"]["criteria"]["t1"].update(text="invented"),
    lambda t: t["request_payload"].update(model="wrong-model"),
    lambda t: t["answers"]["a0"].update(choice="none"),
    lambda t: t.update(resolved_model="wrong-model"),
    lambda t: t.update(usage={"input_tokens": 0, "output_tokens": 0}),
])
def test_plan_rejects_argument_context_question_reply_or_usage_tampering(example, mutator):
    function_trace = trace(*function_question(example), selected="f0")
    state, questions, _ = argument_questions(example, "f0")
    actual = trace(state, questions, selected="t1")
    mutator(actual)
    with pytest.raises((ValueError, DecisionPlanError)):
        build_plan(example, function_trace, [actual])


def test_plan_rejects_missing_duplicate_or_wrong_argument_questions(example):
    function_trace = trace(*function_question(example), selected="f0")
    state, questions, _ = argument_questions(example, "f0")
    actual = trace(state, questions, selected="none")
    with pytest.raises(ValueError, match="cover every"):
        build_plan(example, function_trace, [])
    with pytest.raises(ValueError, match="more than once"):
        build_plan(example, function_trace, [actual, actual])
    wrong = trace(state, {"t0": next(iter(questions.values()))}, "none")
    with pytest.raises(ValueError, match="exact nonempty subsets"):
        build_plan(example, function_trace, [wrong])


@pytest.mark.parametrize("mutator", [
    lambda p: p.update(format="fallible-decision-plan-v1"),
    lambda p: p.update(executed=True),
    lambda p: p["argument_tokens"].reverse(),
    lambda p: p["argument_tokens"][0].update(parameter="wrong"),
    lambda p: p["argument_tokens"][0].update(token_key="t999"),
    lambda p: p["argument_tokens"][0]["token"].update(char_start=99),
    lambda p: p["argument_tokens"][0].update(token={"text": "invented"}),
])
def test_hint_rejects_wrong_format_schema_order_or_source_offset(example, mutator):
    _, plan, _ = plans(example, tokens="t1")
    mutator(plan)
    with pytest.raises(ValueError):
        build_hint_example(example, plan)


def test_mock_choice_transport_wire_and_journal_have_correct_selection_direction(example):
    requests, events = [], []
    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.url.path == "/v1/systemone"
        assert set(payload) == {"model", "state", "questions"}
        assert "PRIVATE_GOLD" not in json.dumps(payload)
        selected = "f0" if "function_selection" in payload["questions"] else "t1"
        return httpx.Response(200, json=response(payload["questions"], selected))
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = ChoiceClient("SECRET", client=http, journal_hook=events.append)
        function_trace = client.ask(*function_question(example))
        state, questions, _ = argument_questions(example, "f0")
        argument_trace = client.ask(state, questions)
        plan = build_plan(example, function_trace, [argument_trace])
        client.close()
    assert len(requests) == 2 and len(events) == 4
    assert list(requests[0]["questions"]) == ["function_selection"]
    assert list(requests[1]["questions"]) == ["a0", "a1", "a2"]
    assert all(len(q["criteria"]) == 11 for q in requests[1]["questions"].values())
    assert all(item["token_key"] == "t1" for item in plan["argument_tokens"])
    assert all(event["status"] == "completed" for event in events if event["event"] == "request_finished")
    assert "SECRET" not in json.dumps(events)


def test_non_text_user_content_is_rejected_even_if_no_call(example):
    example["messages"][1]["content"] = [{"type": "text", "text": "Hello"}]
    with pytest.raises(ValueError, match="every user message"):
        source_spans(example)
    with pytest.raises(ValueError, match="every user message"):
        argument_questions(example, "no_call")
