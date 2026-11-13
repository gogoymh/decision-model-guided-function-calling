"""DM-Call sends exactly what the code behind the paper's tables sent.

``spec/conformance/paper_parity.json`` was recorded from that code (the earlier ``dmgfc`` and ``jev_fc``
packages, commit 4bbb048) before they were merged into ``dmcall``. For each scenario it keeps
every request body sent to the generator and to the decision model, and the result. The same
scripted responses are replayed here, so any change to a prompt, a question, the feedback
record, its representation, the history format or the loop shows up as a difference.
"""

import json
from copy import deepcopy
from pathlib import Path

import httpx
import pytest

import dmcall

GOLDEN = json.loads((Path(__file__).resolve().parents[1] / "spec" / "conformance" / "paper_parity.json").read_text())
LENGTH = "length"


def chat_response(calls):
    if calls == LENGTH:
        return {"model": "gpt-5-nano-2025-08-07", "choices": [{"finish_reason": "length",
                "message": {"role": "assistant", "content": ""}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 4096}}
    tool_calls = [{"id": f"call_{i}", "type": "function",
                   "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                  for i, c in enumerate(calls)]
    message = {"role": "assistant", "content": None if tool_calls else "It is 21 degrees there, I think."}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"model": "gpt-5-nano-2025-08-07", "choices": [{
        "finish_reason": "tool_calls" if tool_calls else "stop", "message": message}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 12, "prompt_tokens_details": {"cached_tokens": 0}}}


def anthropic_response(calls):
    content = [{"type": "tool_use", "id": f"t{i}", "name": c["name"], "input": c["arguments"]}
               for i, c in enumerate(calls)] or [{"type": "text", "text": "It is 21 degrees there, I think."}]
    return {"type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
            "stop_reason": "tool_use" if calls else "end_turn", "content": content,
            "usage": {"input_tokens": 100, "output_tokens": 10}}


def decision_answer(body):
    """Function: right only for get_weather. Arguments: right only for Seoul. No call: wrong."""
    calls = body["state"]["proposed_calls"]
    if not calls:
        function, arguments = 0.05, None
    else:
        function = 0.95 if calls[0]["name"] == "get_weather" else 0.1
        arguments = 0.93 if calls[0]["arguments"].get("city") == "Seoul" else 0.07
    answers = {"function_correct": {"type": "noul", "noul": function}}
    if "arguments_correct" in body["questions"]:
        answers["arguments_correct"] = {"type": "noul", "noul": arguments}
    return {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 400, "output_tokens": 0}}


def recorder(responses, bodies, respond):
    responses = list(responses)

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=respond(responses.pop(0)))
    return httpx.Client(transport=httpx.MockTransport(handler))


def decision_recorder(bodies):
    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(200, json=decision_answer(body))
    return httpx.Client(transport=httpx.MockTransport(handler))


def summary(result):
    return {"calls": result.calls, "accepted": result.accepted, "exhausted": result.exhausted,
            "repairs": result.repairs, "final_attempt": result.final_attempt,
            "initial_no_call_preserved": result.initial_no_call_preserved,
            "attempts": result.attempts}


@pytest.mark.parametrize("name", sorted(GOLDEN["scenarios"]))
def test_loop_matches_the_papers_code(name):
    scenario = GOLDEN["scenarios"][name]
    generator_bodies, decision_bodies = [], []
    jev = dmcall.Jev("jev-key", client=decision_recorder(decision_bodies))
    if scenario["provider"] == "anthropic":
        model = dmcall.Anthropic("claude-sonnet-5-5", api_key="k",
                                 client=recorder(scenario["proposals"], generator_bodies, anthropic_response))
    else:
        model = dmcall.OpenAI("gpt-5-nano", api_key="k", reasoning_effort="none",
                              client=recorder(scenario["proposals"], generator_bodies, chat_response))
    agent = dmcall.DMCall(model, decision_model=jev, tools=GOLDEN["tools"], **scenario["options"])
    result = agent.call(deepcopy(GOLDEN["messages"]))
    assert generator_bodies == scenario["generator_requests"]
    assert decision_bodies == scenario["decision_requests"]
    assert summary(result) == scenario["result"]
    # The history message is sent as text, so its exact characters matter, not only its JSON.
    for ours, theirs in zip(generator_bodies, scenario["generator_requests"]):
        assert json.dumps(ours, ensure_ascii=False) == json.dumps(theirs, ensure_ascii=False)


def test_verify_matches_the_papers_code():
    recorded = GOLDEN["verify"]
    bodies = []
    jev = dmcall.Jev("jev-key", client=decision_recorder(bodies))
    seoul = [{"name": "get_weather", "arguments": {"city": "Seoul"}}]
    verdicts = [dmcall.verify(recorded["messages"], seoul, GOLDEN["tools"], decision_model=jev),
                dmcall.verify("Tell me a joke.", [], GOLDEN["tools"], decision_model=jev)]
    assert bodies == recorded["decision_requests"]
    assert [{"accepted": v.accepted, "calls": v.calls, "function_probability": v.function_probability,
             "arguments_probability": v.arguments_probability, "schema_errors": v.schema_errors,
             "feedback": v.feedback} for v in verdicts] == recorded["verdicts"]
