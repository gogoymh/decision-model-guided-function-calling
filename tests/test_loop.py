"""The DM-Call loop: acceptance, cumulative repair, the paper's variants, fallback and execution."""

import asyncio

import pytest

import dmcall
from conftest import WEATHER, ScriptedJev, ScriptedModel, busan, seoul


def agent(*proposals, decision=None, **options):
    model = ScriptedModel(*proposals)
    decision = decision or ScriptedJev()
    return dmcall.DMCall(model, decision_model=decision.jev, tools=[WEATHER], **options), model, decision


def test_repairs_with_cumulative_feedback_then_runs():
    ran = []

    def get_weather(city: str, unit: str = "celsius"):
        ran.append(city)
        return {"city": city, "temp": 21}

    model, decision = ScriptedModel(busan(), [], seoul()), ScriptedJev()
    result = dmcall.DMCall(model, decision_model=decision.jev, tools=[get_weather]).run("Weather in Seoul?")
    assert result.calls == seoul() and result.accepted is True and result.repairs == 2 and not result.exhausted
    assert ran == ["Seoul"] and result.executed and result.output == {"city": "Seoul", "temp": 21}
    assert [len(history) for history in model.histories] == [0, 1, 2]
    assert model.histories[2][0]["calls"] == busan() and model.histories[2][1]["calls"] == []
    assert model.histories[2][0]["feedback"]["failed_checks"] == ["arguments"]
    # Hybrid feedback: the generator reads the score and its label; the trace keeps the score.
    shown = model.histories[1][0]["feedback"]["checks"]["arguments"]
    assert shown["probability_yes"] == 0.05 and shown["feedback_label"] == "very likely incorrect"
    assert "feedback_label" not in result.trace[0]["feedback"]["checks"]["arguments"]
    assert [step["accepted"] for step in result.attempts] == [False, False, True]
    assert result.attempts[1]["arguments_probability"] is None  # a no call skips the argument question
    assert result.usage == {"generator": {"input_tokens": 30, "output_tokens": 6},
                            "decision_model": {"input_tokens": 900, "output_tokens": 0}}
    assert result.probabilities == {"function": 0.95, "arguments": 0.95}
    assert repr(result).startswith('Result(call=get_weather(city="Seoul"), accepted=True, repairs=2')


def test_rejected_calls_do_not_run_unless_asked():
    ran = []

    def get_weather(city: str):
        ran.append(city)

    model, decision = ScriptedModel(busan(), busan(), busan(), busan()), ScriptedJev()
    result = dmcall.DMCall(model, decision_model=decision.jev, tools=[get_weather]).run("Weather in Seoul?")
    assert result.accepted is False and result.exhausted and result.repairs == 3 and not result.executed and ran == []
    again = dmcall.DMCall(ScriptedModel(busan(), busan()), decision_model=ScriptedJev().jev, tools=[get_weather],
                          max_repairs=1)
    result = again.run("Weather in Seoul?", execute_unverified=True)
    assert result.repairs == 1 and result.executed and ran == ["Busan"]


def test_without_an_accepted_candidate_the_first_one_is_final():
    ran = []

    def get_weather(city: str):
        ran.append(city)
        return city

    first = dmcall.DMCall(ScriptedModel([], busan()), decision_model=ScriptedJev().jev, tools=[get_weather],
                          max_repairs=1, preserve_initial_no_call=False)
    result = first.run("Weather in Seoul?", execute_unverified=True)
    assert result.exhausted and result.accepted is False and result.repairs == 1
    assert result.no_call and result.final_attempt == 0 and not result.executed and ran == []
    assert result.text == "No tool fits." and result.probabilities == {"function": 0.05, "arguments": None}
    last = dmcall.DMCall(ScriptedModel([], busan()), decision_model=ScriptedJev().jev, tools=[get_weather],
                         max_repairs=1, on_exhaustion="last", preserve_initial_no_call=False)
    result = last.run("Weather in Seoul?", execute_unverified=True)
    assert result.calls == busan() and result.final_attempt == 1 and result.executed and ran == ["Busan"]
    per_call, _, _ = agent(busan(), [], busan(), seoul(), max_repairs=1)
    assert per_call.call("Weather in Seoul?", on_exhaustion="last").calls == []
    passed = per_call.call("Weather in Seoul?")
    assert passed.calls == seoul() and passed.accepted and passed.final_attempt == 1
    with pytest.raises(ValueError, match="on_exhaustion"):
        per_call.call("x", on_exhaustion="best")


def test_unreadable_candidate_is_repaired_and_never_run():
    loop, _, decision = agent(None, seoul(), functions={"get_weather": lambda city, unit="celsius": city})
    result = loop.run("Weather in Seoul?")
    assert result.attempts[0]["calls"] is None and result.attempts[0]["schema_errors"]
    assert result.accepted and result.output == "Seoul" and len(decision.requests) == 1
    stuck, _, _ = agent(None, max_repairs=0, functions={"get_weather": lambda city: city})
    result = stuck.run("Weather?", execute_unverified=True)
    assert result.calls is None and not result.executed and repr(result).startswith("Result(unreadable candidate")


def test_initial_no_call_is_returned_without_verification():
    loop, model, decision = agent([], decision=ScriptedJev(no_call=0.97))
    result = loop.run("Tell me a joke.")
    assert result.no_call and result.accepted is None and result.initial_no_call_preserved
    assert not result.executed and result.text == "No tool fits." and decision.requests == []
    assert result.attempts[0]["accepted"] is None and len(model.histories) == 1
    checked, _, decision = agent([], decision=ScriptedJev(no_call=0.97), preserve_initial_no_call=False)
    result = checked.call("Tell me a joke.")
    assert result.accepted is True and not result.initial_no_call_preserved and len(decision.requests) == 1


def test_latest_history_keeps_the_real_round():
    loop, model, _ = agent(busan(), [], seoul(), history="latest")
    loop.call("Weather in Seoul?")
    assert [len(history) for history in model.histories] == [0, 1, 1]
    assert model.histories[2][0]["round"] == 2 and model.histories[2][0]["calls"] == []


def test_single_question_variants():
    function_only, model, _ = agent(busan(), checks="function")
    result = function_only.call("Weather in Seoul?")  # the wrong city passes when only the function is checked
    assert result.accepted and result.repairs == 0
    assert list(result.trace[0]["feedback"]["checks"]) == ["function_selection"]
    arguments_only, _, _ = agent([], seoul(), checks="arguments", preserve_initial_no_call=False)
    result = arguments_only.call("Weather in Seoul?")  # a no call has no arguments to reject
    assert result.accepted and result.no_call and result.repairs == 0


def test_feedback_representations_reach_only_the_generator():
    for representation, field in (("numeric", "threshold"), ("text", "feedback_label"), ("hybrid", "feedback_label")):
        loop, model, _ = agent(busan(), seoul(), feedback=representation)
        result = loop.call("Weather in Seoul?")
        shown = model.histories[1][0]["feedback"]["checks"]["arguments"]
        assert field in shown and result.trace[0]["feedback"]["checks"]["arguments"]["probability_yes"] == 0.05
        assert ("probability_yes" in shown) is (representation != "text")


def test_options_and_defaults(monkeypatch):
    loop, _, _ = agent()
    assert (loop.max_repairs, loop.checks, loop.history, loop.feedback, loop.on_exhaustion,
            loop.preserve_initial_no_call) == (3, "both", "cumulative", "hybrid", "first", True)
    for bad in ({"checks": "neither"}, {"history": "none"}, {"feedback": "words"}, {"on_exhaustion": "best"},
                {"max_repairs": -1}, {"preserve_initial_no_call": "yes"}):
        with pytest.raises(ValueError):
            agent(**bad)
    with pytest.raises(ValueError, match="decision model itself"):
        dmcall.DMCall(ScriptedModel(), decision_model=ScriptedJev().jev, threshold=0.9)
    with pytest.raises(TypeError, match="judge"):
        dmcall.DMCall(ScriptedModel(), decision_model=object())
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        dmcall.DMCall(ScriptedModel())
    monkeypatch.setenv("TYPESAFE_API_KEY", "jev-env")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-env")
    with dmcall.DMCall("openai:gpt-5-nano", threshold=0.9, reasoning_effort="none") as built:
        assert isinstance(built.model, dmcall.OpenAI) and built.model.reasoning_effort == "none"
        assert built.decision_model.function_threshold == built.decision_model.arguments_threshold == 0.9
    with dmcall.DMCall("openai:gpt-5-nano") as built:  # the paper's settings by default
        assert isinstance(built.decision_model, dmcall.Jev) and built.decision_model.model == "jev-1.13.0"
        assert built.decision_model.function_threshold == built.decision_model.arguments_threshold == 0.5


def test_per_call_overrides_and_missing_functions():
    loop, _, _ = agent(busan(), seoul())
    result = loop.call("Weather in Seoul?", max_repairs=0)
    assert result.repairs == 0 and result.accepted is False and result.exhausted
    with pytest.raises(LookupError, match="No Python function"):
        loop.run("Weather in Seoul?", tools=[WEATHER])
    with pytest.raises(ValueError, match="not one of the tools"):
        loop.call("x", functions={"nope": print})


def test_async_wrappers():
    loop, _, _ = agent(seoul(), seoul(), functions={"get_weather": lambda city, unit="celsius": city.upper()})

    async def main():
        called = await loop.acall("Weather in Seoul?")
        ran = await loop.arun("Weather in Seoul?")
        verdict = await loop.averify("Weather in Seoul?", seoul())
        return called, ran, verdict

    called, ran, verdict = asyncio.run(main())
    assert called.accepted and not called.executed and ran.output == "SEOUL" and verdict


def test_messages_are_validated():
    loop, model, _ = agent(seoul(), seoul())
    loop.call([{"role": "user", "content": [{"type": "text", "text": "Weather"}, {"type": "text", "text": "in Seoul?"}]}])
    for bad in ([], [{"role": "tool", "content": "{}"}], [{"role": "user", "content": {"x": 1}}], 3):
        with pytest.raises(ValueError):
            loop.call(bad)


def test_verify_checks_calls_from_any_sdk():
    decision = ScriptedJev()
    shapes = [
        {"name": "get_weather", "arguments": {"city": "Seoul"}},
        {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Seoul"}'}},
        {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Seoul"}},
        {"functionCall": {"name": "get_weather", "args": {"city": "Seoul"}}},
    ]
    for shape in shapes:
        verdict = dmcall.verify("Weather in Seoul?", [shape], [WEATHER], decision_model=decision.jev)
        assert verdict and verdict.calls == seoul() and verdict.arguments_probability == 0.95
    assert not dmcall.verify("Weather in Seoul?", busan(), [WEATHER], decision_model=decision.jev)
    count = len(decision.requests)
    malformed = dmcall.verify("Weather?", {"name": "get_weather", "arguments": "{oops"}, [WEATHER],
                              decision_model=decision.jev)
    assert not malformed and len(decision.requests) == count  # rejected locally; nothing is sent
    assert dmcall.verify("Tell me a joke.", None, [WEATHER], decision_model=decision.jev).calls == []
    assert dmcall.verify("Tell me a joke.", [], [WEATHER], decision_model=ScriptedJev(no_call=0.9).jev)

    class SDKMessage:  # what an SDK returns for an assistant turn with a tool call
        def model_dump(self, mode="json"):
            return {"role": "assistant", "content": None, "refusal": None, "tool_calls": [
                {"id": "c0", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Busan"}'}}]}

    history = [{"role": "user", "content": "Weather in Busan, then Seoul?"}, SDKMessage(),
               {"role": "tool", "tool_call_id": "c0", "content": '{"temp": 18}'}]
    assert dmcall.verify(history, seoul(), [WEATHER], decision_model=decision.jev)
    sent = decision.requests[-1]["state"]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "tool"] and "refusal" not in sent[1]
    with pytest.raises(ValueError):
        dmcall.verify([{"role": "robot", "content": "x"}], seoul(), [WEATHER], decision_model=decision.jev)
    with pytest.raises(ValueError, match="decision model itself"):
        dmcall.verify("x", seoul(), [WEATHER], decision_model=decision.jev, threshold=0.9)
    loop, _, _ = agent(decision=decision)
    assert loop.verify("Weather in Seoul?", seoul()).accepted


def test_close_reaches_owned_clients_only():
    model = ScriptedModel()
    loop = dmcall.DMCall(model, decision_model=ScriptedJev().jev, tools=[WEATHER])
    loop.close()
    assert not model.closed  # a provider object passed in belongs to the caller
