"""DM-Call: propose a function call, verify it with a decision model, repair it with cumulative feedback.

At step t the generator proposes a candidate c_t (one call, or no call). The candidate is
accepted when it passes the schema check S and every decision-model question that applies to
it (Eq. 2 of the paper). A rejected candidate and its feedback d_t join the history H, which
the generator reads, oldest first and marked as not executed, before the unchanged request.
The loop returns the first accepted candidate; if none of c_0..c_r is accepted it returns c_0.
Nothing is executed while the loop runs.
"""

from __future__ import annotations

import asyncio
import json
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

from .decision import Jev
from .feedback import FEEDBACK, represent_history, single_check
from .providers import provider as make_provider
from .schema import check_schema
from .tools import _plain, invoke, normalize_calls, toolset

__all__ = ["CHECKS", "DMCall", "HISTORIES", "ON_EXHAUSTION", "Result", "Verdict", "verify"]

DEFAULT_THRESHOLD = 0.5   # pass mark of both questions in the paper
DEFAULT_MAX_REPAIRS = 3   # r* in the paper
CHECKS = ("both", "function", "arguments")    # DM-Call, DM-Call-Function, DM-Call-Argument
HISTORIES = ("cumulative", "latest")          # DM-Call, DM-Call-Latest
ON_EXHAUSTION = ("first", "last")
_ROLES = {"system", "developer", "user", "assistant"}
_HISTORY_ROLES = _ROLES | {"tool", "function"}
_HISTORY_KEYS = ("role", "content", "name", "tool_calls", "tool_call_id")
_TEXT_PARTS = {"text", "input_text", "output_text"}


def _choice(name: str, value: Any, choices: tuple[str, ...]) -> str:
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(map(repr, choices))}.")
    return value


def _repairs(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("max_repairs must be a nonnegative integer.")
    return value


def _messages(messages: Any) -> list[dict]:
    """A prompt string or chat messages as [{"role", "content"}] with text content."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    elif isinstance(messages, dict):
        messages = [messages]
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("messages must be a string or a nonempty list of {'role', 'content'} messages.")
    result = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in _ROLES:
            raise ValueError("Each message needs a role (system, developer, user or assistant) and text "
                             "content; give earlier tool results to the model as text.")
        content = message.get("content")
        if isinstance(content, list):
            if not all(isinstance(part, dict) and part.get("type") in _TEXT_PARTS and isinstance(part.get("text"), str)
                       for part in content):
                raise ValueError("Only text message content is supported.")
            content = "\n".join(part["text"] for part in content)
        if not isinstance(content, str):
            raise ValueError("Message content must be text.")
        result.append({"role": message["role"], "content": content})
    return result


def _conversation(messages: Any) -> list[dict]:
    """A conversation as the decision model sees it: text turns, earlier tool calls and tool results.

    Messages may be dicts or SDK message objects; only role, content, name, tool_calls and
    tool_call_id are kept.
    """
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    if isinstance(messages, dict) or not isinstance(messages, (list, tuple)):
        messages = [messages]
    result = []
    for message in messages:
        message = _plain(message)
        if not isinstance(message, dict) or message.get("role") not in _HISTORY_ROLES:
            raise ValueError("Each message needs a role: system, developer, user, assistant or tool.")
        result.append({key: deepcopy(message[key]) for key in _HISTORY_KEYS if message.get(key) is not None})
    if not result:
        raise ValueError("messages must not be empty.")
    return result


def _format(call: Any) -> str:
    if not isinstance(call, dict):
        return "<malformed call>"
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        return f"{call.get('name')}(<malformed arguments>)"
    return f"{call.get('name')}(" + ", ".join(f"{key}={json.dumps(value, ensure_ascii=False)}"
                                             for key, value in arguments.items()) + ")"


def _add(total: dict, counts: Any) -> None:
    for key, value in (counts or {}).items():
        if isinstance(value, int) and not isinstance(value, bool):
            total[key] = total.get(key, 0) + value


def _readable_no_call(proposal: dict) -> bool:
    """A completely parsed decision to make no call. Unreadable output (calls=None) is not one."""
    return (isinstance(proposal.get("calls"), list) and proposal["calls"] == []
            and not proposal.get("incomplete") and not proposal.get("parse_errors"))


@dataclass
class Result:
    """The outcome of one DM-Call request.

    ``calls`` is the final candidate as ``[{"name", "arguments"}]``: ``[]`` is a decision to make
    no call and ``None`` an unreadable candidate (cut off, filtered or refused). ``accepted``
    says whether it passed every check; it is ``None`` for an initial no-call returned without
    verification (``initial_no_call_preserved``). When no candidate passed within the repair
    limit (``exhausted``), the final candidate is the first one (``on_exhaustion="first"``, the
    paper's rule) or the last one; ``final_attempt`` is its index in ``trace``, which keeps every
    candidate, judgment and feedback record in order.
    """

    calls: list[dict] | None
    accepted: bool | None
    exhausted: bool
    repairs: int
    text: str | None = None
    executed: bool = False
    outputs: list = field(default_factory=list)
    latency_s: float = 0.0
    trace: list[dict] = field(default_factory=list, repr=False)
    final_attempt: int | None = None
    initial_no_call_preserved: bool = False

    @property
    def call(self) -> dict | None:
        """The first call of the final candidate, or ``None`` for no call."""
        return self.calls[0] if self.calls else None

    @property
    def no_call(self) -> bool:
        return self.calls == []

    @property
    def output(self) -> Any:
        """What the executed function returned (a list if several calls ran)."""
        if not self.outputs:
            return None
        return self.outputs[0] if len(self.outputs) == 1 else list(self.outputs)

    @property
    def attempts(self) -> list[dict]:
        """Each candidate with its verdict, scores and schema errors, oldest first."""
        rounds = []
        for step in self.trace:
            check = step.get("verification") or {}
            rounds.append({
                "calls": deepcopy(step["proposal"].get("calls")),
                "accepted": step.get("gate_pass"),
                "function_probability": check.get("function_probability"),
                "arguments_probability": check.get("arguments_probability"),
                "schema_errors": deepcopy((step.get("schema") or {}).get("errors", [])),
            })
        return rounds

    @property
    def probabilities(self) -> dict:
        """The decision model's probabilities of yes for the final candidate (``None`` if not asked)."""
        final = -1 if self.final_attempt is None else self.final_attempt
        check = (self.trace[final].get("verification") or {}) if self.trace else {}
        return {"function": check.get("function_probability"), "arguments": check.get("arguments_probability")}

    @property
    def usage(self) -> dict:
        """Token counts summed over every generator request and every decision-model request."""
        generator: dict = {}
        decision_model: dict = {}
        for step in self.trace:
            _add(generator, step["proposal"].get("usage"))
            _add(decision_model, (step.get("verification") or {}).get("usage"))
        return {"generator": generator, "decision_model": decision_model}

    def __repr__(self) -> str:
        if self.calls is None:
            decision = "unreadable candidate"
        elif not self.calls:
            decision = "no call"
        else:
            decision = "call=" + "; ".join(_format(call) for call in self.calls)
        parts = [decision, f"accepted={self.accepted}", f"repairs={self.repairs}"]
        if self.executed:
            parts.append(f"output={self.output!r}")
        return f"Result({', '.join(parts)})"


@dataclass
class Verdict:
    """The schema check's and the decision model's judgment of one candidate. Truthy if accepted."""

    accepted: bool
    calls: list[dict]
    function_probability: float | None
    arguments_probability: float | None
    schema_errors: list[dict]
    feedback: dict = field(repr=False)
    raw: dict = field(repr=False)

    def __bool__(self) -> bool:
        return self.accepted


def _verdict(decision_model, messages, calls, tools) -> Verdict:
    schemas, _ = toolset(tools)
    conversation = _conversation(messages)
    proposed = normalize_calls(calls)
    local = check_schema(schemas, proposed)
    check = decision_model.judge(conversation, schemas, proposed)
    accepted = bool(local["accepted"] and check["function_pass"]
                    and (check["arguments_skipped"] or check["arguments_pass"]))
    return Verdict(accepted, proposed, check["function_probability"], check["arguments_probability"],
                   local["errors"], check["feedback"], check)


def verify(messages, calls, tools, *, decision_model=None, threshold: float | None = None,
           jev_api_key: str | None = None) -> Verdict:
    """Check calls from any source before running them; nothing is generated or executed.

    ``messages`` is the conversation so far, including earlier tool calls and tool results.
    ``calls`` may be plain dicts or the tool-call objects of the OpenAI, Anthropic or Google
    SDKs. The calls pass if they pass the schema check and the decision model's function
    question and, unless there is no call, its argument question (0.5 by default).
    ``decision_model`` defaults to :class:`dmcall.Jev`.
    """
    if decision_model is not None:
        if jev_api_key is not None or threshold is not None:
            raise ValueError("Configure the decision model itself (api_key, threshold).")
        return _verdict(decision_model, messages, calls, tools)
    with Jev(jev_api_key, threshold=DEFAULT_THRESHOLD if threshold is None else threshold) as owned:
        return _verdict(owned, messages, calls, tools)


class DMCall:
    """Decision-model-guided function calling.

    >>> agent = DMCall("openai:gpt-5-nano", tools=[get_weather])   # OPENAI_API_KEY, TYPESAFE_API_KEY
    >>> agent.run("Is it warm in Seoul right now?")
    Result(call=get_weather(city="Seoul"), accepted=True, repairs=0, output={...})

    ``model`` is ``"provider:model"`` (openai, anthropic, gemini, xai, meta, azure, openrouter) or a
    provider object; other keyword options (``api_key=``, ``base_url=``, ``reasoning_effort=``,
    ...) go to the provider. ``decision_model`` defaults to :class:`dmcall.Jev` with
    ``threshold`` (0.5) for both questions; pass a configured :class:`dmcall.Jev` or
    :class:`dmcall.PPLXDecider` instead to change it.

    The defaults are the paper's settings: at most ``max_repairs=3`` repairs, both questions
    (``checks="both"``), the whole history (``history="cumulative"``), Hybrid feedback
    (``feedback="hybrid"``), the first candidate when none passes (``on_exhaustion="first"``)
    and a readable initial no-call returned without verification
    (``preserve_initial_no_call=True``). The paper's ablations are ``checks="function"``
    (DM-Call-Function), ``checks="arguments"`` (DM-Call-Argument) and ``history="latest"``
    (DM-Call-Latest); its feedback representations are ``"hybrid"``, ``"numeric"`` and ``"text"``.
    """

    def __init__(self, model, *, tools=None, functions: dict[str, Callable] | None = None, decision_model=None,
                 threshold: float | None = None, max_repairs: int = DEFAULT_MAX_REPAIRS, checks: str = "both",
                 history: str = "cumulative", feedback: str = "hybrid", on_exhaustion: str = "first",
                 preserve_initial_no_call: bool = True, execute_unverified: bool = False,
                 jev_api_key: str | None = None, **model_options):
        self.max_repairs = _repairs(max_repairs)
        self.checks = _choice("checks", checks, CHECKS)
        self.history = _choice("history", history, HISTORIES)
        self.feedback = _choice("feedback", feedback, FEEDBACK)
        self.on_exhaustion = _choice("on_exhaustion", on_exhaustion, ON_EXHAUSTION)
        if not isinstance(preserve_initial_no_call, bool):
            raise ValueError("preserve_initial_no_call must be True or False.")
        self.preserve_initial_no_call = preserve_initial_no_call
        self.execute_unverified = bool(execute_unverified)
        self.tools = list(tools) if isinstance(tools, (list, tuple)) else ([] if tools is None else [tools])
        self.functions = dict(functions or {})
        toolset(self.tools, self.functions)  # fail fast on a bad tool
        if decision_model is not None:
            if jev_api_key is not None or threshold is not None:
                raise ValueError("Configure the decision model itself (api_key, threshold), not DMCall.")
            if not callable(getattr(decision_model, "judge", None)):
                raise TypeError("decision_model must have judge(messages, tools, calls), like dmcall.Jev.")
        self.model = make_provider(model, **model_options)
        self._owned = [self.model] if isinstance(model, str) else []
        if decision_model is None:
            try:
                decision_model = Jev(jev_api_key, threshold=DEFAULT_THRESHOLD if threshold is None else threshold)
            except Exception:
                self.close()
                raise
            self._owned.append(decision_model)
        self.decision_model = decision_model

    # Public API ---------------------------------------------------------------------------
    def call(self, messages, tools=None, *, functions: dict[str, Callable] | None = None,
             max_repairs: int | None = None, on_exhaustion: str | None = None) -> Result:
        """Propose, verify and repair a call. Nothing is executed."""
        return self._solve(messages, tools, functions, max_repairs, on_exhaustion, execute=False, unverified=False)

    def run(self, messages, tools=None, *, functions: dict[str, Callable] | None = None,
            max_repairs: int | None = None, on_exhaustion: str | None = None,
            execute_unverified: bool | None = None) -> Result:
        """``call()``, then execute the final call with its Python function if it was accepted.

        With ``execute_unverified=True`` the final candidate also runs when none was accepted:
        the first one, or the last with ``on_exhaustion="last"``.
        """
        unverified = self.execute_unverified if execute_unverified is None else bool(execute_unverified)
        return self._solve(messages, tools, functions, max_repairs, on_exhaustion, execute=True, unverified=unverified)

    def verify(self, messages, calls, tools=None) -> Verdict:
        """Check a proposed call (dicts or SDK objects) with the schema check and both questions."""
        return _verdict(self.decision_model, messages, calls, self.tools if tools is None else tools)

    async def acall(self, messages, tools=None, **options) -> Result:
        """``call()`` in a worker thread, so async code is not blocked."""
        return await asyncio.to_thread(self.call, messages, tools, **options)

    async def arun(self, messages, tools=None, **options) -> Result:
        """``run()`` in a worker thread; the tool functions run in that thread too."""
        return await asyncio.to_thread(self.run, messages, tools, **options)

    async def averify(self, messages, calls, tools=None) -> Verdict:
        """``verify()`` in a worker thread."""
        return await asyncio.to_thread(self.verify, messages, calls, tools)

    def close(self) -> None:
        """Close the HTTP clients this object created."""
        for owned in getattr(self, "_owned", []):
            owned.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self) -> str:
        return (f"DMCall(model={self.model!r}, decision_model={self.decision_model!r}, checks={self.checks!r}, "
                f"history={self.history!r}, feedback={self.feedback!r}, max_repairs={self.max_repairs}, "
                f"on_exhaustion={self.on_exhaustion!r}, preserve_initial_no_call={self.preserve_initial_no_call})")

    # The loop ----------------------------------------------------------------------------
    def _accept(self, local: dict, check: dict) -> tuple[bool, dict]:
        """A_t (Eq. 2) for the configured questions, and the feedback record the generator gets."""
        if self.checks == "function":
            return bool(local["accepted"] and check["function_pass"]), single_check(check["feedback"], "function")
        arguments = check["arguments_skipped"] or check["arguments_pass"]
        if self.checks == "arguments":
            return bool(local["accepted"] and arguments), single_check(check["feedback"], "arguments")
        return bool(local["accepted"] and check["function_pass"] and arguments), check["feedback"]

    def _solve(self, messages, tools, functions, max_repairs, on_exhaustion, *, execute, unverified) -> Result:
        limit = self.max_repairs if max_repairs is None else _repairs(max_repairs)
        rule = self.on_exhaustion if on_exhaustion is None else _choice("on_exhaustion", on_exhaustion, ON_EXHAUSTION)
        schemas, registry = (toolset(self.tools, {**self.functions, **(functions or {})}) if tools is None
                             else toolset(tools, functions))
        example = {"messages": _messages(messages), "tools": schemas}
        started = time.perf_counter()
        history: list[dict] = []
        trace: list[dict] = []
        accepted = False
        for attempt in range(limit + 1):
            shown = history
            if self.history == "latest" and history:
                # Only the latest rejection, with its real round number.
                shown = [{**deepcopy(history[-1]), "round": attempt}]
            proposal = self.model.generate(example, represent_history(shown, self.feedback))
            calls = proposal["calls"]
            local = check_schema(schemas, calls)
            if attempt == 0 and self.preserve_initial_no_call and _readable_no_call(proposal) and local["accepted"]:
                # A readable initial no-call is returned without verification or repair.
                check = {"kind": "initial_no_call_preserved", "accepted": None, "function_probability": None,
                         "arguments_probability": None, "function_pass": None, "arguments_pass": None,
                         "arguments_skipped": True, "api_calls": 0, "usage": {}, "latency_s": 0.0, "feedback": {}}
                trace.append({"attempt": 0, "proposal": proposal, "verification": check, "schema": local,
                              "feedback": {}, "gate_pass": None})
                return Result(calls=[], accepted=None, exhausted=False, repairs=0, text=proposal.get("content"),
                              latency_s=time.perf_counter() - started, trace=trace, final_attempt=0,
                              initial_no_call_preserved=True)
            check = self.decision_model.judge(example["messages"], schemas, calls)
            accepted, feedback = self._accept(local, check)
            trace.append({"attempt": attempt, "proposal": proposal, "verification": check, "schema": local,
                          "feedback": deepcopy(feedback), "gate_pass": accepted})
            if accepted:
                break
            history.append({"calls": deepcopy(calls), "feedback": deepcopy(feedback)})

        exhausted = not accepted
        final = 0 if exhausted and rule == "first" else len(trace) - 1
        calls = deepcopy(trace[final]["proposal"]["calls"])
        outputs: list = []
        executed = False
        if execute and (accepted or unverified):
            executed = self._execute(calls, registry, outputs)
        return Result(calls=calls, accepted=accepted, exhausted=exhausted, repairs=len(trace) - 1,
                      text=trace[final]["proposal"].get("content"), executed=executed, outputs=outputs,
                      latency_s=time.perf_counter() - started, trace=trace, final_attempt=final)

    @staticmethod
    def _execute(calls, registry: dict, outputs: list) -> bool:
        """Run the final calls with their Python functions. An unreadable candidate never runs."""
        if not isinstance(calls, list) or not all(
                isinstance(call, dict) and isinstance(call.get("name"), str) and isinstance(call.get("arguments"), dict)
                for call in calls):
            return False
        missing = [call["name"] for call in calls if call["name"] not in registry]
        if missing:
            raise LookupError(f"No Python function for tool {missing[0]!r}; pass it in tools= or functions=.")
        for call in calls:
            outputs.append(invoke(registry[call["name"]], deepcopy(call["arguments"])))
        return bool(calls)
