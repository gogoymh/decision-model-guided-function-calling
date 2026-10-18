"""JBFC: a model proposes one function call (or none), Jev checks it before it runs,
and the same model repairs it with the accumulated feedback until it passes or the
repair limit is reached. The loop is the paper's own implementation, ``jev_fc``."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

from jev_fc.pipeline import run_example

from .jev import Jev, validate_calls
from .providers import provider as make_provider
from .tools import _plain, invoke, normalize_calls, toolset

__all__ = ["JBFC", "METHODS", "Result", "Verdict", "verify"]

# Public method names and the paper's names for them.
METHODS = {
    "jbfc": "jev_joint",              # JBFC: function and argument checks, cumulative feedback
    "jbfc-latest": "jev_joint_latest",  # JBFC-L: only the latest rejection as feedback
    "jev-function": "jev_function",   # only the function check
    "jev-arguments": "jev_arguments",  # only the argument check
    "schema": "schema_only",          # only the local schema check, no Jev
    "self-refine": "self_refine",     # the model reviews itself for every allowed repair
    "direct": "direct",               # one proposal, no checks
}
_PUBLIC = {internal: public for public, internal in METHODS.items()}
# The paper's final settings: Jev's pass mark for both checks, and which proposal is final when none passes.
DEFAULT_THRESHOLD = 0.5
ON_EXHAUSTION = ("first", "last")
_ROLES = {"system", "developer", "user", "assistant"}
_HISTORY_ROLES = _ROLES | {"tool", "function"}
_HISTORY_KEYS = ("role", "content", "name", "tool_calls", "tool_call_id")
_TEXT_PARTS = {"text", "input_text", "output_text"}


def _method(name) -> str:
    key = str(name).strip().lower()
    if key in _PUBLIC:
        return key
    key = key.replace("_", "-")
    if key in METHODS:
        return METHODS[key]
    raise ValueError(f"Unknown method {name!r}; choose from {', '.join(METHODS)}.")


def _repairs(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("max_repairs must be a nonnegative integer.")
    return value


def _on_exhaustion(value) -> str:
    if value not in ON_EXHAUSTION:
        raise ValueError(f"on_exhaustion must be one of {', '.join(map(repr, ON_EXHAUSTION))}.")
    return value


def _messages(messages) -> list[dict]:
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


def _history(messages) -> list[dict]:
    """An agent's chat history as Jev sees it: text turns, earlier tool calls and tool results.

    Messages may be dicts or SDK message objects. Only role, content, name, tool_calls and
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


def _format(call) -> str:
    if not isinstance(call, dict):
        return "<malformed call>"
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        return f"{call.get('name')}(<malformed arguments>)"
    return f"{call.get('name')}(" + ", ".join(f"{key}={json.dumps(value, ensure_ascii=False)}"
                                             for key, value in arguments.items()) + ")"


def _model_usage(usage) -> dict:
    """Token counts from jbfc providers or from jev_fc's Chat Completions style usage."""
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("prompt_tokens_details")
    return {
        "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
        "cached_input_tokens": usage.get("cached_input_tokens",
                                         details.get("cached_tokens") if isinstance(details, dict) else None),
        "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
    }


def _add(total: dict, counts: dict) -> None:
    for key, value in counts.items():
        if isinstance(value, int) and not isinstance(value, bool):
            total[key] = total.get(key, 0) + value


@dataclass
class Result:
    """The outcome of one JBFC request.

    ``calls`` is the final proposal as ``[{"name", "arguments"}]``: ``[]`` is a decision to
    make no call, and ``None`` an unreadable proposal (cut off, filtered or refused).
    ``accepted`` says whether it passed the checks (``None`` for methods without a
    verifier). When no proposal passed within the repair limit (``exhausted``), the final
    proposal is the first one by default (``on_exhaustion="first"``, as in the paper) or the
    last one; ``final_attempt`` is its index in ``trace``, which keeps every proposal, verdict
    and feedback in order.
    """

    calls: list[dict] | None
    accepted: bool | None
    exhausted: bool
    repairs: int
    method: str
    text: str | None = None
    executed: bool = False
    outputs: list = field(default_factory=list)
    latency_s: float = 0.0
    trace: list[dict] = field(default_factory=list, repr=False)
    final_attempt: int | None = None

    @property
    def call(self) -> dict | None:
        """The first proposed call, or None for no call."""
        return self.calls[0] if self.calls else None

    @property
    def no_call(self) -> bool:
        return self.calls == []

    @property
    def output(self) -> Any:
        """The executed function's return value (a list if several calls ran)."""
        if not self.outputs:
            return None
        return self.outputs[0] if len(self.outputs) == 1 else list(self.outputs)

    @property
    def attempts(self) -> list[dict]:
        """Each proposal with its verdict, oldest first."""
        verified = self.method not in {"direct", "self-refine"}
        rounds = []
        for step in self.trace:
            check = step.get("verification") or {}
            rounds.append({
                "calls": deepcopy(step["proposal"].get("calls")),
                "accepted": step.get("gate_pass") if verified else None,
                "function_probability": check.get("function_probability"),
                "arguments_probability": check.get("arguments_probability"),
                "schema_errors": deepcopy((step.get("schema") or {}).get("errors", [])),
            })
        return rounds

    @property
    def probabilities(self) -> dict:
        """Jev's probabilities of yes for the final proposal (None where a check did not run)."""
        final = -1 if self.final_attempt is None else self.final_attempt
        check = (self.trace[final].get("verification") or {}) if self.trace else {}
        return {"function": check.get("function_probability"), "arguments": check.get("arguments_probability")}

    @property
    def usage(self) -> dict:
        """Token counts summed over every proposal (model) and every check (jev)."""
        model: dict = {}
        jev: dict = {}
        for step in self.trace:
            _add(model, _model_usage(step["proposal"].get("usage")))
            check = step.get("verification") or {}
            if isinstance(check.get("usage"), dict):
                _add(jev, check["usage"])
        return {"model": model, "jev": jev}

    def __repr__(self) -> str:
        if self.calls is None:
            decision = "unreadable proposal"
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
    """Jev's and the schema check's judgment of a proposed call collection. Truthy if accepted."""

    accepted: bool
    calls: list[dict]
    function_probability: float | None
    arguments_probability: float | None
    schema_errors: list[dict]
    feedback: dict = field(repr=False)
    raw: dict = field(repr=False)

    def __bool__(self) -> bool:
        return self.accepted


def _verify(jev, messages, calls, tools) -> Verdict:
    schemas, _ = toolset(tools)
    example = {"messages": _history(messages), "tools": schemas}
    proposed = normalize_calls(calls)
    local = validate_calls(example, proposed)
    check = jev.verify(example, proposed)
    accepted = bool(local["accepted"] and check["function_pass"]
                    and (check["arguments_skipped"] or check["arguments_pass"]))
    return Verdict(accepted, proposed, check["function_probability"], check["arguments_probability"],
                   local["errors"], check["feedback"], check)


def verify(messages, calls, tools, *, jev: Jev | None = None, jev_api_key: str | None = None,
           threshold: float | None = None) -> Verdict:
    """Check a call from any source (dicts or OpenAI, Anthropic and Gemini SDK objects) before running it.

    ``messages`` is the conversation so far, including earlier tool calls and results. The
    call passes if it is well formed for the tools and Jev's function check, plus its
    argument check when there is a call, reach the threshold (0.5 by default, as in the paper).
    """
    if jev is not None:
        if jev_api_key is not None or threshold is not None:
            raise ValueError("Configure the Jev instance itself (api_key, threshold).")
        return _verify(jev, messages, calls, tools)
    with Jev(jev_api_key, threshold=DEFAULT_THRESHOLD if threshold is None else threshold) as owned:
        return _verify(owned, messages, calls, tools)


class JBFC:
    """Verified function calling with any model.

    >>> agent = JBFC("openai:gpt-5-nano", tools=[get_weather])  # OPENAI_API_KEY, TYPESAFE_API_KEY
    >>> agent.run("How warm is it in Seoul right now?")
    Result(call=get_weather(city="Seoul"), accepted=True, repairs=0, output={...})

    ``model`` is ``"provider:model"`` (openai, anthropic, gemini, xai, azure, openrouter) or a
    provider object; extra keyword options (api_key=, base_url=, ...) go to the provider.
    ``max_repairs`` bounds the repairs after the first proposal (3, as in the paper's tables).
    ``threshold`` is Jev's pass mark for both checks (0.5, as in the paper). When no proposal
    passes within the limit, ``on_exhaustion`` picks the final one: ``"first"`` (the paper's
    rule: a repair replaces the model's first answer only once it passes) or ``"last"``.
    """

    def __init__(self, model, *, tools=None, functions: dict[str, Callable] | None = None,
                 jev: Jev | None = None, method: str = "jbfc", max_repairs: int = 3,
                 threshold: float | None = None, on_exhaustion: str = "first", execute_unverified: bool = False,
                 jev_api_key: str | None = None, **model_options):
        self.method = _method(method)
        self.max_repairs = _repairs(max_repairs)
        self.on_exhaustion = _on_exhaustion(on_exhaustion)
        self.execute_unverified = bool(execute_unverified)
        self.tools = list(tools) if isinstance(tools, (list, tuple)) else ([] if tools is None else [tools])
        self.functions = dict(functions or {})
        toolset(self.tools, self.functions)  # fail fast on a bad tool
        if jev is not None and (jev_api_key is not None or threshold is not None):
            raise ValueError("Configure the Jev instance itself (api_key, threshold), not JBFC.")
        needs_jev = self.method.startswith("jev_")
        if threshold is not None and jev is None and not needs_jev and jev_api_key is None:
            raise ValueError(f"threshold applies to Jev; method {_PUBLIC[self.method]!r} does not use it.")
        self.model = make_provider(model, **model_options)
        self._owned = [self.model] if isinstance(model, str) else []
        try:
            if jev is None and (needs_jev or jev_api_key is not None):
                jev = Jev(jev_api_key, threshold=DEFAULT_THRESHOLD if threshold is None else threshold)
                self._owned.append(jev)
        except Exception:
            self.close()
            raise
        self.jev = jev

    def call(self, messages, tools=None, *, functions: dict[str, Callable] | None = None,
             max_repairs: int | None = None, on_exhaustion: str | None = None) -> Result:
        """Propose, check and repair a call. Nothing is executed."""
        return self._solve(messages, tools, functions, max_repairs, execute=False, unverified=False,
                           on_exhaustion=on_exhaustion)

    def run(self, messages, tools=None, *, functions: dict[str, Callable] | None = None,
            max_repairs: int | None = None, execute_unverified: bool | None = None,
            on_exhaustion: str | None = None) -> Result:
        """``call()``, then execute the final call with its Python function if it passed the checks.

        With ``execute_unverified=True`` the final call runs even if no proposal passed (the
        paper's benchmark policy): the first proposal, or the last with ``on_exhaustion="last"``.
        Methods without a verifier always run.
        """
        unverified = self.execute_unverified if execute_unverified is None else bool(execute_unverified)
        return self._solve(messages, tools, functions, max_repairs, execute=True, unverified=unverified,
                           on_exhaustion=on_exhaustion)

    def verify(self, messages, calls, tools=None) -> Verdict:
        """Check any proposed call (dicts or SDK objects) with Jev and the schema check."""
        if self.jev is None:
            raise ValueError("verify() needs Jev: use a Jev method, or pass jev= or jev_api_key=.")
        return _verify(self.jev, messages, calls, self.tools if tools is None else tools)

    async def acall(self, messages, tools=None, **options) -> Result:
        """``call()`` in a worker thread, so async code is not blocked."""
        return await asyncio.to_thread(self.call, messages, tools, **options)

    async def arun(self, messages, tools=None, **options) -> Result:
        """``run()`` in a worker thread; the tool functions run in that thread too."""
        return await asyncio.to_thread(self.run, messages, tools, **options)

    async def averify(self, messages, calls, tools=None) -> Verdict:
        """``verify()`` in a worker thread."""
        return await asyncio.to_thread(self.verify, messages, calls, tools)

    def _toolset(self, tools, functions):
        if tools is None:
            return toolset(self.tools, {**self.functions, **(functions or {})})
        return toolset(tools, functions)

    def _solve(self, messages, tools, functions, max_repairs, *, execute, unverified, on_exhaustion=None) -> Result:
        rule = self.on_exhaustion if on_exhaustion is None else _on_exhaustion(on_exhaustion)
        schemas, registry = self._toolset(tools, functions)
        example = {"messages": _messages(messages), "tools": schemas}
        outputs: list = []
        executed: list = []

        def executor(calls):
            if not isinstance(calls, list) or not all(
                    isinstance(call, dict) and isinstance(call.get("name"), str)
                    and isinstance(call.get("arguments"), dict) for call in calls):
                return None  # an unreadable or malformed proposal never runs
            missing = [call["name"] for call in calls if call["name"] not in registry]
            if missing:
                raise LookupError(f"No Python function for tool {missing[0]!r}; pass it in tools= or functions=.")
            for call in calls:
                executed.append(call)
                outputs.append(invoke(registry[call["name"]], deepcopy(call["arguments"])))
            return outputs

        # The loop runs a passing call (or, without a verifier, its final call); nothing else runs inside it.
        record = run_example(example, self.model, self.jev, method=self.method,
                             max_repairs=self.max_repairs if max_repairs is None else _repairs(max_repairs),
                             exhaustion_policy="abstain", executor=executor if execute else None)
        trace, final, calls = record["trace"], len(record["trace"]) - 1, record["calls"]
        if record["exhausted"]:  # no proposal passed within the limit
            if rule == "first":
                final, calls = 0, trace[0]["proposal"]["calls"]
            if execute and unverified:
                executor(calls)
        return Result(calls=deepcopy(calls), accepted=record["accepted"], exhausted=record["exhausted"],
                      repairs=record["repairs"], method=_PUBLIC[self.method],
                      text=trace[final]["proposal"].get("content"), executed=bool(executed), outputs=outputs,
                      latency_s=record["wall_time_s"], trace=trace, final_attempt=final)

    def close(self) -> None:
        """Close the HTTP clients this object created."""
        for owned in getattr(self, "_owned", []):
            owned.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self) -> str:
        return (f"JBFC(model={self.model!r}, method={_PUBLIC[self.method]!r}, max_repairs={self.max_repairs}, "
                f"on_exhaustion={self.on_exhaustion!r}, jev={self.jev!r})")
