"""Decision models: D in the paper.

A decision model answers two yes/no questions about a candidate call and returns the
probability of *yes* for each: the function question D_f (is this the right function, or is
making no call right?) and the argument question D_a (are the arguments present, supported by
the request and in the right parameters?). The argument question is not asked for a no call,
and neither question is asked for an unreadable candidate; such a score is ``None`` (⊥).

Both decision models of the paper speak the same HTTP protocol (``POST /v1/systemone``):

* :class:`Jev`, TypeSafe's hosted decision model (``jev-1.13.0``), and
* :class:`PPLXDecider`, the open-weight Perplexity Decider v1.1 27B served on your own machine.

Only the request, the function schemas and the candidate are sent; never a reference answer.
To plug in another decision model, subclass :class:`DecisionModel` and override :meth:`ask`.
"""

from __future__ import annotations

import copy
import math
import os
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx

from .prompts import ARGUMENT_QUESTION, FUNCTION_QUESTION, REPAIR_INSTRUCTION
from .schema import check_schema, function_definitions

__all__ = ["DecisionModel", "DecisionModelError", "Jev", "PPLXDecider"]

RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 529})
_OBSERVABLE_KEYS = ("role", "content", "name", "tool_calls", "tool_call_id")


class DecisionModelError(RuntimeError):
    """A transport or protocol failure. It is never a rejection, and it never holds keys or bodies."""

    def __init__(self, message: str, *, code: str = "decision_model_error", status_code: int | None = None,
                 request_id: str | None = None, attempts: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.request_id = request_id
        self.attempts = attempts


def _base_url(value: str | None, owner: str) -> str:
    url = (value or "").strip().rstrip("/")
    parts = urlsplit(url)
    if (parts.scheme not in {"http", "https"} or not parts.netloc or parts.username is not None
            or parts.password is not None or parts.query or parts.fragment):
        raise ValueError(f"{owner} needs an http(s) base_url without credentials, query or fragment.")
    return url


def _threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Thresholds must be numbers between 0 and 1.")
    return float(value)


def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
    if response is not None:
        value = response.headers.get("Retry-After")
        if value is not None:
            try:
                delay = float(value)
            except ValueError:
                try:
                    stamp = parsedate_to_datetime(value)
                    if stamp.tzinfo is None:
                        stamp = stamp.replace(tzinfo=timezone.utc)
                    delay = (stamp - datetime.now(timezone.utc)).total_seconds()
                except (TypeError, ValueError, OverflowError):
                    delay = -1
            if math.isfinite(delay) and delay >= 0:
                return min(delay, 30.0)
    return min(2.0 ** attempt, 30.0)


class DecisionModel:
    """A decision model reached over the ``/v1/systemone`` protocol.

    ``threshold`` is the pass mark of both questions (0.5 in the paper); ``function_threshold``
    and ``arguments_threshold`` override one of them. ``max_retries`` counts HTTP retries after
    the first attempt and has nothing to do with DM-Call's repair limit. Pass ``client=`` (for
    example an ``httpx.Client`` with ``httpx.MockTransport``) to control transport; the caller
    then owns it.
    """

    name = "Decision model"
    # The ``source`` field of the feedback records the generator reads.
    feedback_source = "decision_model_noul_and_deterministic_schema_validation"

    def __init__(self, base_url: str, *, model: str, api_key: str | None = None, threshold: float = 0.5,
                 function_threshold: float | None = None, arguments_threshold: float | None = None,
                 timeout: float = 60.0, max_retries: int = 3, client: httpx.Client | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.function_threshold = _threshold(threshold if function_threshold is None else function_threshold)
        self.arguments_threshold = _threshold(threshold if arguments_threshold is None else arguments_threshold)
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer.")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive number of seconds.")
        if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()
                                    or "\n" in api_key or "\r" in api_key):
            raise ValueError("API keys must be nonempty single-line strings.")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"{self.name} needs a model name.")
        self.base_url = _base_url(base_url, type(self).__name__)
        self.model = model
        self.timeout = float(timeout)
        self.max_retries = max_retries
        self._headers = {"Content-Type": "application/json"}
        if api_key is not None:
            self._headers = {"Authorization": "Bearer " + api_key, **self._headers}
        self._client = client or httpx.Client(timeout=self.timeout)
        self._owns_client = client is None
        self._sleep = sleep

    # HTTP --------------------------------------------------------------------------------
    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(model={self.model!r}, base_url={self.base_url!r}, "
                f"function_threshold={self.function_threshold}, arguments_threshold={self.arguments_threshold})")

    def _request_id(self, response: httpx.Response) -> str | None:
        """A narrow diagnostic ID, never an echoed credential."""
        secrets = {value.removeprefix("Bearer ") for key, value in self._headers.items()
                   if key.lower() == "authorization"}
        for header in ("x-request-id", "request-id", "apim-request-id"):
            value = response.headers.get(header)
            if value and re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", value) and not any(s in value for s in secrets):
                return value
        return None

    def _error(self, response: httpx.Response, attempts: int) -> DecisionModelError:
        return DecisionModelError(f"{self.name} HTTP {response.status_code}; no judgment recorded.",
                                  code="http_error", status_code=response.status_code,
                                  request_id=self._request_id(response), attempts=attempts)

    def _wait(self, delay: float) -> float:
        started = time.perf_counter()
        self._sleep(delay)
        return time.perf_counter() - started

    def _post(self, payload: dict) -> tuple[dict, int, float]:
        """POST with bounded retries. Returns the body, the attempts and the time spent waiting."""
        waited = 0.0
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(self.base_url + "/v1/systemone", headers=self._headers,
                                             json=payload, timeout=self.timeout)
            except httpx.TransportError:
                if attempt < self.max_retries:
                    waited += self._wait(_retry_delay(None, attempt))
                    continue
                raise DecisionModelError(f"{self.name} connection failed after {attempt + 1} attempts.",
                                         code="connection_failed", attempts=attempt + 1) from None
            if response.status_code in RETRY_STATUSES and attempt < self.max_retries:
                waited += self._wait(_retry_delay(response, attempt))
                continue
            if not 200 <= response.status_code < 300:
                raise self._error(response, attempt + 1)
            try:
                body = response.json()
            except ValueError:
                raise DecisionModelError(f"{self.name} response was not valid JSON.", code="response_invalid") from None
            if not isinstance(body, dict):
                raise DecisionModelError(f"{self.name} response must be a JSON object.", code="response_invalid")
            return self._checked(body), attempt + 1, waited
        raise AssertionError("unreachable")

    def _checked(self, body: dict) -> dict:
        """Hook for a subclass to check the response's model identity."""
        return body

    # The protocol ------------------------------------------------------------------------
    def ask(self, state: dict, questions: dict) -> dict:
        """Ask ``questions`` about ``state``; return the probabilities of yes, model, usage and timing.

        ``state`` holds ``messages``, ``tools`` and ``proposed_calls``. Override this method to
        connect a decision model that speaks another protocol.
        """
        body, api_calls, waited = self._post({"model": self.model, "state": state, "questions": questions})
        answers = body.get("answers")
        probabilities = {}
        for key in questions:
            answer = answers.get(key) if isinstance(answers, dict) else None
            if not isinstance(answer, dict) or answer.get("type") != "noul":
                raise DecisionModelError(f"{self.name} response is missing the answer {key!r}.",
                                         code="response_invalid")
            value = answer.get("noul")
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or not 0 <= value <= 1):
                raise DecisionModelError(f"{self.name} returned an invalid probability for {key!r}.",
                                         code="response_invalid")
            probabilities[key] = float(value)
        model = body.get("model")
        if not isinstance(model, str) or not model:
            raise DecisionModelError(f"{self.name} response is missing the resolved model ID.", code="response_invalid")
        usage = body.get("usage")
        if not isinstance(usage, dict):
            raise DecisionModelError(f"{self.name} response is missing token usage.", code="response_invalid")
        counts = {}
        for kind in ("input_tokens", "output_tokens"):
            count = usage.get(kind)
            if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
                raise DecisionModelError(f"{self.name} response contains invalid token usage.",
                                         code="response_invalid")
            counts[kind] = count
        return {"probabilities": probabilities, "model": model, "usage": counts,
                "api_calls": api_calls, "retry_wait_s": waited}

    def judge(self, messages: list[dict], tools: list[dict], calls: Any) -> dict:
        """Judge one candidate: the schema check S, both questions, verdicts V and the feedback record.

        ``messages`` is the conversation (dicts with ``role`` and ``content``, plus ``tool_calls``
        or ``tool_call_id`` for earlier tool turns), ``tools`` the function definitions and
        ``calls`` the candidate as ``[{"name", "arguments"}]`` (``[]`` for no call).
        """
        started = time.perf_counter()
        local = check_schema(tools, calls)
        arguments_skipped = isinstance(calls, list) and not calls
        function_probability = arguments_probability = None
        usage: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}
        resolved_model, api_calls, waited = None, 0, 0.0
        if local["well_formed"]:  # an unreadable candidate is not sent to the decision model
            if not isinstance(messages, list):
                raise ValueError("messages must be a list")
            observable = []
            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError("Each message must be an object")
                observable.append({key: copy.deepcopy(message[key]) for key in _OBSERVABLE_KEYS if key in message})
            state = {"messages": observable, "tools": function_definitions(tools),
                     "proposed_calls": [{"name": call["name"], "arguments": copy.deepcopy(call["arguments"])}
                                        for call in calls]}
            questions = {"function_correct": copy.deepcopy(FUNCTION_QUESTION)}
            if not arguments_skipped:
                questions["arguments_correct"] = copy.deepcopy(ARGUMENT_QUESTION)
            answer = self.ask(state, questions)
            function_probability = answer["probabilities"]["function_correct"]
            if not arguments_skipped:
                arguments_probability = answer["probabilities"]["arguments_correct"]
            resolved_model, usage = answer["model"], answer["usage"]
            api_calls, waited = answer["api_calls"], answer["retry_wait_s"]

        # V(p) is true iff p is not null and p >= threshold; S must pass too.
        function_pass = (local["function_pass"] and function_probability is not None
                         and function_probability >= self.function_threshold)
        arguments_pass = None if arguments_skipped else (
            local["arguments_pass"] and arguments_probability is not None
            and arguments_probability >= self.arguments_threshold)
        checks = {
            "function_selection": {"pass": function_pass, "probability_yes": function_probability,
                                   "threshold": self.function_threshold, "schema_pass": local["function_pass"]},
            "arguments": {"skipped": arguments_skipped, "pass": arguments_pass,
                          "probability_yes": arguments_probability, "threshold": self.arguments_threshold,
                          "schema_pass": local["arguments_pass"]},
        }
        feedback = {
            "source": self.feedback_source,
            "checks": checks,
            "failed_checks": [name for name, check in checks.items()
                              if not check.get("skipped", False) and not check["pass"]],
            "schema_errors": local["errors"],
            "instruction": REPAIR_INSTRUCTION,
        }
        return {
            "function_pass": function_pass,
            "arguments_pass": arguments_pass,
            "arguments_skipped": arguments_skipped,
            "function_probability": function_probability,
            "arguments_probability": arguments_probability,
            "accepted": function_pass and (arguments_skipped or arguments_pass),
            "feedback": feedback,
            "schema_errors": local["errors"],
            "usage": usage,
            "model": resolved_model,
            "requested_model": self.model,
            "latency_s": max(0.0, time.perf_counter() - started - waited),
            "retry_wait_s": waited,
            "api_calls": api_calls,
        }


class Jev(DecisionModel):
    """TypeSafe's hosted decision model. Key: ``api_key=`` or ``TYPESAFE_API_KEY``.

    The default model is ``jev-1.13.0``, the one used in the paper (``TYPESAFE_MODEL`` or
    ``model=`` changes it), and the default endpoint is ``https://api.typesafe.ai``
    (``TYPESAFE_BASE_URL`` or ``base_url=``).
    """

    name = "Jev"
    feedback_source = "jev_noul_and_deterministic_schema_validation"
    DEFAULT_MODEL = "jev-1.13.0"
    DEFAULT_BASE_URL = "https://api.typesafe.ai"

    def __init__(self, api_key: str | None = None, *, model: str | None = None, base_url: str | None = None,
                 threshold: float = 0.5, function_threshold: float | None = None,
                 arguments_threshold: float | None = None, timeout: float = 60.0, max_retries: int = 3,
                 client: httpx.Client | None = None, sleep: Callable[[float], None] = time.sleep) -> None:
        api_key = api_key or os.getenv("TYPESAFE_API_KEY", "").strip() or None
        if not api_key:
            raise ValueError("Jev needs an API key: pass api_key= or set TYPESAFE_API_KEY.")
        super().__init__(base_url or os.getenv("TYPESAFE_BASE_URL") or self.DEFAULT_BASE_URL,
                         model=model or os.getenv("TYPESAFE_MODEL") or self.DEFAULT_MODEL, api_key=api_key,
                         threshold=threshold, function_threshold=function_threshold,
                         arguments_threshold=arguments_threshold, timeout=timeout, max_retries=max_retries,
                         client=client, sleep=sleep)


class PPLXDecider(DecisionModel):
    """Perplexity Decider v1.1 27B, an open-weight decision model, served on your own machine.

    ``base_url`` is the origin of a server running the checkpoint
    (``perplexity-ai/pplx-decider-v1.1-27b`` at revision ``14c9d901...``) with its own native
    inference code, which serves ``/v1/systemone``. A generic causal chat server is not this
    model. Plain HTTP is accepted only for ``localhost``; use an SSH tunnel or HTTPS otherwise.

    With ``check_provenance=True`` the client first reads ``GET /provenance`` and stops unless the
    server reports the pinned checkpoint, revision, attention mode and calibration temperature.
    The paper's runs used that check through a small deployment wrapper; the checkpoint's own
    server does not provide the route, so it is off by default.
    """

    name = "PPLX Decider"
    feedback_source = "perplexity_decider_noul_and_deterministic_schema_validation"
    MODEL_ID = "perplexity-ai/pplx-decider-v1.1-27b"
    MODEL_REVISION = "14c9d90137962d300f3de95633f7b6535e80c5ce"
    CALIBRATION_TEMPERATURE = 1.0087417621345625
    MAX_INPUT_TOKENS = 8192
    # The checkpoint's server reports this inherited name and accepts "jev-latest" as the request model.
    SERVER_MODEL = "autojev-qwen3.8-27b"
    WIRE_MODEL = "jev-latest"

    def __init__(self, base_url: str, *, api_key: str | None = None, threshold: float = 0.5,
                 function_threshold: float | None = None, arguments_threshold: float | None = None,
                 timeout: float = 180.0, max_retries: int = 0, check_provenance: bool = False,
                 client: httpx.Client | None = None, sleep: Callable[[float], None] = time.sleep) -> None:
        parts = urlsplit(base_url or "")
        if parts.path not in {"", "/"}:
            raise ValueError("PPLXDecider base_url must be an origin without a path.")
        if parts.scheme == "http" and parts.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Use HTTPS, or an SSH tunnel to localhost, for a remote PPLX Decider.")
        if not isinstance(check_provenance, bool):
            raise ValueError("check_provenance must be True or False.")
        super().__init__(base_url, model=self.WIRE_MODEL, api_key=api_key, threshold=threshold,
                         function_threshold=function_threshold, arguments_threshold=arguments_threshold,
                         timeout=timeout, max_retries=max_retries,
                         client=client or httpx.Client(timeout=timeout, trust_env=False), sleep=sleep)
        self._owns_client = client is None
        self.check_provenance = check_provenance
        self._provenance: dict | None = None

    def provenance(self) -> dict:
        """The server's reported checkpoint, verified against the pinned one."""
        try:
            response = self._client.get(self.base_url + "/provenance", headers=self._headers, timeout=self.timeout)
        except httpx.TransportError:
            raise DecisionModelError("PPLX Decider provenance could not be reached.", code="connection_failed") from None
        if response.status_code != 200:
            raise DecisionModelError("PPLX Decider provenance is unavailable.", code="provenance_unavailable",
                                     status_code=response.status_code)
        try:
            body = response.json()
        except ValueError:
            raise DecisionModelError("PPLX Decider provenance was not valid JSON.", code="provenance_invalid") from None
        if not isinstance(body, dict):
            raise DecisionModelError("PPLX Decider provenance must be an object.", code="provenance_invalid")
        expected = {"model_id": self.MODEL_ID, "revision": self.MODEL_REVISION,
                    "native_attention": "noncausal_full_attention", "max_input_tokens": self.MAX_INPUT_TOKENS}
        temperature = body.get("temperature")
        if (any(body.get(key) != value for key, value in expected.items())
                or isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                or not math.isclose(temperature, self.CALIBRATION_TEMPERATURE, rel_tol=0, abs_tol=1e-12)):
            raise DecisionModelError("The server does not run the pinned PPLX Decider checkpoint.",
                                     code="provenance_mismatch")
        return {**expected, "temperature": float(temperature)}

    def ask(self, state: dict, questions: dict) -> dict:
        if self.check_provenance and self._provenance is None:
            self._provenance = self.provenance()
        return super().ask(state, questions)

    def _error(self, response: httpx.Response, attempts: int) -> DecisionModelError:
        error = super()._error(response, attempts)
        if response.status_code == 422:
            try:
                if "8192-token limit" in str(response.json().get("detail", "")):
                    error.code = "context_limit_exceeded"
            except (ValueError, AttributeError):
                pass
        return error

    def _checked(self, body: dict) -> dict:
        if body.get("model") not in {self.MODEL_ID, self.SERVER_MODEL}:
            raise DecisionModelError("PPLX Decider response has an unexpected model ID.", code="model_mismatch")
        if "revision" in body and body["revision"] != self.MODEL_REVISION:
            raise DecisionModelError("PPLX Decider response has an unexpected model revision.", code="model_mismatch")
        # Missing usage is unknown, never zero.
        return {**body, "model": self.MODEL_ID, "usage": body.get("usage", {"input_tokens": None, "output_tokens": None})}

    def __repr__(self) -> str:
        return (f"PPLXDecider(model={self.MODEL_ID!r}, revision={self.MODEL_REVISION[:8]!r}, base_url={self.base_url!r}, "
                f"function_threshold={self.function_threshold}, arguments_threshold={self.arguments_threshold})")
