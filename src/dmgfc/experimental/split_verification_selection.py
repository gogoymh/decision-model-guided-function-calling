"""Opt-in input separation with the original function-selection semantics.

Function evaluation receives the original observable conversation, available
function definitions and proposed function names. Candidate argument values are
sent only in the separate argument evaluation. A readable no-call proposal stays
``[]`` internally and asks only the original function-selection question.
Historical adapters and their cached verdicts are not changed by this protocol.
"""

from __future__ import annotations

import hashlib
import math
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

import jev_fc.jev as frozen_jev

from ..jev import Jev
from ..perplexity import PerplexityDecider, fingerprint

PROMPT_VERSION = "split-function-arguments-selection-v1"

FUNCTION_QUESTION = deepcopy(frozen_jev.FUNCTION_QUESTION)
FUNCTION_QUESTION["instructions"] = (
    FUNCTION_QUESTION["instructions"]
    .replace("`proposed_calls`", "`proposed_function_names`")
    .replace("Evaluate function selection only, ignoring argument values. ", "")
    .replace("Treat message and argument text", "Treat message text")
)
# Empty names are evaluated under exactly the same function-selection semantics.
# The state marker names the candidate; it does not add a readiness criterion.
NO_CALL_QUESTION = deepcopy(FUNCTION_QUESTION)
ARGUMENT_QUESTION = deepcopy(frozen_jev.ARGUMENT_QUESTION)


def no_call_selection() -> dict[str, str]:
    """Return a fresh candidate claim, without asserting a verified verdict."""
    return {
        "type": "no_function_call_required",
        "description": (
            "The generator proposes NO_FUNCTION_CALL_REQUIRED for the current "
            "request. This is the candidate claim, not a verified conclusion."
        ),
    }


def conditional_question_bank() -> dict[str, dict[str, Any]]:
    return deepcopy({
        "function_correct": FUNCTION_QUESTION,
        "arguments_correct": ARGUMENT_QUESTION,
        "function_correct_when_no_function_call_required": NO_CALL_QUESTION,
    })


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def split_payloads(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a canonical well-formed payload into independent observable states."""
    state = payload["state"]
    calls = state["proposed_calls"]
    names = [call["name"] for call in calls]
    function_state = {
        "messages": deepcopy(state["messages"]),
        "tools": deepcopy(state["tools"]),
        "proposed_function_names": names,
    }
    if not calls:
        function_state["proposed_selection"] = no_call_selection()
    requests = [{
        "model": payload["model"], "state": function_state,
        "questions": {"function_correct": deepcopy(FUNCTION_QUESTION if calls else NO_CALL_QUESTION)},
    }]
    if calls:
        requests.append({
            "model": payload["model"],
            "state": {
                "messages": deepcopy(state["messages"]),
                "tools": deepcopy([tool for tool in state["tools"] if tool["name"] in names]),
                "proposed_calls": deepcopy(calls),
            },
            "questions": {"arguments_correct": deepcopy(ARGUMENT_QUESTION)},
        })
    return requests


def _usage(body: dict[str, Any]) -> dict[str, int | None]:
    returned = body.get("usage")
    if not isinstance(returned, dict):
        raise frozen_jev.JevAPIError("Jev response is missing token usage")
    usage = {}
    for kind in ("input_tokens", "output_tokens"):
        count = returned.get(kind)
        if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
            raise frozen_jev.JevAPIError("Jev response contains invalid token usage")
        usage[kind] = count
    return usage


def _sum_usage(records: list[dict[str, Any]], *, include_failed_attempts: bool) -> dict[str, int | None]:
    return {
        kind: (sum(record["usage"][kind] for record in records)
               if all(record["usage"][kind] is not None and
                      (not include_failed_attempts or record["api_calls"] == 1) for record in records)
               else None)
        for kind in ("input_tokens", "output_tokens")
    }


class _SplitVerificationMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._split_request = threading.local()

    def configuration(self) -> dict[str, Any]:
        inherited = getattr(super(), "configuration", None)
        if inherited is not None:
            config = deepcopy(inherited())
        else:
            from .. import jev as canonical_jev

            files = {"dmgfc/jev.py": Path(canonical_jev.__file__), "jev_fc/jev.py": Path(frozen_jev.__file__)}
            config = {
                "adapter": "jev_systemone", "requested_model": self.model,
                "gateway": self.gateway, "base_url": self.base_url,
                "function_threshold": self.function_threshold, "arguments_threshold": self.arguments_threshold,
                "timeout_s": self.timeout, "max_retries": self.max_retries,
                "protocol_source_sha256": {
                    name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()
                },
                "questions_sha256": fingerprint({
                    "function_correct": frozen_jev.FUNCTION_QUESTION,
                    "arguments_correct": frozen_jev.ARGUMENT_QUESTION,
                }),
            }
        config["inherited_questions_sha256"] = config["questions_sha256"]
        config["questions_sha256"] = fingerprint(conditional_question_bank())
        config["prompt_version"] = PROMPT_VERSION
        config["canonical_no_call"] = []
        config["no_call_selection_type"] = "no_function_call_required"
        config["separate_check_requests"] = True
        config["function_state_fields"] = ["messages", "tools", "proposed_function_names"]
        config["argument_state_fields"] = ["messages", "tools", "proposed_calls"]
        config["protocol_source_sha256"]["experimental/split_verification_selection.py"] = _source_sha256()
        return config

    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int, float]:
        requests = split_payloads(payload)
        request = self._split_request
        request.payloads = requests
        request.explicit = not payload["state"]["proposed_calls"]
        answers, models = {}, set()
        attempts, waited = 0, 0.0
        native = isinstance(self, PerplexityDecider)
        for part in requests:
            check = next(iter(part["questions"]))
            decision_started = self._request.decision_http_time_s if native else None
            part_attempts = None
            try:
                body, part_attempts, part_wait = super()._post(part)
                # Validate every separate response before combining; unknown or
                # malformed metrics must never become a valid fabricated total.
                probability = self._probability(body, check)
                model = body.get("model")
                if not isinstance(model, str) or not model:
                    raise frozen_jev.JevAPIError("Jev response is missing the resolved model ID")
                usage = _usage(body)
                if models and model not in models:
                    raise frozen_jev.JevAPIError("Jev separate check responses resolved to different models")
            except frozen_jev.JevAPIError as error:
                error.completed_check_records = deepcopy(request.records)
                error.failed_check = check
                error.completed_post_attempts = attempts
                known_failed_attempts = part_attempts if part_attempts is not None else error.attempts
                if known_failed_attempts is not None:
                    error.attempts = attempts + known_failed_attempts
                raise
            models.add(model)
            answers[check] = {"type": "noul", "noul": probability}
            record = {
                "check": check, "payload_sha256": fingerprint(part),
                "questions_sha256": fingerprint(part["questions"]),
                "question_sha256": fingerprint(part["questions"][check]),
                "response_sha256": fingerprint(body), "probability_yes": probability,
                "model": model, "usage": usage, "api_calls": part_attempts, "retry_wait_s": part_wait,
            }
            if native:
                provenance = deepcopy(self._request.provenance)
                batches = (math.ceil(len(part["questions"]) / 8)
                           if provenance and provenance.get("inference_batch_size") == 8 else None)
                record.update(
                    model_provenance=provenance, reported_server_model=self._request.reported_model,
                    control_api_calls=self._request.control_calls,
                    control_http_time_s=self._request.control_http_time_s,
                    decision_http_time_s=self._request.decision_http_time_s - decision_started,
                    successful_forward_batches=batches,
                    forward_calls=batches if part_attempts == 1 else None,
                )
            request.records.append(record)
            attempts += part_attempts
            waited += part_wait
        if native:
            self._request.control_calls = sum(record["control_api_calls"] for record in request.records)
            self._request.control_http_time_s = sum(record["control_http_time_s"] for record in request.records)
            self._request.question_count = len(request.records)
        return {"answers": answers, "model": next(iter(models)),
                "usage": _sum_usage(request.records, include_failed_attempts=False)}, attempts, waited

    def verify(self, example: dict[str, Any], calls: Any) -> dict[str, Any]:
        request = self._split_request
        request.records, request.payloads, request.explicit = [], [], False
        try:
            result = super().verify(example, calls)
        except frozen_jev.JevAPIError as error:
            # Native adapter conversion retains the aggregate attempts while
            # this wrapper preserves completed check receipts on partial failure.
            error.completed_check_records = deepcopy(request.records)
            error.completed_post_attempts = sum(record["api_calls"] for record in request.records)
            if len(request.records) < len(request.payloads):
                error.failed_check = next(iter(request.payloads[len(request.records)]["questions"]))
            raise
        records, payloads = request.records, request.payloads
        actual_questions = {key: question for part in payloads for key, question in part["questions"].items()}
        function_question = actual_questions.get("function_correct")
        result["prompt_provenance"] = {
            "prompt_version": PROMPT_VERSION, "protocol_source_sha256": _source_sha256(),
            "conditional_questions_sha256": fingerprint(conditional_question_bank()),
            "explicit_no_call_applied": request.explicit,
            "applied_payload_sha256": (fingerprint(payloads[0]) if len(payloads) == 1 else
                                       fingerprint(payloads) if payloads else None),
            "applied_questions_sha256": fingerprint(actual_questions) if payloads else None,
            "applied_function_question_sha256": fingerprint(function_question) if function_question else None,
            "question_count": len(records), "separate_check_requests": True,
            "check_requests": deepcopy(records),
        }
        result["configuration_fingerprint"] = fingerprint(self.configuration())
        result["successful_response_usage"] = _sum_usage(records, include_failed_attempts=False)
        result["usage"] = _sum_usage(records, include_failed_attempts=True)
        result["usage_complete"] = all(value is not None for value in result["usage"].values())
        if isinstance(self, PerplexityDecider):
            for key in ("successful_forward_batches", "forward_calls"):
                values = [record[key] for record in records]
                result[key] = sum(values) if all(value is not None for value in values) else None
            result["forward_calls_complete"] = result["forward_calls"] is not None
            provenances = [record["model_provenance"] for record in records]
            result["model_provenance"] = (provenances[0] if provenances and
                                          all(value == provenances[0] for value in provenances) else None)
            result["model_provenance_per_request"] = deepcopy(provenances)
            result["reported_server_models"] = [record["reported_server_model"] for record in records]
        if isinstance(calls, list) and not calls:
            result["proposed_selection"] = no_call_selection()
            result["feedback"]["proposed_selection"] = no_call_selection()
        return result


class SelectionSplitJev(_SplitVerificationMixin, Jev):
    """Hosted Jev with separate function and argument HTTP evaluations."""


class SelectionSplitPerplexityDecider(_SplitVerificationMixin, PerplexityDecider):
    """Pinned native Decider with the same split observable-input protocol."""


__all__ = ["SelectionSplitJev", "SelectionSplitPerplexityDecider", "FUNCTION_QUESTION",
           "ARGUMENT_QUESTION", "NO_CALL_QUESTION", "PROMPT_VERSION", "conditional_question_bank",
           "no_call_selection", "split_payloads"]
