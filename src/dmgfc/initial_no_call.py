"""Initial no-call policy around the byte-preserved historical repair engine.

Only a readable, completely parsed initial ``calls == []`` is preserved. Other
candidates use the historical checks and chronological repair prompts unchanged.
The evaluation CLI still invokes that historical engine directly; new experiment
wrappers must explicitly record this policy and its source hash.
"""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Callable

from jev_fc.jev import validate_calls
from jev_fc.pipeline import METHODS, run_example as historical_run_example

POLICY_VERSION = "preserve-initial-no-call-v1"
_VERIFIED_METHODS = {"jev_function", "jev_arguments", "jev_joint", "jev_joint_latest"}


def run_example(
    example: dict,
    generator,
    verifier=None,
    method: str = "jev_joint",
    max_repairs: int = 3,
    exhaustion_policy: str = "execute_last",
    initial_proposal: dict | None = None,
    executor: Callable | None = None,
    *,
    preserve_initial_no_call: bool = True,
) -> dict:
    """Preserve initial no-call or delegate the exact remaining repair trajectory.

    ``accepted=None`` for the preserved no-call means no decision-model verdict
    was obtained. ``gate_pass=None`` and the explicit policy marker distinguish
    termination by this rule from an accepted semantic judgment. No callback is
    invoked, even when an executor was supplied.

    ``preserve_initial_no_call=False`` delegates directly to the frozen engine.
    Later no-calls, malformed calls, and unreadable ``calls=None`` are not covered
    by this policy. Direct, schema-only, and self-refine methods are unchanged.
    """
    if not isinstance(preserve_initial_no_call, bool):
        raise ValueError("preserve_initial_no_call must be a boolean.")
    if not preserve_initial_no_call or method not in _VERIFIED_METHODS:
        return historical_run_example(
            example, generator, verifier, method, max_repairs,
            exhaustion_policy, initial_proposal, executor,
        )

    # Preserve the historical validation order before making an initial request.
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    if isinstance(max_repairs, bool) or not isinstance(max_repairs, int) or max_repairs < 0:
        raise ValueError("max_repairs must be a nonnegative integer")
    if exhaustion_policy not in {"execute_last", "abstain"}:
        raise ValueError("exhaustion_policy must be execute_last or abstain")
    if verifier is None:
        raise ValueError("A Jev verifier is required for this method")
    public = {key: deepcopy(example[key]) for key in ("id", "category", "messages", "tools")
              if key in example}
    start = time.perf_counter()
    proposal = (deepcopy(initial_proposal) if initial_proposal is not None
                else generator.generate(public, []))
    calls = proposal["calls"]
    readable_empty = (isinstance(calls, list) and calls == []
                      and not proposal.get("incomplete") and not proposal.get("parse_errors"))
    if readable_empty:
        local = validate_calls(public, calls)
        if local["accepted"]:
            check = {
                "kind": "initial_no_call_preserved", "policy_version": POLICY_VERSION,
                "accepted": None, "function_probability": None, "arguments_probability": None,
                "function_pass": None, "arguments_pass": None,
                "function_skipped": True, "arguments_skipped": True,
                "api_calls": 0, "usage": {"input_tokens": 0, "output_tokens": 0},
                "latency_s": 0.0, "feedback": {},
            }
            trace = [{"attempt": 0, "proposal": deepcopy(proposal), "verification": check,
                      "schema": local, "feedback": {}, "gate_pass": None,
                      "policy_terminated": True}]
            return {
                "id": example.get("id"), "category": example.get("category"), "method": method,
                "max_repairs": max_repairs, "exhaustion_policy": exhaustion_policy,
                "calls": [], "accepted": None, "exhausted": False,
                "dispatched": True, "abstained": False, "dispatch_mode": "preserved_no_call",
                "execution_result": None, "executor_invoked": False,
                "repairs": 0, "trace": trace, "wall_time_s": time.perf_counter() - start,
                "initial_proposal_shared": initial_proposal is not None,
                "logical_latency_s": proposal.get("latency_s", 0),
                "initial_no_call_preserved": True, "initial_no_call_policy": POLICY_VERSION,
            }

    # Supplying the already generated candidate avoids a duplicate initial call.
    record = historical_run_example(
        example, generator, verifier, method, max_repairs,
        exhaustion_policy, proposal, executor,
    )
    record["wall_time_s"] = time.perf_counter() - start
    record["initial_proposal_shared"] = initial_proposal is not None
    record["initial_no_call_preserved"] = False
    record["initial_no_call_policy"] = POLICY_VERSION
    return record
