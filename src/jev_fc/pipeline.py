"""Bounded proposal/verification state machine, independent of benchmark labels."""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Callable

from .jev import validate_calls

METHODS = ("direct", "schema_only", "self_refine", "jev_function", "jev_arguments", "jev_joint", "jev_joint_latest")


def run_example(
    example: dict,
    generator,
    verifier=None,
    method: str = "jev_joint",
    max_repairs: int = 3,
    exhaustion_policy: str = "execute_last",
    initial_proposal: dict | None = None,
    executor: Callable | None = None,
) -> dict:
    """Perform at most 1 + max_repairs proposals; execute exactly once at termination.

    max_repairs=0 still verifies the initial proposal for Jev methods. For the
    self_refine control, every allowed repair is used with a generic self-review
    instruction; it is a matched maximum-generation-budget control, not equal
    measured cost. No learned critic is used for schema_only/self_refine.
    jev_joint_latest sends only the last rejected record to the generator;
    its complete trace and all other joint-verification rules are unchanged.
    Benchmark mode leaves executor=None and records a simulated dispatch.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    if isinstance(max_repairs, bool) or not isinstance(max_repairs, int) or max_repairs < 0:
        raise ValueError("max_repairs must be a nonnegative integer")
    if exhaustion_policy not in {"execute_last", "abstain"}:
        raise ValueError("exhaustion_policy must be execute_last or abstain")
    if method.startswith("jev_") and verifier is None:
        raise ValueError("A Jev verifier is required for this method")
    # The state machine only receives public fields, even if called with a full dataset record.
    public = {k: deepcopy(example[k]) for k in ("id", "category", "messages", "tools") if k in example}
    history, trace = [], []
    start = time.perf_counter()
    budget = 0 if method == "direct" else max_repairs
    for attempt in range(budget + 1):
        prompt_history = history
        if method == "jev_joint_latest" and history:
            # Keep the real one-based proposal round, not its index in this
            # one-record prompt. Preserve every round in the internal trace.
            prompt_history = [{**deepcopy(history[-1]), "round": attempt}]
        proposal = deepcopy(initial_proposal) if attempt == 0 and initial_proposal is not None else generator.generate(public, prompt_history)
        calls = proposal["calls"]
        local = validate_calls(public, calls)
        if method.startswith("jev_"):
            check = verifier.verify(public, calls)
            if method == "jev_function":
                accepted = local["accepted"] and check["function_pass"]
                feedback = {"function_pass": check["function_pass"], "function_probability": check.get("function_probability"),
                            "structural_errors": local["errors"],
                            "instruction": "Reconsider whether the chosen function (or no-call decision) satisfies the request."}
            elif method == "jev_arguments":
                accepted = bool(local["accepted"] and (check["arguments_skipped"] or check["arguments_pass"]))
                feedback = {"arguments_pass": check["arguments_pass"], "arguments_probability": check.get("arguments_probability"),
                            "arguments_skipped": check["arguments_skipped"],
                            "structural_errors": local["errors"],
                            "instruction": "Reconsider the values, required arguments and constraints of the proposed call."}
            else:
                accepted = bool(local["accepted"] and check["function_pass"]
                                and (check["arguments_skipped"] or check["arguments_pass"]))
                feedback = check["feedback"]
        elif method == "schema_only":
            check = local
            accepted = local["accepted"]
            feedback = local["feedback"]
        elif method == "self_refine":
            check = {"kind": "generic_self_review", "accepted": None}
            accepted = attempt == budget
            feedback = {"instruction": "Independently review this proposal for wrong function selection, missing or unsupported arguments, and an unnecessary or missing call. Correct any errors; otherwise preserve it."}
        else:
            check = {"kind": "no_verifier", "accepted": None}
            accepted = True
            feedback = {}
        trace.append({"attempt": attempt, "proposal": proposal, "verification": check,
                      "schema": local, "feedback": deepcopy(feedback), "gate_pass": accepted})
        if accepted:
            break
        history.append({"calls": deepcopy(calls), "feedback": deepcopy(feedback)})
    exhausted = not accepted and attempt == budget
    dispatch = accepted or exhaustion_policy == "execute_last"
    # Only an explicit application-supplied callback can execute a real function.
    execution_result = executor(calls) if dispatch and executor is not None else None
    return {
        "id": example.get("id"), "category": example.get("category"), "method": method,
        "max_repairs": max_repairs, "exhaustion_policy": exhaustion_policy,
        "calls": calls, "accepted": accepted if method not in {"direct", "self_refine"} else None,
        "exhausted": exhausted, "dispatched": dispatch, "abstained": not dispatch,
        "dispatch_mode": "callback" if executor else "simulated",
        "execution_result": execution_result,
        "repairs": len(trace) - 1, "trace": trace,
        "wall_time_s": time.perf_counter() - start,
        "initial_proposal_shared": initial_proposal is not None,
        # Charge the shared initial proposal to each method when comparing standalone cost.
        "logical_latency_s": sum(t["proposal"].get("latency_s", 0) + t["verification"].get("latency_s", 0) for t in trace),
    }
