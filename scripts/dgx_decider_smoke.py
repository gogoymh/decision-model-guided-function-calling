#!/usr/bin/env python3
"""Bounded fictional protocol smoke; no generator, tool execution or benchmark claims."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dmgfc.perplexity import DeciderError, PerplexityDecider, fingerprint  # noqa: E402
from dmgfc import DMGFC  # noqa: E402

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get weather for the city named by the user.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"], "additionalProperties": False},
}}


def toy_cases() -> list[dict]:
    """Expected labels are outside the observable examples sent to the model."""
    return [
        {"id": "fictional-correct-call", "request": "What is the weather in Seoul?",
         "calls": [{"name": "get_weather", "arguments": {"city": "Seoul"}}],
         "expected_function": True, "expected_arguments": True},
        {"id": "fictional-wrong-argument", "request": "What is the weather in Seoul?",
         "calls": [{"name": "get_weather", "arguments": {"city": "Busan"}}],
         "expected_function": True, "expected_arguments": False},
        {"id": "fictional-missing-argument", "request": "What is the weather in Seoul?",
         "calls": [{"name": "get_weather", "arguments": {}}],
         "expected_function": True, "expected_arguments": False},
        {"id": "fictional-correct-no-call", "request": "Say hello. Do not request weather.",
         "calls": [], "expected_function": True, "expected_arguments": None},
        {"id": "fictional-missing-call", "request": "What is the weather in Seoul?",
         "calls": [], "expected_function": False, "expected_arguments": None},
    ]


def run_smoke(verifier: PerplexityDecider, cases: list[dict] | None = None) -> dict:
    records = []
    for case in toy_cases() if cases is None else cases:
        observable = {"messages": [{"role": "user", "content": case["request"]}], "tools": [WEATHER]}
        record = verifier.verify(observable, case["calls"])
        records.append({"id": case["id"], "synthetic": True, "observable_input_sha256": fingerprint(observable),
                        "proposed_calls_sha256": fingerprint(case["calls"]), "verification": record,
                        "expected_function": case["expected_function"],
                        "expected_arguments": case["expected_arguments"]})
    # A deterministic missing-argument candidate must fail the local schema
    # gate, then the scripted generator supplies the requested city. The script
    # is not an LLM and never executes the weather function.
    class ScriptedGenerator:
        def __init__(self, *calls):
            self.calls, self.generated = list(calls), 0

        def generate(self, _example, _history):
            calls = self.calls.pop(0)
            self.generated += 1
            return {"calls": calls, "content": None, "usage": {"input_tokens": 0, "output_tokens": 0},
                    "latency_s": 0.0, "api_calls": 0, "source": "fictional_scripted_generator"}

    scripted = ScriptedGenerator([{ "name": "get_weather", "arguments": {}}],
                                 [{"name": "get_weather", "arguments": {"city": "Seoul"}}])
    repaired = DMGFC(scripted, jev=verifier, tools=[WEATHER], max_repairs=1).call("What is the weather in Seoul?")
    no_call_generator = ScriptedGenerator([])
    no_call = DMGFC(no_call_generator, jev=verifier, tools=[WEATHER], max_repairs=0).call(
        "Say hello. Do not request weather.")
    trajectories = [{"name": "fictional-repair-loop", "calls": repaired.calls, "accepted": repaired.accepted,
                     "repairs": repaired.repairs, "trace": repaired.trace, "executed": repaired.executed},
                    {"name": "fictional-no-call-loop", "calls": no_call.calls, "accepted": no_call.accepted,
                     "repairs": no_call.repairs, "trace": no_call.trace, "executed": no_call.executed}]
    checks = [row["verification"] for row in records] + [step["verification"] for trajectory in trajectories
                                                         for step in trajectory["trace"]]
    configuration = verifier.configuration()
    def known_total(field):
        values = [check[field] for check in checks]
        return sum(values) if all(value is not None for value in values) else None
    return {"format_version": "dgx-decider-toy-smoke-v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "synthetic": True, "purpose": "protocol smoke, not a benchmark or paper result",
            "configuration": configuration, "configuration_fingerprint": fingerprint(configuration),
            "records": records, "scripted_trajectories": trajectories,
            "summary": {"cases": len(records), "generator_calls": 0,
                "scripted_generator_calls": scripted.generated + no_call_generator.generated,
                "tool_executions": 0, "api_calls": known_total("api_calls"),
                "control_api_calls": known_total("control_api_calls"),
                "question_evaluations": known_total("question_evaluations"),
                "forward_calls": known_total("forward_calls"),
                "wall_time_s": known_total("wall_time_s"), "estimated_usd": None,
                "scripted_repair_accepted": repaired.accepted,
                "scripted_no_call_accepted": no_call.accepted}}


def write_new_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18081")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", required=True, type=Path, help="New JSON receipt; existing files are never replaced.")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new receipt path.")
    try:
        # No .env is loaded. Authentication is optional and explicitly named.
        with PerplexityDecider(base_url=args.base_url, timeout=args.timeout,
                              api_key=os.getenv("DGX_DECIDER_API_KEY"), max_retries=0) as verifier:
            receipt = run_smoke(verifier)
        write_new_json(args.output, receipt)
    except DeciderError as error:
        print(json.dumps({"status": "failed", "error": error.code,
                          "http_status": error.status_code, "attempts": error.attempts}), file=sys.stderr)
        return 1
    print(json.dumps({"status": "protocol_smoke_completed", "synthetic": True,
                      "output": str(args.output), "summary": receipt["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
