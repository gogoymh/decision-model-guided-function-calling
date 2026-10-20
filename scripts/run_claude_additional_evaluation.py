"""Claude BFCL evaluation with explicit native-schema compatibility accounting.

The canonical tools and all example IDs stay unchanged. An actual HTTP 400 for
an input with a property key outside the provider's documented pattern becomes
an unreadable proposal, counted as an error. Other errors keep the frozen
runner's stop behavior. --run is still required for network requests.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_additional_evaluation as runner  # noqa: E402
from jev_fc.llm import LLMAPIError  # noqa: E402

PROPERTY_PATTERN = r"^[a-zA-Z0-9_.-]{1,64}$"
PROVIDER_INPUT_RULE = (
    "Canonical BFCL schemas and IDs unchanged. After an actual Claude HTTP 400, an input containing "
    "a schema property key outside ^[a-zA-Z0-9_.-]{1,64}$ is recorded as calls=None, "
    "incomplete=unsupported_schema and scored as an error; other errors stop. No case is excluded."
)
ORIGINAL_MAKE_CLIENTS = runner.make_clients
ORIGINAL_SETTINGS_FOR = runner.settings_for


def adapter_settings():
    return {"provider_adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "provider_input_rule": PROVIDER_INPUT_RULE}


def unsupported_property_names(example):
    invalid = []

    def walk(schema):
        if isinstance(schema, dict):
            properties = schema.get("properties")
            if isinstance(properties, dict):
                invalid.extend(name for name in properties if not re.fullmatch(PROPERTY_PATTERN, name))
            for value in schema.values():
                walk(value)
        elif isinstance(schema, list):
            for value in schema:
                walk(value)

    for tool in example.get("tools", []):
        walk(tool.get("function", {}).get("parameters", {}))
    return invalid


def wrap_generator(generator):
    if generator.provider != "anthropic" or generator.api_style != "anthropic_messages":
        raise ValueError("The native-schema adapter supports only Claude Anthropic Messages generators.")
    original_generate = generator.generate

    def generate(example, history=None):
        invalid = unsupported_property_names(example)
        started = time.perf_counter()
        # The normal 400 has one attempt and no retry delay. The adapter does not
        # patch shared module waits: concurrent requests keep the frozen behavior.
        try:
            return original_generate(example, history)
        except LLMAPIError as exc:
            if exc.status_code != 400 or not invalid:
                raise
            elapsed = max(0.0, time.perf_counter() - started)
            # The frozen exception exposes attempt count but not retry durations.
            # A multi-attempt failure cannot provide exact API/retry separation.
            attempts = exc.attempts if isinstance(exc.attempts, int) and exc.attempts > 0 else 1
            return {
                "calls": None, "content": None,
                "parse_errors": ["The provider rejected a canonical schema property key unsupported by its native interface."],
                "incomplete": "unsupported_schema", "model": None, "requested_model": generator.model,
                "usage": {}, "raw_usage": None, "latency_s": elapsed, "retry_wait_s": 0.0,
                "api_calls": attempts, "gateway": generator.gateway, "api_style": generator.api_style,
                "provider": generator.provider, "provider_adapter": "claude-native-schema-v1",
                "provider_input_error": {"status_code": 400, "constraint": PROPERTY_PATTERN,
                                         "unsupported_property_count": len(invalid),
                                         "request_sent": True, "usage_reported": False,
                                         "retry_timing_known": attempts == 1,
                                         "timing_basis": "Measured failed-request wall duration; retry separation unavailable if attempts exceed one"},
            }

    generator.generate = generate
    return generator


def make_clients(args, *, preflight=False):
    generator, verifier = ORIGINAL_MAKE_CLIENTS(args, preflight=preflight)
    try:
        wrap_generator(generator)
    except BaseException:
        generator.close()
        verifier.close()
        raise
    return generator, verifier


def settings_for(args, examples, provenance, generator, verifier):
    return {**ORIGINAL_SETTINGS_FOR(args, examples, provenance, generator, verifier), **adapter_settings()}


@contextmanager
def patched_runner():
    previous_clients, previous_settings = runner.make_clients, runner.settings_for
    runner.make_clients, runner.settings_for = make_clients, settings_for
    try:
        yield
    finally:
        runner.make_clients, runner.settings_for = previous_clients, previous_settings


def main(argv=None):
    args = runner.parser().parse_args(argv)
    if (args.benchmark != "bfcl" or args.api_style != "anthropic_messages"
            or not args.model.startswith("claude")):
        raise ValueError("Use this compatibility wrapper only for BFCL with a Claude Anthropic Messages model.")
    with patched_runner():
        return runner.main(argv)


if __name__ == "__main__":
    main()
