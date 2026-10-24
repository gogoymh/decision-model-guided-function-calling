"""Offline preparation and real frozen repair-loop transport with fake HTTP endpoints."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from dmfc.perplexity import (CALIBRATION_TEMPERATURE, INHERITED_SERVER_MODEL,
                             MODEL_ID, MODEL_REVISION, PerplexityDecider)
from jev_fc.benchmarks import BFCL_REVISION
from jev_fc.llm import FunctionCallingLLM

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_dgx_decider_experiment", ROOT / "scripts/run_dgx_decider_experiment.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

TOOL = {"type": "function", "function": {"name": "weather", "description": "Weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
RIGHT = [{"name": "weather", "arguments": {"city": "Seoul"}}]
WRONG = [{"name": "weather", "arguments": {"city": "Busan"}}]
PROVENANCE = {"model_id": MODEL_ID, "revision": MODEL_REVISION, "native_attention": "noncausal_full_attention",
              "temperature": CALIBRATION_TEMPERATURE, "max_input_tokens": 8192,
              "inference_batch_size": 8, "one_branch_per_question": True}


def inputs(tmp_path):
    dataset, manifest, cache, source_manifest, records, checker = [tmp_path / filename for filename in (
        "dataset.jsonl", "data-manifest.json", "initial.jsonl", "source-manifest.json", "records.jsonl", "checker.py")]
    example = {"id": "offline-case-1", "category": "simple_python", "dataset_revision": BFCL_REVISION,
               "messages": [{"role": "user", "content": "Weather in Seoul?"}], "tools": [TOOL],
               "ground_truth": RIGHT, "bfcl_functions": []}
    proposal = {"calls": WRONG, "usage": {"prompt_tokens": 100, "completion_tokens": 10,
                                            "prompt_tokens_details": {"cached_tokens": 25}},
                "latency_s": 1.0, "api_calls": 1, "model": "test-generator", "requested_model": "test-generator"}
    dataset.write_text(json.dumps(example) + "\n")
    manifest.write_text(json.dumps({"revision": BFCL_REVISION,
                                   "normalized_sha256": {dataset.name: runner.sha256_file(dataset)}}))
    cache.write_text(json.dumps({"id": example["id"], "replicate": 0, "proposal": proposal}) + "\n")
    source_manifest.write_text(json.dumps({"config_id": "old-config", "llm_model": "test-generator",
        "dataset_sha256": runner.sha256_file(dataset), "source_sha256": "historical-source-hash",
        "llm_api_style": "chat_completions", "llm_gateway": "direct", "llm_base_url": "https://generator.invalid/v1",
        "llm_endpoint": "https://generator.invalid/v1/chat/completions", "reasoning_effort": "none",
        "temperature": None, "max_completion_tokens": 4096}))
    records.write_text(json.dumps({"id": example["id"], "replicate": 0, "method": "direct", "config_id": "old-config",
                                  "trace": [{"proposal": proposal}]}) + "\n")
    checker.write_text("# isolated offline test checker\n")
    args = SimpleNamespace(dataset=dataset, dataset_manifest=manifest, initials=cache,
        generator_manifest=source_manifest, source_records=records, checker=checker,
        base_url="http://127.0.0.1:18081", limit=1, replicate=0, max_repairs=3,
        max_new_generator_calls=3, output=tmp_path / "new-output", execute=False)
    return args, example


def fake_native(seen, *, always_reject=False, fail_after_repair=False):
    count = 0
    def handler(request):
        nonlocal count
        if request.url.path == "/provenance":
            return httpx.Response(200, json=PROVENANCE)
        body = json.loads(request.content)
        seen.append(body)
        count += 1
        if fail_after_repair and count > 1:
            return httpx.Response(500, json={"detail": "private-key-must-not-leak"})
        correct = body["state"]["proposed_calls"] == RIGHT
        return httpx.Response(200, json={"model": INHERITED_SERVER_MODEL,
            "answers": {"function_correct": {"type": "noul", "noul": 0.9},
                        "arguments_correct": {"type": "noul", "noul": 0.9 if correct and not always_reject else 0.1}},
            "usage": {"input_tokens": 120, "output_tokens": 0}})
    return PerplexityDecider(client=httpx.Client(transport=httpx.MockTransport(handler)))


def fake_generator(seen):
    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"model": "test-generator", "choices": [{"finish_reason": "tool_calls",
            "message": {"role": "assistant", "tool_calls": [{"type": "function", "id": "fake-call",
                "function": {"name": "weather", "arguments": json.dumps({"city": "Seoul"})}}]}}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 8, "prompt_tokens_details": {"cached_tokens": 5}}})
    return FunctionCallingLLM(api_key="offline-test-key", model="test-generator", api_style="chat_completions",
        gateway="direct", base_url="https://generator.invalid/v1", max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)))


def preflight(args, monkeypatch):
    monkeypatch.setattr(runner, "load_official_checker", lambda _path: lambda *_args: None)
    return runner.preflight(args)


def test_default_preflight_verifies_actual_cache_and_makes_zero_network_requests(tmp_path, monkeypatch):
    args, _example = inputs(tmp_path)
    originals = {path: path.read_bytes() for path in [args.dataset, args.initials, args.source_records]}
    monkeypatch.setattr(httpx.Client, "get", lambda *_args, **_kwargs: pytest.fail("Offline preflight cannot GET"))
    monkeypatch.setattr(httpx.Client, "post", lambda *_args, **_kwargs: pytest.fail("Offline preflight cannot POST"))
    protocol, examples, initials = preflight(args, monkeypatch)
    assert protocol["expected_records"] == 2 and protocol["source_differs_from_initial_run"] is True
    assert protocol["threshold"] == 0.5 and protocol["final_rule"] == "keep_first"
    assert protocol["generator"]["max_retries"] == 0 and protocol["workers"] == 1
    assert len(examples) == len(initials) == 1
    assert originals == {path: path.read_bytes() for path in originals}
    with pytest.raises(ValueError, match="--execute"):
        runner.execute(args, protocol, examples, initials)
    assert not args.output.exists()


def test_fake_http_repair_loop_scores_only_after_model_calls_and_accounts_initial_once(tmp_path, monkeypatch):
    args, example = inputs(tmp_path)
    protocol, examples, initials = preflight(args, monkeypatch)
    args.execute = True
    native_calls, generator_calls, score_calls = [], [], []
    native, generator = fake_native(native_calls), fake_generator(generator_calls)
    def score(item, calls):
        assert len(native_calls) == 2 and len(generator_calls) == 1
        score_calls.append((item["id"], deepcopy(calls)))
        return {"correct": calls == RIGHT, "metric": "isolated-test-score"}
    summary = runner.execute(args, protocol, examples, initials, verifier=native, generator=generator, scorer=score)
    rows = runner.read_rows(args.output / "records.jsonl")
    direct, repaired = rows
    assert direct["calls"] == WRONG and repaired["calls"] == RIGHT
    assert repaired["repairs"] == 1 and repaired["accepted"] is True
    assert repaired["trace"][0]["proposal"] == direct["trace"][0]["proposal"] == initials[example["id"]]["proposal"]
    assert summary["new_generator_requests_attempted"] == 1 and summary["tool_executions"] == 0
    assert direct["accounting"]["standalone_generator_input_tokens"] == 100
    assert repaired["accounting"]["standalone_generator_input_tokens"] == 140
    assert repaired["accounting"]["standalone_generator_cached_input_tokens"] == 30
    assert repaired["accounting"]["standalone_generator_output_tokens"] == 18
    assert repaired["accounting"]["new_generator_input_tokens"] == 40
    assert repaired["accounting"]["new_generator_api_calls"] == 1
    assert repaired["accounting"]["native_input_tokens"] == 240
    assert repaired["accounting"]["native_forward_calls"] == 2
    assert repaired["accounting"]["estimated_usd"] is None
    assert len(runner.read_rows(args.output / "repair-generations.jsonl")) == 1
    assert len(runner.read_rows(args.output / "native-verifications.jsonl")) == 2
    assert score_calls
    for payload in native_calls + generator_calls:
        serialized = json.dumps(payload)
        assert "ground_truth" not in serialized and "offline-case-1" not in serialized and "isolated-test-score" not in serialized
    with pytest.raises(FileExistsError):
        runner.execute(args, protocol, examples, initials, verifier=native, generator=generator, scorer=score)


def test_exhaustion_returns_same_initial_and_preserves_all_repair_attempts(tmp_path, monkeypatch):
    args, _example = inputs(tmp_path)
    args.max_repairs = 1
    protocol, examples, initials = preflight(args, monkeypatch)
    args.execute = True
    runner.execute(args, protocol, examples, initials, verifier=fake_native([], always_reject=True),
                   generator=fake_generator([]), scorer=lambda *_args: {"correct": False})
    repaired = runner.read_rows(args.output / "records.jsonl")[1]
    assert repaired["final_attempt"] == 0 and repaired["calls"] == WRONG
    assert repaired["exhausted"] is True and repaired["trace"][1]["proposal"]["calls"] == RIGHT


def test_failure_keeps_generated_proposals_and_sanitizes_provider_bodies(tmp_path, monkeypatch):
    args, _example = inputs(tmp_path)
    protocol, examples, initials = preflight(args, monkeypatch)
    args.execute = True
    with pytest.raises(RuntimeError, match="sanitized"):
        runner.execute(args, protocol, examples, initials, verifier=fake_native([], fail_after_repair=True),
                       generator=fake_generator([]), scorer=lambda *_args: pytest.fail("Failed case cannot be scored"))
    assert len(runner.read_rows(args.output / "repair-generations.jsonl")) == 1
    assert len(runner.read_rows(args.output / "native-verifications.jsonl")) == 1
    error = runner.read_rows(args.output / "errors.jsonl")[0]
    assert error["phase"] == "model_operations" and error["failed_attempt_token_usage"] is None
    assert "private-key" not in json.dumps(error)


def test_generator_request_budget_stops_before_extra_paid_request():
    generator = SimpleNamespace(generate=lambda *_args: {"calls": []})
    budget = runner.RequestBudget(generator, 1)
    budget.generate({}, [])
    with pytest.raises(RuntimeError, match="cap"):
        budget.generate({}, [])
    assert budget.attempted == 1


def test_unknown_usage_is_not_zero_and_alias_schemas_are_supported():
    base = {"trace": [{"proposal": {"usage": {}, "api_calls": 1}, "verification": {"kind": "no_verifier"}}]}
    metrics = runner.accounting(base)
    assert metrics["standalone_generator_input_tokens"] is None
    assert metrics["standalone_generator_cached_input_tokens"] is None
    assert metrics["native_api_calls"] == 0  # no verifier was used, rather than missing usage
    base["trace"][0]["proposal"]["usage"] = {"input_tokens": 10, "output_tokens": 3, "cached_input_tokens": 2}
    metrics = runner.accounting(base)
    assert metrics["standalone_generator_input_tokens"] == 10
    assert metrics["standalone_generator_output_tokens"] == 3
    assert metrics["standalone_generator_cached_input_tokens"] == 2
    base["trace"][0]["proposal"]["usage"]["input_tokens"] = 10.5
    assert runner.accounting(base)["standalone_generator_input_tokens"] is None


def test_wrong_dataset_or_source_fingerprint_blocks_preflight(tmp_path, monkeypatch):
    args, _example = inputs(tmp_path)
    manifest = json.loads(args.dataset_manifest.read_text())
    manifest["revision"] = "wrong-revision"
    args.dataset_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="revision/checksum"):
        preflight(args, monkeypatch)
