"""Native backend protocol and provenance gates; every HTTP operation is mocked."""

import copy
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

import dmgfc
from dmgfc.perplexity import (CALIBRATION_TEMPERATURE, DeciderError, INHERITED_SERVER_MODEL,
                             MODEL_ID, MODEL_REVISION, PerplexityDecider, fingerprint)
from jev_fc.jev import ARGUMENT_QUESTION, FUNCTION_QUESTION, JevVerifier

ROOT = Path(__file__).resolve().parents[1]
TOOL = {"type": "function", "function": {"name": "weather", "description": "Weather.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
CALL = [{"name": "weather", "arguments": {"city": "Seoul"}}]
EXAMPLE = {"messages": [{"role": "user", "content": "Weather in Seoul?", "ground_truth": "hidden"}],
           "tools": [TOOL], "id": "hidden-id", "category": "hidden-category", "ground_truth": CALL}
PROVENANCE = {"model_id": MODEL_ID, "revision": MODEL_REVISION, "native_attention": "noncausal_full_attention",
              "temperature": CALIBRATION_TEMPERATURE, "max_input_tokens": 8192,
              "inference_batch_size": 8, "one_branch_per_question": True}


def response(p=0.9, *, arguments=0.9, usage=True, model=INHERITED_SERVER_MODEL):
    body = {"model": model, "answers": {"function_correct": {"type": "noul", "noul": p},
                                           "arguments_correct": {"type": "noul", "noul": arguments}}}
    if usage:
        body["usage"] = {"input_tokens": 120, "output_tokens": 0}
    return body


def transport_handler(post_body=None, provenance=None, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(request)
        if request.url.path == "/provenance":
            return httpx.Response(200, json=PROVENANCE if provenance is None else provenance)
        assert request.url.path == "/v1/systemone"
        return httpx.Response(200, json=response() if post_body is None else post_body)
    return handler


def verifier(handler=None, **kwargs):
    client = httpx.Client(transport=httpx.MockTransport(handler or transport_handler()))
    return PerplexityDecider(client=client, **kwargs)


def test_explicit_export_keeps_default_jev_and_no_environment_routes(monkeypatch):
    monkeypatch.setenv("API_GATEWAY", "apim")
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://wrong.invalid")
    monkeypatch.setenv("TYPESAFE_MODEL", "wrong-model")
    instance = dmgfc.PerplexityDecider()
    assert instance.base_url == "http://127.0.0.1:18081" and instance.gateway == "native_systemone"
    assert dmgfc.Jev is not dmgfc.PerplexityDecider
    assert isinstance(instance, JevVerifier)
    assert "wrong-model" not in repr(instance)
    instance.close()


def test_same_questions_observable_input_and_threshold_gate_as_jev():
    seen = []
    instance = verifier(transport_handler(response(arguments=0.2), seen=seen))
    result = instance.verify(EXAMPLE, CALL)
    payload = json.loads(seen[-1].content)
    assert payload["model"] == "jev-latest"
    assert payload["questions"] == {"function_correct": FUNCTION_QUESTION, "arguments_correct": ARGUMENT_QUESTION}
    assert payload["state"] == {"messages": [{"role": "user", "content": "Weather in Seoul?"}],
                               "tools": [TOOL["function"]], "proposed_calls": CALL}
    assert result["function_pass"] is True and result["arguments_pass"] is False
    assert result["feedback"]["failed_checks"] == ["arguments"]
    assert result["model"] == MODEL_ID and result["model_provenance"]["revision"] == MODEL_REVISION
    assert result["reported_server_model"] == INHERITED_SERVER_MODEL
    assert result["api_calls"] == 1 and result["control_api_calls"] == 1
    assert result["question_evaluations"] == 2 and result["forward_calls"] == 1
    assert result["usage"] == {"input_tokens": 120, "output_tokens": 0}
    assert result["estimated_usd"] is None
    assert result["decision_http_time_s"] >= 0 and result["control_http_time_s"] >= 0
    assert result["configuration_fingerprint"] == fingerprint(instance.configuration())


def test_no_call_skips_arguments_and_function_probability_remains_decisive():
    seen = []
    result = verifier(transport_handler(response(0.2), seen=seen)).verify(EXAMPLE, [])
    assert list(json.loads(seen[-1].content)["questions"]) == ["function_correct"]
    assert result["arguments_skipped"] is True
    assert result["arguments_probability"] is result["arguments_pass"] is None
    assert result["accepted"] is False and result["question_evaluations"] == 1


def test_missing_arguments_are_checked_and_schema_error_cannot_be_overruled():
    seen = []
    result = verifier(transport_handler(seen=seen)).verify(EXAMPLE, [{"name": "weather", "arguments": {}}])
    assert "arguments_correct" in json.loads(seen[-1].content)["questions"]
    assert result["arguments_probability"] == 0.9 and result["arguments_pass"] is False


def test_unreadable_proposal_makes_no_network_calls_and_records_real_zeros():
    def no_network(request):
        pytest.fail("Malformed proposal must not reach the service")
    result = verifier(no_network).verify(EXAMPLE, None)
    assert result["api_calls"] == result["control_api_calls"] == result["forward_calls"] == 0
    assert result["usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert result["model_provenance"] is None and result["model"] is None


@pytest.mark.parametrize("value", [True, False, -0.01, 1.01, float("nan"), float("inf"), "0.9", None])
def test_invalid_probabilities_cannot_be_valid_verdicts(value):
    body = response(value)
    def handler(request):
        if request.url.path == "/provenance":
            return httpx.Response(200, json=PROVENANCE)
        return httpx.Response(200, content=json.dumps(body), headers={"Content-Type": "application/json"})
    with pytest.raises(DeciderError, match="invalid Noul"):
        verifier(handler).verify(EXAMPLE, CALL)


@pytest.mark.parametrize("field,value", [("revision", "0" * 40), ("model_id", "Qwen/Qwen3.8-27B"),
                                         ("native_attention", "causal"), ("temperature", 1.0),
                                         ("max_input_tokens", 16384)])
def test_provenance_mismatch_stops_before_any_inference(field, value):
    seen, provenance = [], {**PROVENANCE, field: value}
    with pytest.raises(DeciderError) as failure:
        verifier(transport_handler(provenance=provenance, seen=seen)).verify(EXAMPLE, CALL)
    assert failure.value.code == "provenance_mismatch"
    assert [request.url.path for request in seen] == ["/provenance"]


@pytest.mark.parametrize("model", ["jev-1.13.0", "jev-latest", "Qwen/Qwen3.8-27B", None])
def test_unexpected_response_model_is_not_relabelled_as_pplx(model):
    with pytest.raises(DeciderError) as failure:
        verifier(transport_handler(response(model=model))).verify(EXAMPLE, CALL)
    assert failure.value.code == "model_mismatch"


def test_missing_metrics_remain_unknown_and_missing_native_batch_marker_is_unknown():
    provenance = {key: value for key, value in PROVENANCE.items() if key not in {"inference_batch_size", "one_branch_per_question"}}
    result = verifier(transport_handler(response(usage=False), provenance=provenance)).verify(EXAMPLE, CALL)
    assert result["usage"] == {"input_tokens": None, "output_tokens": None}
    assert result["usage_complete"] is False and result["forward_calls"] is None


def test_deployment_native_batch_alias_is_normalized_without_changing_model_identity():
    provenance = {key: value for key, value in PROVENANCE.items() if key != "inference_batch_size"}
    provenance["native_inference_batch_size"] = 8
    result = verifier(transport_handler(provenance=provenance)).verify(EXAMPLE, CALL)
    assert result["forward_calls"] == 1
    assert result["model_provenance"]["model_id"] == MODEL_ID


def test_retry_attempts_and_unknown_failed_attempt_usage_are_accounted_separately():
    seen = []
    def handler(request):
        if request.url.path == "/provenance":
            return httpx.Response(200, json=PROVENANCE)
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(529, headers={"Retry-After": "0"})
        return httpx.Response(200, json=response())
    result = verifier(handler, max_retries=1, sleep=lambda _: None).verify(EXAMPLE, CALL)
    assert result["api_calls"] == 2 and result["successful_forward_batches"] == 1
    assert result["forward_calls"] is None
    assert result["usage"] == {"input_tokens": None, "output_tokens": None}
    assert result["successful_response_usage"] == {"input_tokens": 120, "output_tokens": 0}


def test_context_overflow_is_a_sanitized_error_not_a_false_prediction():
    def handler(request):
        if request.url.path == "/provenance":
            return httpx.Response(200, json=PROVENANCE)
        return httpx.Response(422, json={"detail": "Question branch exceeds the 8192-token limit; secret text"})
    with pytest.raises(DeciderError) as failure:
        verifier(handler).verify(EXAMPLE, CALL)
    assert failure.value.code == "context_limit_exceeded" and failure.value.status_code == 422
    assert "secret text" not in str(failure.value)


def test_configuration_fingerprints_are_stable_and_model_settings_change_identity():
    a, b, c = verifier(), verifier(), verifier(threshold=0.6)
    assert fingerprint(a.configuration()) == fingerprint(b.configuration())
    assert fingerprint(a.configuration()) != fingerprint(c.configuration())
    assert "Authorization" not in json.dumps(verifier(api_key="private-test-token").configuration())
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    with pytest.raises(ValueError):
        fingerprint({"x": float("nan")})


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_toy_smoke_keeps_expected_labels_out_of_prompts_and_remains_synthetic():
    module, seen = load_script("dgx_decider_smoke"), []
    receipt = module.run_smoke(verifier(transport_handler(seen=seen)))
    assert receipt["synthetic"] is True and receipt["summary"]["generator_calls"] == 0
    assert receipt["summary"]["api_calls"] == 8 and receipt["summary"]["question_evaluations"] == 13
    assert receipt["summary"]["scripted_generator_calls"] == 3
    assert receipt["summary"]["scripted_repair_accepted"] is True
    assert receipt["summary"]["scripted_no_call_accepted"] is True
    repair, no_call = receipt["scripted_trajectories"]
    assert repair["repairs"] == 1 and repair["executed"] is False
    assert no_call["calls"] == [] and no_call["executed"] is False
    for request in seen:
        if request.method == "POST":
            assert "expected_" not in request.content.decode() and "fictional-" not in request.content.decode()


def write_inputs(tmp_path):
    example = {"id": "case-1", "messages": EXAMPLE["messages"], "tools": [TOOL], "ground_truth": CALL}
    dataset, initials, manifest = [tmp_path / name for name in ("dataset.jsonl", "initial.jsonl", "manifest.json")]
    dataset.write_text(json.dumps(example) + "\n")
    initials.write_text(json.dumps({"id": "case-1", "replicate": 0, "config_id": "old-run",
                                    "proposal": {"calls": CALL, "usage": {}, "latency_s": 1.0}}) + "\n")
    manifest.write_text(json.dumps({"config_id": "old-run", "llm_model": "historical-generator"}))
    return dataset, initials, manifest


def test_prepare_hashes_inputs_without_changing_files_or_contacting_models(tmp_path, monkeypatch):
    module = load_script("prepare_dgx_decider_experiment")
    paths = write_inputs(tmp_path)
    original = [path.read_bytes() for path in paths]
    monkeypatch.setattr(httpx.Client, "get", lambda *_args, **_kwargs: pytest.fail("Preparation cannot contact models"))
    plan = module.prepare(*paths, base_url="http://127.0.0.1:18081")
    assert plan["dataset"]["examples"] == plan["initial_cache"]["rows"] == 1
    assert plan["execution"]["model_api_calls"] == 0 and plan["ready_for_benchmark"] is False
    assert original == [path.read_bytes() for path in paths]
    assert plan["configuration_fingerprint"] == fingerprint({key: value for key, value in plan.items()
                                                            if key != "configuration_fingerprint"})
    assert "ground_truth" not in json.dumps(plan)


@pytest.mark.parametrize("change", ["duplicate", "foreign", "different-config", "missing-case"])
def test_prepare_rejects_unpaired_initials(tmp_path, change):
    module = load_script("prepare_dgx_decider_experiment")
    dataset, initials, manifest = write_inputs(tmp_path)
    row = json.loads(initials.read_text())
    if change == "duplicate":
        initials.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    elif change == "foreign":
        row["id"] = "other"
        initials.write_text(json.dumps(row) + "\n")
    elif change == "different-config":
        row["config_id"] = "different"
        initials.write_text(json.dumps(row) + "\n")
    else:
        example = copy.deepcopy(json.loads(dataset.read_text()))
        example["id"] = "case-2"
        dataset.write_text(dataset.read_text() + json.dumps(example) + "\n")
    with pytest.raises(ValueError):
        module.prepare(dataset, initials, manifest, base_url="http://127.0.0.1:18081")


def test_legacy_initials_require_exact_source_trace_proof_without_modifying_cache(tmp_path):
    module = load_script("prepare_dgx_decider_experiment")
    dataset, initials, manifest = write_inputs(tmp_path)
    row = json.loads(initials.read_text())
    del row["config_id"]
    initials.write_text(json.dumps(row) + "\n")
    original = initials.read_bytes()
    with pytest.raises(ValueError, match="source records"):
        module.prepare(dataset, initials, manifest, base_url="http://127.0.0.1:18081")
    records = tmp_path / "records.jsonl"
    source = {"id": row["id"], "replicate": 0, "method": "direct", "config_id": "old-run",
              "trace": [{"proposal": row["proposal"]}]}
    records.write_text(json.dumps(source) + "\n")
    plan = module.prepare(dataset, initials, manifest, base_url="http://127.0.0.1:18081", source_records=records)
    assert plan["source_record_proof"]["proven_initial_pairs"] == 1
    assert initials.read_bytes() == original
    source["trace"][0]["proposal"]["calls"] = []
    records.write_text(json.dumps(source) + "\n")
    with pytest.raises(ValueError, match="attempt-zero"):
        module.prepare(dataset, initials, manifest, base_url="http://127.0.0.1:18081", source_records=records)


@pytest.mark.parametrize("url", ["http://public.invalid", "http://user:pass@localhost", "http://localhost/api", "ftp://localhost"])
def test_plaintext_remote_or_credential_urls_rejected(url):
    with pytest.raises(ValueError):
        PerplexityDecider(base_url=url)
