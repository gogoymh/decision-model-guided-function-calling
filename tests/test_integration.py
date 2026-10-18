"""Actual CLI/pipeline/scorer/reporting with HTTP transports replaced offline."""

from __future__ import annotations

import json
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import httpx
import pytest

from jev_fc import cli
from jev_fc.benchmarks import load_examples, stratified_split
from jev_fc.jev import JevVerifier
from jev_fc.llm import FunctionCallingLLM
from jev_fc.pipeline import METHODS
from jev_fc.reporting import build_summary, read_jsonl, update_paper


def cut_off_generation(style):
    """A 200 response stopped at the output token limit in the middle of a call."""
    if style == "responses":
        return {"model": "offline-model-resolved", "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [{"type": "function_call", "name": "calculate_triangle_area", "arguments": '{"base": 1',
                            "call_id": "call_0", "status": "incomplete"}],
                "usage": {"input_tokens": 100, "output_tokens": 4096, "input_tokens_details": {"cached_tokens": 0}}}
    if style == "anthropic_messages":
        return {"type": "message", "role": "assistant", "model": "claude-offline-model-resolved",
                "stop_reason": "max_tokens", "content": [{"type": "text", "text": "Partial"}],
                "usage": {"input_tokens": 100, "output_tokens": 4096}}
    return {"model": "offline-model", "choices": [{"finish_reason": "length", "message": {
        "role": "assistant", "content": None, "tool_calls": [{"id": "call_0", "type": "function", "function": {
            "name": "calculate_triangle_area", "arguments": '{"base": 1'}}]}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 4096}}


@pytest.fixture
def offline_run(tmp_path, monkeypatch, request):
    api_style = getattr(request, "param", "chat_completions")
    gateway = "direct" if api_style == "chat_completions" else "apim"
    monkeypatch.setenv("API_GATEWAY", gateway)
    monkeypatch.setenv("LLM_API_STYLE", api_style)
    monkeypatch.setenv("LLM_REASONING_EFFORT", "none")
    monkeypatch.setenv("LLM_MAX_COMPLETION_TOKENS", "4096")
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
    monkeypatch.setenv("APIM_BASE_URL", "https://enhans-aoai-prod-apim.azure-api.net")
    monkeypatch.setenv("APIM_SUBSCRIPTION_KEY", "integration-subscription-before-rotation")
    monkeypatch.setenv("LLM_API_KEY", "unused-provider-secret")
    for key, value in {"LLM_INPUT_PRICE": "1", "LLM_CACHED_INPUT_PRICE": ".1", "LLM_OUTPUT_PRICE": "2",
                       "LLM_CACHE_WRITE_5M_PRICE": "1.25", "LLM_CACHE_WRITE_1H_PRICE": "2", "JEV_INPUT_PRICE": ".042"}.items():
        monkeypatch.setenv(key, value)
    prepared = Path(__file__).resolve().parents[1] / "data/bfcl"
    if not (prepared / "examples.jsonl").exists():
        pytest.skip("Run python -m jev_fc.benchmarks once; integration tests never download data")
    all_rows = load_examples(prepared)
    selected = [next(row for row in all_rows if row["id"] == key) for key in ("simple_python_0", "multiple_0", "irrelevance_0")]
    # These are offline fixtures, and their oracle calls never enter model inputs.
    oracle_calls = {
        "simple_python_0": [{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}}],
        "multiple_0": [{"name": "triangle_properties_get", "arguments": {"side1": 5, "side2": 4, "side3": 3}}],
        "irrelevance_0": [],
    }
    by_prompt = {row["messages"][0]["content"]: row["id"] for row in selected}
    for row in selected:
        row["private_canary"] = "ORACLE_MUST_NOT_APPEAR_ON_WIRE"
    data_path = tmp_path / "tiny.jsonl"
    data_path.write_text("".join(json.dumps(row) + "\n" for row in selected))
    observed = {"llm": [], "jev": [], "histories": [], "fail_next_jev": False, "fail_next_llm": False,
                "incomplete_next": {}, "filter_prompt": {},
                "api_style": api_style, "gateway": gateway,
                "repairs_until_correct": 1,
                "subscription_key": "integration-subscription-before-rotation", "auth_providers": set()}

    def check_wire(body):
        text = json.dumps(body)
        assert "ORACLE_MUST_NOT_APPEAR_ON_WIRE" not in text
        assert "REPAIR_SELECTION_MUST_NOT_APPEAR_ON_WIRE" not in text
        for secret in ("ground_truth", "bfcl_functions", "dataset_revision", "name_map", "private_canary"):
            assert secret not in text
        for metadata in ("repair_budget_selection", "selected_max_repairs", "selection_rule", "selection_id", "selection_method"):
            assert metadata not in text
        assert "category" not in body and "id" not in body

    def generator_http(request):
        style = observed["api_style"]
        if gateway == "apim":
            provider, route = ("anthropic", "messages") if style == "anthropic_messages" else ("openai", "responses")
            assert str(request.url) == f"https://enhans-aoai-prod-apim.azure-api.net/llm/{provider}/v1/{route}"
            assert request.headers["api-key"] == observed["subscription_key"]
            for header in ("authorization", "x-api-key", "anthropic-version"):
                assert header not in request.headers
            observed["auth_providers"].add(provider)
        else:
            assert str(request.url) == "https://mock-llm.invalid/v1/chat/completions"
            assert request.headers["authorization"] == "Bearer offline-fixture-key"
            assert "api-key" not in request.headers
        body = json.loads(request.content)
        check_wire(body)
        observed["llm"].append(deepcopy(body))
        if observed["fail_next_llm"]:
            observed["fail_next_llm"] = False
            return httpx.Response(403, headers={"apim-request-id": "req-llm-403"},
                                  json={"error": "PRIVATE_RESPONSE_BODY " + observed["subscription_key"]})
        user_messages = [message for message in body["input" if style == "responses" else "messages"] if message["role"] == "user"]
        example_id = by_prompt[user_messages[-1]["content"]]
        history = []
        if len(user_messages) > 1:
            assert len(user_messages) == 2
            context = user_messages[0]["content"]
            assert context.startswith("Accumulated revision history (oldest first; proposals NOT executed):\n")
            history, _ = json.JSONDecoder().raw_decode(context.split("\n", 1)[1])
            rounds = [step["round"] for step in history]
            assert rounds == list(range(rounds[0], rounds[0] + len(history)))
            assert rounds[0] >= 1
            assert all(set(step) == {"round", "calls", "feedback"} for step in history)
        observed["histories"].append((example_id, deepcopy(history)))
        if observed["incomplete_next"].get(example_id):
            observed["incomplete_next"][example_id] -= 1
            return httpx.Response(200, json=cut_off_generation(style))
        if len(history) in observed["filter_prompt"].get(example_id, ()):
            return httpx.Response(400, headers={"apim-request-id": "req-llm-filtered"}, json={"error": {
                "code": "content_filter", "message": "PRIVATE_FILTER_MESSAGE", "param": "prompt", "status": 400}})
        calls = deepcopy(oracle_calls[example_id])
        if style != "anthropic_messages":
            assert body["tool_choice"] == "auto"
        if not history or history[-1]["round"] < observed["repairs_until_correct"]:
            # The initial proposal has valid JSON but missing required arguments.
            for call in calls:
                call["arguments"] = {}
        if style == "responses":
            assert body["store"] is False and body["parallel_tool_calls"] is False
            assert all(tool["strict"] is False and "function" not in tool for tool in body["tools"])
            output = [{"type": "function_call", "name": call["name"], "arguments": json.dumps(call["arguments"]),
                       "call_id": f"call_{i}", "status": "completed"} for i, call in enumerate(calls)]
            if not output:
                output = [{"type": "message", "role": "assistant", "status": "completed",
                           "content": [{"type": "output_text", "text": "No available function applies."}]}]
            return httpx.Response(200, json={"model": "offline-model-resolved", "status": "completed", "output": output,
                                            "usage": {"input_tokens": 100, "output_tokens": 10,
                                                      "input_tokens_details": {"cached_tokens": 20}}})
        if style == "anthropic_messages":
            assert body["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
            assert all("input_schema" in tool and "function" not in tool for tool in body["tools"])
            assert "reasoning_effort" not in body and "reasoning" not in body
            content = [{"type": "tool_use", "id": f"call_{i}", "name": call["name"], "input": call["arguments"]} for i, call in enumerate(calls)]
            return httpx.Response(200, json={"type": "message", "role": "assistant", "model": "claude-offline-model-resolved",
                                            "stop_reason": "tool_use" if content else "end_turn",
                                            "content": content or [{"type": "text", "text": "No available function applies."}],
                                            "usage": {"input_tokens": 100, "output_tokens": 10,
                                                      "cache_read_input_tokens": 20, "cache_creation_input_tokens": 5,
                                                      "cache_creation": {"ephemeral_5m_input_tokens": 5, "ephemeral_1h_input_tokens": 0}}})
        native = [{"id": f"call_{i}", "type": "function", "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}} for i, call in enumerate(calls)]
        return httpx.Response(200, json={
            "model": "offline-model", "choices": [{"finish_reason": "tool_calls" if native else "stop", "message": {"role": "assistant", "content": None if native else "No available function applies.", "tool_calls": native}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10},
        })

    def verifier_http(request):
        if gateway == "apim":
            assert str(request.url) == "https://enhans-aoai-prod-apim.azure-api.net/llm/typesafe/v1/systemone"
            assert request.headers["api-key"] == observed["subscription_key"]
            for header in ("authorization", "x-api-key", "anthropic-version"):
                assert header not in request.headers
            observed["auth_providers"].add("typesafe")
        else:
            assert str(request.url) == "https://mock-jev.invalid/v1/systemone"
        body = json.loads(request.content)
        check_wire(body)
        observed["jev"].append(deepcopy(body))
        if observed["fail_next_jev"]:
            observed["fail_next_jev"] = False
            return httpx.Response(401, headers={"apim-request-id": "req-jev-401"},
                                  json={"error": "PRIVATE_RESPONSE_BODY " + observed["subscription_key"]})
        calls = body["state"]["proposed_calls"]
        assert set(body["questions"]) == ({"function_correct", "arguments_correct"} if calls else {"function_correct"})
        assert all(question["type"] == "noul" for question in body["questions"].values())
        assert set(body["state"]) == {"messages", "tools", "proposed_calls"}
        valid_arguments = all(call["arguments"] for call in calls)
        answers = {"function_correct": {"type": "noul", "noul": .99}}
        if calls:
            answers["arguments_correct"] = {"type": "noul", "noul": .99 if valid_arguments else .01}
        return httpx.Response(200, json={
            "model": "jev-1.13.0",
            "answers": answers,
            "usage": {"input_tokens": 300 if calls else 200, "output_tokens": 0},
        })

    llm_http = httpx.Client(transport=httpx.MockTransport(generator_http))
    jev_http = httpx.Client(transport=httpx.MockTransport(verifier_http))
    def make_generator():
        if gateway == "apim":
            return FunctionCallingLLM(model="claude-offline-model" if observed["api_style"] == "anthropic_messages" else "offline-model",
                                      client=llm_http, max_retries=0, timeout=35)
        return FunctionCallingLLM(api_key="offline-fixture-key", model="offline-model", base_url="https://mock-llm.invalid/v1",
                                  client=llm_http, max_retries=0, gateway="direct", api_style="chat_completions")

    def make_verifier(**thresholds):
        if gateway == "apim":
            return JevVerifier(client=jev_http, max_retries=0, timeout=40, **thresholds)
        return JevVerifier(api_key="offline-fixture-key", base_url="https://mock-jev.invalid", client=jev_http,
                           max_retries=0, gateway="direct", **thresholds)

    monkeypatch.setattr(cli, "FunctionCallingLLM", make_generator)
    monkeypatch.setattr(cli, "JevVerifier", make_verifier)
    monkeypatch.setattr(cli, "_source_hash", lambda: "offline-source-fingerprint")
    args = Namespace(
        data=str(data_path), checker=str(prepared / "upstream/ast_checker.py"), output=str(tmp_path / "run"),
        categories=None, split="all", split_seed=1729, limit=None,
        methods="direct,jev_joint", max_repairs=1, replicates=1, workers=1,
        threshold=.8, exhaustion_policy="execute_last", resume=False,
        repair_budget_selection=None,
    )
    yield args, observed, selected
    llm_http.close()
    jev_http.close()


def test_cli_run_resume_and_report_use_saved_pairs_without_new_requests(offline_run, monkeypatch):
    args, observed, _ = offline_run
    cli.run(args)
    directory = Path(args.output)
    records = read_jsonl(directory / "records.jsonl")
    initials = read_jsonl(directory / "initial.jsonl")
    summary = json.loads((directory / "summary.json").read_text())
    assert len(initials) == 3 and len(records) == 6
    assert len({(row["id"], row["replicate"], row["method"]) for row in records}) == 6
    assert len(observed["llm"]) == 5 and len(observed["jev"]) == 5
    for example_id in ("simple_python_0", "multiple_0", "irrelevance_0"):
        pair = {row["method"]: row for row in records if row["id"] == example_id}
        assert pair["direct"]["trace"][0]["proposal"] == pair["jev_joint"]["trace"][0]["proposal"]
        assert pair["jev_joint"]["score"]["correct"] is True
        assert pair["jev_joint"]["repairs"] == (0 if example_id == "irrelevance_0" else 1)
        assert pair["jev_joint"]["dispatch_mode"] == "simulated"
        assert pair["jev_joint"]["execution_result"] is None
    assert summary["status"] == "complete" and summary["completed_records"] == 6
    assert summary["methods"]["direct"]["accuracy"] == pytest.approx(1 / 3)
    assert summary["methods"]["jev_joint"]["accuracy"] == 1
    assert summary["paired_vs_direct"]["jev_joint"]["n_paired_ids"] == 3

    saved_records = (directory / "records.jsonl").read_bytes()
    args.resume = True
    cli.run(args)
    assert (directory / "records.jsonl").read_bytes() == saved_records
    assert len(observed["llm"]) == 5 and len(observed["jev"]) == 5
    monkeypatch.setattr("sys.argv", ["jev-fc", "report", str(directory)])
    assert cli.main() == 0
    assert (directory / "summary.csv").exists()
    assert json.loads((directory / "summary.json").read_text())["status"] == "complete"


@pytest.mark.parametrize("offline_run", ["chat_completions", "responses", "anthropic_messages"], indirect=True)
def test_latest_and_cumulative_use_same_initial_but_different_second_repair_context(offline_run):
    args, observed, _ = offline_run
    args.methods = "jev_joint,jev_joint_latest"
    args.max_repairs = observed["repairs_until_correct"] = 2
    cli.run(args)
    directory = Path(args.output)
    rows = read_jsonl(directory / "records.jsonl")
    assert len(rows) == 6 and len(read_jsonl(directory / "initial.jsonl")) == 3
    assert len(observed["llm"]) == 11 and len(observed["jev"]) == 14
    for example_id in ("simple_python_0", "multiple_0", "irrelevance_0"):
        pair = {row["method"]: row for row in rows if row["id"] == example_id}
        cumulative, latest = pair["jev_joint"], pair["jev_joint_latest"]
        assert cumulative["trace"][0]["proposal"] == latest["trace"][0]["proposal"]
        assert cumulative["accepted"] is latest["accepted"] is True
        assert cumulative["score"] == latest["score"]
        assert latest["initial_proposal_shared"] is True
        assert latest["repairs"] == (0 if example_id == "irrelevance_0" else 2)
        assert len(latest["trace"]) == len(cumulative["trace"]) == latest["repairs"] + 1
        assert [step["attempt"] for step in latest["trace"]] == list(range(latest["repairs"] + 1))
        rounds = [[record["round"] for record in history]
                  for key, history in observed["histories"] if key == example_id]
        assert rounds == ([[]] if example_id == "irrelevance_0" else [[], [1], [1, 2], [1], [2]])
    before = len(observed["llm"]), len(observed["jev"])
    args.resume = True
    cli.run(args)
    assert (len(observed["llm"]), len(observed["jev"])) == before
    assert read_jsonl(directory / "records.jsonl") == rows


@pytest.mark.parametrize("offline_run", ["chat_completions", "responses", "anthropic_messages"], indirect=True)
def test_cut_off_generation_is_scored_wrong_repaired_and_counted_without_stopping(offline_run):
    args, observed, _ = offline_run
    observed["incomplete_next"]["irrelevance_0"] = 1
    cli.run(args)
    directory = Path(args.output)
    assert not (directory / "errors.jsonl").exists()
    rows = read_jsonl(directory / "records.jsonl")
    assert len(rows) == 6 and len(observed["llm"]) == 6 and len(observed["jev"]) == 5
    pair = {row["method"]: row for row in rows if row["id"] == "irrelevance_0"}
    direct, joint = pair["direct"], pair["jev_joint"]
    # The shared initial proposal is unreadable: never a correct no-call.
    assert direct["calls"] is None and direct["trace"][0]["proposal"]["incomplete"] == "length"
    assert direct["score"]["correct"] is False
    assert direct["accounting"]["complete"] is True and direct["accounting"]["llm_output"] == 4096
    first = joint["trace"][0]
    assert first["proposal"] == direct["trace"][0]["proposal"]
    assert first["verification"]["api_calls"] == 0 and first["gate_pass"] is False
    repair_histories = [history for key, history in observed["histories"] if key == "irrelevance_0"]
    assert repair_histories[1][0]["calls"] is None
    assert joint["repairs"] == 1 and joint["accepted"] is True and joint["score"]["correct"] is True
    summary = build_summary(directory)
    assert summary["status"] == "complete" and summary["api_error_events"] == 0
    assert summary["methods"]["direct"]["accuracy"] == 0
    assert (summary["methods"]["direct"]["incomplete_proposal_n"], summary["methods"]["direct"]["incomplete_final_n"]) == (1, 1)
    assert (summary["methods"]["jev_joint"]["incomplete_proposal_n"], summary["methods"]["jev_joint"]["incomplete_final_n"]) == (1, 0)
    assert summary["methods"]["direct"]["mean_cost_usd"] is not None
    assert_saved_files_have_no_credentials(directory)


@pytest.mark.parametrize("offline_run", ["chat_completions", "responses"], indirect=True)
def test_prompt_rejected_by_filter_during_repair_is_scored_wrong_without_stopping(offline_run):
    args, observed, _ = offline_run
    observed["filter_prompt"]["simple_python_0"] = {1}
    cli.run(args)
    directory = Path(args.output)
    assert not (directory / "errors.jsonl").exists()
    rows = read_jsonl(directory / "records.jsonl")
    assert len(rows) == 6 and len(observed["llm"]) == 5 and len(observed["jev"]) == 4
    joint = next(row for row in rows if row["id"] == "simple_python_0" and row["method"] == "jev_joint")
    filtered = joint["trace"][1]
    assert filtered["proposal"]["incomplete"] == "prompt_filter" and filtered["proposal"]["usage"] == {}
    assert filtered["verification"]["api_calls"] == 0 and filtered["gate_pass"] is False
    assert joint["exhausted"] is True and joint["calls"] is None and joint["score"]["correct"] is False
    assert joint["accounting"]["tokens_complete"] is False
    summary = build_summary(directory)
    assert summary["status"] == "complete" and summary["api_error_events"] == 0
    joint_summary = summary["methods"]["jev_joint"]
    assert (joint_summary["incomplete_proposal_n"], joint_summary["incomplete_final_n"]) == (1, 1)
    assert joint_summary["cost_observed_n"] == 2 and joint_summary["mean_cost_usd"] is None
    assert summary["methods"]["direct"]["mean_cost_usd"] is not None
    for path in directory.iterdir():
        assert "PRIVATE_FILTER_MESSAGE" not in path.read_text()
    assert_saved_files_have_no_credentials(directory)


def test_partial_resume_keeps_completed_method_and_shared_initial(offline_run):
    args, observed, _ = offline_run
    observed["fail_next_jev"] = True
    with pytest.raises(RuntimeError, match="Jev HTTP 401"):
        cli.run(args)
    directory = Path(args.output)
    saved = read_jsonl(directory / "records.jsonl")
    first_initials = read_jsonl(directory / "initial.jsonl")
    assert len(saved) == len(first_initials) == 1
    assert saved[0]["method"] == "direct"
    assert build_summary(directory)["status"] == "partial"
    args.resume = True
    cli.run(args)
    records = read_jsonl(directory / "records.jsonl")
    assert len(records) == 6 and records[0] == saved[0]
    assert len({(row["id"], row["replicate"], row["method"]) for row in records}) == 6
    assert len(read_jsonl(directory / "initial.jsonl")) == 3
    assert len(observed["llm"]) == 5  # Failed verifier does not regenerate initial.
    assert build_summary(directory)["api_error_events"] == 1


def test_resume_rejects_changed_settings_before_new_requests(offline_run):
    args, observed, _ = offline_run
    cli.run(args)
    before = len(observed["llm"]), len(observed["jev"])
    args.resume = True
    args.max_repairs = 2
    with pytest.raises(ValueError, match="different code/data/config fingerprint"):
        cli.run(args)
    assert (len(observed["llm"]), len(observed["jev"])) == before


def test_report_refuses_smoke_or_synthetic_results_as_manuscript(offline_run, tmp_path):
    args, _, _ = offline_run
    cli.run(args)
    summary = build_summary(args.output)
    paper = tmp_path / "paper.tex"
    source = "% BEGIN AUTO RESULTS\n--\n% END AUTO RESULTS\n"
    paper.write_text(source)
    with pytest.raises(ValueError, match="declared protocol"):
        update_paper(paper, summary)
    assert paper.read_text() == source
    manifest_path = Path(args.output) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["synthetic"] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Synthetic demonstrations"):
        build_summary(args.output)


def test_cli_and_preparation_split_membership_agree():
    examples = [{"id": f"{category}_{i}", "category": category} for category in ("simple_python", "multiple", "irrelevance") for i in range(30)]
    development, test = stratified_split(examples, seed=1729)
    assert {row["id"] for row in cli.stable_split(examples, "dev", 1729)} == {row["id"] for row in development}
    assert {row["id"] for row in cli.stable_split(examples, "test", 1729)} == {row["id"] for row in test}


def assert_saved_files_have_no_credentials(directory):
    forbidden = ("offline-fixture-key", "integration-subscription-before-rotation",
                 "integration-subscription-after-rotation", "unused-provider-secret", "PRIVATE_RESPONSE_BODY")
    for path in Path(directory).iterdir():
        if path.is_file():
            content = path.read_text()
            for secret in forbidden:
                assert secret not in content, f"Private fixture data leaked into {path.name}"


@pytest.mark.parametrize("offline_run", ["responses", "anthropic_messages"], indirect=True)
def test_apim_cli_pipeline_scoring_report_and_key_rotation_resume(offline_run, monkeypatch):
    args, observed, _ = offline_run
    cli.run(args)
    directory = Path(args.output)
    manifest = json.loads((directory / "manifest.json").read_text())
    records = read_jsonl(directory / "records.jsonl")
    summary = build_summary(directory)
    anthropic = observed["api_style"] == "anthropic_messages"
    provider, route = ("anthropic", "messages") if anthropic else ("openai", "responses")
    assert manifest["llm_provider"] == provider
    assert manifest["llm_api_style"] == observed["api_style"]
    assert manifest["llm_gateway"] == manifest["jev_gateway"] == "apim"
    assert manifest["llm_endpoint"] == f"https://enhans-aoai-prod-apim.azure-api.net/llm/{provider}/v1/{route}"
    assert manifest["jev_endpoint"] == "https://enhans-aoai-prod-apim.azure-api.net/llm/typesafe/v1/systemone"
    assert manifest["llm_auth_header"] == "api-key"
    assert manifest["workers"] == args.workers == 1
    assert manifest["llm_timeout_s"] == 35 and manifest["jev_timeout_s"] == 40
    assert manifest["llm_max_retries"] == manifest["jev_max_retries"] == 0
    assert observed["auth_providers"] == {provider, "typesafe"}
    assert summary["status"] == "complete" and summary["completed_records"] == 6
    assert len(read_jsonl(directory / "initial.jsonl")) == 3
    assert len(observed["llm"]) == len(observed["jev"]) == 5
    assert summary["methods"]["direct"]["accuracy"] == pytest.approx(1 / 3)
    assert summary["methods"]["jev_joint"]["accuracy"] == 1

    for example_id in ("simple_python_0", "multiple_0", "irrelevance_0"):
        pair = {row["method"]: row for row in records if row["id"] == example_id}
        direct, joint = pair["direct"], pair["jev_joint"]
        assert direct["trace"][0]["proposal"] == joint["trace"][0]["proposal"]
        assert joint["score"]["correct"] is True
        assert joint["repairs"] == (0 if example_id == "irrelevance_0" else 1)
        assert direct["score"]["correct"] is (example_id == "irrelevance_0")
        assert joint["dispatch_mode"] == "simulated" and joint["execution_result"] is None
        for step in joint["trace"]:
            proposal = step["proposal"]
            assert proposal["gateway"] == "apim" and proposal["provider"] == provider
            assert proposal["api_style"] == observed["api_style"]
            assert proposal["usage"]["prompt_tokens"] == (125 if anthropic else 100)
            assert proposal["usage"]["completion_tokens"] == 10
            assert proposal["usage"]["prompt_tokens_details"]["cached_tokens"] == 20
            assert proposal["raw_usage"]["input_tokens"] == 100
            assert proposal["model"] == ("claude-offline-model-resolved" if anthropic else "offline-model-resolved")
            assert proposal["requested_model"] == manifest["llm_model"]
            if anthropic:
                assert proposal["usage"]["cache_creation_input_tokens"] == 5
                assert proposal["raw_usage"]["cache_creation"]["ephemeral_5m_input_tokens"] == 5
            assert step["verification"]["usage"]["input_tokens"] == (300 if proposal["calls"] else 200)
            assert step["verification"]["arguments_skipped"] is (not proposal["calls"])
            if not proposal["calls"]:
                assert step["verification"]["arguments_pass"] is None
                assert step["verification"]["arguments_probability"] is None
        assert joint["accounting"]["complete"] is True
        assert joint["accounting"]["estimated_usd"] > 0
        if example_id == "irrelevance_0":
            assert direct["calls"] == joint["calls"] == []
            assert joint["accounting"]["jev_input"] == 200

    before = {name: (directory / name).read_bytes() for name in ("manifest.json", "initial.jsonl", "records.jsonl")}
    before_counts = len(observed["llm"]), len(observed["jev"])
    monkeypatch.setenv("APIM_SUBSCRIPTION_KEY", "integration-subscription-after-rotation")
    observed["subscription_key"] = "integration-subscription-after-rotation"
    args.resume = True
    cli.run(args)
    assert (len(observed["llm"]), len(observed["jev"])) == before_counts
    assert all((directory / name).read_bytes() == data for name, data in before.items())
    assert json.loads((directory / "manifest.json").read_text())["config_id"] == manifest["config_id"]
    monkeypatch.setattr("sys.argv", ["jev-fc", "report", str(directory)])
    assert cli.main() == 0 and (directory / "summary.csv").exists()
    assert_saved_files_have_no_credentials(directory)


def install_tiny_selected_run(args, monkeypatch, selected_budget, tmp_path):
    """Mock source validation for a tiny integration fixture, never production runs.

    tuning.py's own tests check real provenance and canonical split membership.
    Here the real CLI, HTTP adapters, state machine, scoring and fingerprinting
    operate on three saved examples rather than issuing thousands of requests.
    """
    args.split = "test"
    args.methods = ",".join(METHODS)
    args.max_repairs = selected_budget
    selection_path = tmp_path / "repair_selection_fixture.json"
    selection = {"kind": "repair_budget_selection", "selection_id": "fixture-selection-id",
                 "selected_max_repairs": selected_budget, "grid": [0, 1, 2, 3, 4, 5],
                 "selection_method": "jev_joint",
                 "selection_rule": "max_dev_accuracy_then_smallest_budget",
                 "provenance": {"split": "dev", "canary": "REPAIR_SELECTION_MUST_NOT_APPEAR_ON_WIRE"}}
    selection_path.write_text(json.dumps(selection))
    args.repair_budget_selection = str(selection_path)
    monkeypatch.setattr(cli, "stable_split", lambda rows, split, seed: rows)
    validations = []

    def load_selection(path, test_config):
        loaded = json.loads(Path(path).read_text())
        assert test_config["split"] == "test" and test_config["limit"] is None
        assert test_config["max_repairs"] == loaded["selected_max_repairs"]
        assert test_config["methods"] == list(METHODS)
        assert "repair_budget_selection" not in test_config
        validations.append(deepcopy(test_config))
        return loaded

    monkeypatch.setattr(cli, "load_repair_budget_selection", load_selection)
    return selection, selection_path, validations


@pytest.mark.parametrize("offline_run", ["chat_completions", "responses", "anthropic_messages"], indirect=True)
@pytest.mark.parametrize("selected_budget", [0, 5])
def test_selected_budget_reaches_all_methods_and_manifest_but_never_model_inputs(
    offline_run, monkeypatch, tmp_path, selected_budget
):
    args, observed, _ = offline_run
    selection, _, validations = install_tiny_selected_run(args, monkeypatch, selected_budget, tmp_path)
    cli.run(args)
    assert len(validations) == 1
    directory = Path(args.output)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["repair_budget_selection"] == selection
    assert manifest["max_repairs"] == selected_budget
    assert manifest["methods"] == list(METHODS)
    rows = read_jsonl(directory / "records.jsonl")
    assert len(rows) == 3 * len(METHODS)
    assert all(row["max_repairs"] == selected_budget for row in rows)
    assert all(row["repairs"] == 0 for row in rows if row["method"] == "direct")
    assert all(row["repairs"] == selected_budget for row in rows if row["method"] == "self_refine")
    joint = [row for row in rows if row["method"] in {"jev_joint", "jev_joint_latest"}]
    assert all(row["repairs"] <= min(1, selected_budget) for row in joint)
    assert len(METHODS) == 7
    assert all(row["initial_proposal_shared"] is True for row in rows)
    assert manifest["shared_initial_methods"] == list(METHODS)
    assert max(len(history) for _, history in observed["histories"]) == selected_budget
    assert len(observed["llm"]) > 0 and len(observed["jev"]) > 0
    assert_saved_files_have_no_credentials(directory)


@pytest.mark.parametrize("orphan", ["records.jsonl", "initial.jsonl", "unfinished.jsonl"])
def test_orphan_run_data_cannot_start_without_manifest(offline_run, orphan):
    args, observed, _ = offline_run
    directory = Path(args.output)
    directory.mkdir()
    (directory / orphan).write_text("{}\n")
    with pytest.raises(ValueError, match="orphan run files"):
        cli.run(args)
    assert observed["llm"] == observed["jev"] == []
    assert not (directory / "manifest.json").exists()


def test_changed_selection_invalidates_resume_before_network_or_run_changes(offline_run, monkeypatch, tmp_path):
    args, observed, _ = offline_run
    selection, path, validations = install_tiny_selected_run(args, monkeypatch, 5, tmp_path)
    cli.run(args)
    directory = Path(args.output)
    before_counts = len(observed["llm"]), len(observed["jev"])
    before_files = {name: (directory / name).read_bytes() for name in ("manifest.json", "records.jsonl", "initial.jsonl")}
    selection["selection_id"] = "different-development-selection"
    path.write_text(json.dumps(selection))
    args.resume = True
    with pytest.raises(ValueError, match="different code/data/config fingerprint"):
        cli.run(args)
    assert len(validations) == 2
    assert (len(observed["llm"]), len(observed["jev"])) == before_counts
    assert all((directory / name).read_bytes() == value for name, value in before_files.items())


def test_invalid_development_selection_denied_before_requests_and_output_creation(offline_run, monkeypatch, tmp_path):
    args, observed, _ = offline_run
    install_tiny_selected_run(args, monkeypatch, 5, tmp_path)

    def reject_selection(*_, **__):
        raise ValueError("Source development configuration does not match this test run")

    monkeypatch.setattr(cli, "load_repair_budget_selection", reject_selection)
    with pytest.raises(ValueError, match="does not match"):
        cli.run(args)
    assert observed["llm"] == observed["jev"] == []
    assert not Path(args.output).exists()


def test_real_selection_validator_rejects_forged_artifact_before_api_use(offline_run, monkeypatch, tmp_path):
    args, observed, _ = offline_run
    actual_validator = cli.load_repair_budget_selection
    install_tiny_selected_run(args, monkeypatch, 5, tmp_path)
    monkeypatch.setattr(cli, "load_repair_budget_selection", actual_validator)
    with pytest.raises(ValueError, match="fingerprint"):
        cli.run(args)
    assert observed["llm"] == observed["jev"] == []
    assert not Path(args.output).exists()


def test_selection_metadata_alone_cannot_promote_fixture_to_paper_results(offline_run, monkeypatch, tmp_path):
    args, _, _ = offline_run
    install_tiny_selected_run(args, monkeypatch, 0, tmp_path)
    cli.run(args)
    paper = tmp_path / "guarded-paper.tex"
    original = "% BEGIN AUTO RESULTS\n--\n% END AUTO RESULTS\n"
    paper.write_text(original)
    with pytest.raises(ValueError):
        update_paper(paper, build_summary(args.output))
    assert paper.read_text() == original


@pytest.mark.parametrize("split,limit", [("dev", None), ("all", None), ("test", 1)])
def test_selection_cannot_authorize_non_test_or_limited_run(offline_run, monkeypatch, tmp_path, split, limit):
    args, observed, _ = offline_run
    install_tiny_selected_run(args, monkeypatch, 0, tmp_path)
    args.split, args.limit = split, limit
    with pytest.raises(ValueError, match="complete test split"):
        cli.run(args)
    assert observed["llm"] == observed["jev"] == []
    assert not Path(args.output).exists()


def test_standalone_exploration_allows_repair_caps_above_three_without_selection(offline_run):
    args, _, _ = offline_run
    args.max_repairs = 5
    args.methods = "direct,jev_joint"
    cli.run(args)
    manifest = json.loads((Path(args.output) / "manifest.json").read_text())
    assert manifest["max_repairs"] == 5
    assert "repair_budget_selection" not in manifest


@pytest.mark.parametrize("offline_run", ["responses", "anthropic_messages"], indirect=True)
def test_apim_resume_rejects_changed_api_format_before_requests(offline_run, monkeypatch):
    args, observed, _ = offline_run
    cli.run(args)
    before = len(observed["llm"]), len(observed["jev"])
    observed["api_style"] = "anthropic_messages" if observed["api_style"] == "responses" else "responses"
    monkeypatch.setenv("LLM_API_STYLE", observed["api_style"])
    args.resume = True
    with pytest.raises(ValueError, match="different code/data/config fingerprint"):
        cli.run(args)
    assert (len(observed["llm"]), len(observed["jev"])) == before
    assert_saved_files_have_no_credentials(args.output)


@pytest.mark.parametrize("offline_run", ["responses", "anthropic_messages"], indirect=True)
@pytest.mark.parametrize("failing_provider", ["llm", "jev"])
def test_apim_sanitized_http_errors_and_partial_resume_after_key_rotation(offline_run, monkeypatch, failing_provider):
    args, observed, _ = offline_run
    observed[f"fail_next_{failing_provider}"] = True
    with pytest.raises(RuntimeError, match=f"HTTP {403 if failing_provider == 'llm' else 401}"):
        cli.run(args)
    directory = Path(args.output)
    manifest_before = json.loads((directory / "manifest.json").read_text())
    saved_records = read_jsonl(directory / "records.jsonl")
    errors = read_jsonl(directory / "errors.jsonl")
    assert len(errors) == 1
    error = errors[0]
    assert error["status_code"] == (403 if failing_provider == "llm" else 401)
    assert error["request_id"] == ("req-llm-403" if failing_provider == "llm" else "req-jev-401")
    assert error["attempts"] == 1
    assert error["error_type"] == ("LLMAPIError" if failing_provider == "llm" else "JevAPIError")
    assert error["method"] == ("initial" if failing_provider == "llm" else "jev_joint")
    assert set(error) == {"id", "replicate", "method", "error_type", "at", "status_code", "request_id", "attempts", "message"}
    assert error["message"].startswith("LLM HTTP 403" if failing_provider == "llm" else "Jev HTTP 401")
    assert len(saved_records) == (0 if failing_provider == "llm" else 1)
    assert build_summary(directory)["status"] == "partial"
    assert_saved_files_have_no_credentials(directory)

    observed["subscription_key"] = "integration-subscription-after-rotation"
    monkeypatch.setenv("APIM_SUBSCRIPTION_KEY", observed["subscription_key"])
    args.resume = True
    cli.run(args)
    rows = read_jsonl(directory / "records.jsonl")
    assert len(rows) == 6 and len({(r["id"], r["replicate"], r["method"]) for r in rows}) == 6
    assert all(row in rows for row in saved_records)
    assert len(read_jsonl(directory / "initial.jsonl")) == 3
    assert len(observed["llm"]) == (6 if failing_provider == "llm" else 5)
    assert len(observed["jev"]) == (5 if failing_provider == "llm" else 6)
    summary = build_summary(directory)
    assert summary["status"] == "complete" and summary["api_error_events"] == 1
    assert summary["manifest"]["config_id"] == manifest_before["config_id"]
    assert summary["methods"]["jev_joint"]["accuracy"] == 1
    assert_saved_files_have_no_credentials(directory)


def _script(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extension_continues_only_exhausted_jbfc_rows_and_cuts_back_to_the_source(offline_run, monkeypatch, tmp_path):
    args, observed, _ = offline_run
    extend_run, table3 = _script("extend_run"), _script("table3")
    monkeypatch.setattr(extend_run, "load_dotenv", lambda *_: None)
    observed["repairs_until_correct"] = 3  # wrong until the history reaches round 3
    cli.run(args)  # r=1: two calls end exhausted, the no-call is accepted at once
    source = Path(args.output)
    before = len(observed["llm"]), len(observed["jev"]), len(observed["histories"])
    out = tmp_path / "extended"
    argv = [str(source), "--max-repairs", "4", "--output", str(out), "--workers", "1",
            "--data", args.data, "--checker", args.checker]
    summary = extend_run.main(argv)
    assert summary["status"] == "complete" and summary["manifest"]["extends"]["max_repairs"] == 1
    rows = {row["id"]: row for row in read_jsonl(out / "records.jsonl")}
    assert rows["irrelevance_0"]["extended"] is False and rows["irrelevance_0"]["repairs"] == 0
    for example_id in ("simple_python_0", "multiple_0"):
        row = rows[example_id]
        assert row["extended"] is True and row["accepted"] is True and row["repairs"] == 3
        assert row["score"]["correct"] is True and row["max_repairs"] == 4
    # Only attempts 2 and 3 of the two exhausted rows made new requests.
    assert (len(observed["llm"]) - before[0], len(observed["jev"]) - before[1]) == (4, 4)
    new = [[step["round"] for step in history] for _, history in observed["histories"][before[2]:]]
    assert sorted(new) == [[1, 2], [1, 2], [1, 2, 3], [1, 2, 3]]
    # Cutting the r=4 trajectories at r=1 reproduces the source rows exactly.
    summaries = table3.derive_limits(out, limits=(1, 2, 4))
    table3.check_matches_source(out, source)
    assert summaries[1]["methods"]["jev_joint"]["accuracy"] == build_summary(source)["methods"]["jev_joint"]["accuracy"]
    assert summaries[4]["methods"]["jev_joint"]["accuracy"] == 1
    # Resuming the complete extension makes no new requests.
    after = len(observed["llm"]), len(observed["jev"])
    extend_run.main(argv + ["--resume"])
    assert (len(observed["llm"]), len(observed["jev"])) == after
    # A changed source can no longer back the table.
    (source / "records.jsonl").write_bytes((source / "records.jsonl").read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed after the extension"):
        table3.check_extension(out)
    assert_saved_files_have_no_credentials(out)


def test_extension_refuses_other_code_or_settings_before_requests(offline_run, monkeypatch, tmp_path):
    args, observed, _ = offline_run
    extend_run = _script("extend_run")
    monkeypatch.setattr(extend_run, "load_dotenv", lambda *_: None)
    cli.run(args)
    before = len(observed["llm"]), len(observed["jev"])
    argv = [args.output, "--max-repairs", "3", "--output", str(tmp_path / "x"), "--workers", "1",
            "--data", args.data, "--checker", args.checker]
    monkeypatch.setenv("LLM_REASONING_EFFORT", "low")
    with pytest.raises(ValueError, match="reasoning_effort"):
        extend_run.main(argv)
    monkeypatch.setenv("LLM_REASONING_EFFORT", "none")
    monkeypatch.setattr(cli, "_source_hash", lambda: "other-code")
    with pytest.raises(ValueError, match="different code"):
        extend_run.main(argv)
    with pytest.raises(ValueError, match="larger"):
        extend_run.main([args.output, "--max-repairs", "1", "--output", str(tmp_path / "y")])
    assert (len(observed["llm"]), len(observed["jev"])) == before
    assert not (tmp_path / "x").exists()


SEVEN = "direct,schema_only,self_refine,jev_function,jev_arguments,jev_joint,jev_joint_latest"


def test_raised_limit_continues_each_method_by_its_own_stopping_rule(offline_run, monkeypatch, tmp_path):
    args, observed, _ = offline_run
    raise_limit = _script("raise_limit")
    monkeypatch.setattr(raise_limit, "load_dotenv", lambda *_: None)
    observed["repairs_until_correct"] = 2  # calls become right at the second repair
    args.methods = SEVEN
    cli.run(args)  # r=1
    source = Path(args.output)
    old = {(row["id"], row["method"]): row for row in read_jsonl(source / "records.jsonl")}
    before = len(observed["llm"]), len(observed["jev"])
    out = tmp_path / "r2"
    argv = [str(source), "--max-repairs", "2", "--output", str(out), "--workers", "1",
            "--data", args.data, "--checker", args.checker]
    summary = raise_limit.main(argv)
    assert summary["status"] == "complete" and summary["manifest"]["max_repairs"] == 2
    assert summary["manifest"]["extends"]["max_repairs"] == 1 and summary["history_ablation"]["n_paired_ids"] == 3
    new = {(row["id"], row["method"]): row for row in read_jsonl(out / "records.jsonl")}
    for (example_id, method), row in new.items():
        before_row = old[(example_id, method)]
        assert row["max_repairs"] == 2 and row["trace"][0]["proposal"] == before_row["trace"][0]["proposal"]
        if method == "direct" or before_row.get("accepted") is True:
            assert row["trace"] == before_row["trace"] and row.get("extended") is False
        else:  # exhausted at r=1 (and every self-refine row): one more attempt on the saved history
            assert row["extended"] is True and len(row["trace"]) == 3
            assert [s["proposal"] for s in row["trace"][:2]] == [s["proposal"] for s in before_row["trace"]]
    for method in ("schema_only", "jev_function", "jev_arguments", "jev_joint", "jev_joint_latest", "self_refine"):
        assert new[("simple_python_0", method)]["score"]["correct"] is True
    # Exhausted rows: 2 calls x 5 stop-at-pass methods, plus every self-refine row (3); Jev checks only the new
    # attempts of the three Jev methods that use them, two per row.
    assert len(observed["llm"]) - before[0] == 2 * 5 + 3
    assert len(observed["jev"]) - before[1] == 2 * 4
    # Resuming the complete run makes no new requests.
    after = len(observed["llm"]), len(observed["jev"])
    raise_limit.main(argv + ["--resume"])
    assert (len(observed["llm"]), len(observed["jev"])) == after
    assert_saved_files_have_no_credentials(out)


def test_raised_limit_reuses_jbfc_from_a_longer_run_and_adds_jbfc_to_direct(offline_run, monkeypatch, tmp_path):
    args, observed, _ = offline_run
    raise_limit, extend_run = _script("raise_limit"), _script("extend_run")
    for module in (raise_limit, extend_run):
        monkeypatch.setattr(module, "load_dotenv", lambda *_: None)
    observed["repairs_until_correct"] = 3
    args.methods = "direct,jev_joint,jev_joint_latest"
    cli.run(args)  # r=1
    source = Path(args.output)
    common = ["--workers", "1", "--data", args.data, "--checker", args.checker]
    long = tmp_path / "jbfc_r4"
    extend_run.main([str(source), "--max-repairs", "4", "--output", str(long)] + common)
    before = len(observed["llm"]), len(observed["jev"])
    out = tmp_path / "r3"
    summary = raise_limit.main([str(source), "--max-repairs", "3", "--output", str(out),
                                "--reuse", f"jev_joint={long}"] + common)
    assert summary["manifest"]["reused"]["jev_joint"]["max_repairs"] == 4
    longer = {row["id"]: row for row in read_jsonl(long / "records.jsonl")}
    for row in read_jsonl(out / "records.jsonl"):
        if row["method"] == "jev_joint":  # the longer run cut at r=3: no new JBFC request
            assert row["reused"] is True and row["trace"] == longer[row["id"]]["trace"][:4]
    # Only JBFC-L's two exhausted rows continued, by two attempts each.
    assert (len(observed["llm"]) - before[0], len(observed["jev"]) - before[1]) == (4, 4)
    with pytest.raises(ValueError, match="different run"):
        raise_limit.main([str(source), "--max-repairs", "3", "--output", str(out)] + common)

    # A Direct-only run gains JBFC from its saved first proposals: no new first proposal is generated.
    args.methods, args.output = "direct", str(tmp_path / "direct")
    cli.run(args)
    before = len(observed["llm"]), len(observed["jev"])
    added = tmp_path / "direct_plus_jbfc"
    summary = raise_limit.main([args.output, "--max-repairs", "3", "--output", str(added), "--add", "jev_joint"] + common)
    assert summary["manifest"]["methods"] == ["direct", "jev_joint"] and summary["manifest"]["jev_model"] == "jev-1.13.0"
    rows = {(row["id"], row["method"]): row for row in read_jsonl(added / "records.jsonl")}
    direct = {row["id"]: row for row in read_jsonl(Path(args.output) / "records.jsonl")}
    for example_id, row in direct.items():
        assert rows[(example_id, "jev_joint")]["trace"][0]["proposal"] == row["trace"][0]["proposal"]
        assert rows[(example_id, "direct")]["trace"] == row["trace"]
    # Two exhausted-at-first calls repaired three times; the no-call is accepted at once.
    assert len(observed["llm"]) - before[0] == 2 * 3
    assert len(observed["jev"]) - before[1] == 2 * 4 + 1
    with pytest.raises(ValueError, match="--add"):
        raise_limit.main([args.output, "--max-repairs", "3", "--output", str(tmp_path / "z"), "--add", "direct"] + common)


def test_raised_limit_refuses_other_settings_code_or_first_proposals(offline_run, monkeypatch, tmp_path):
    args, observed, _ = offline_run
    raise_limit = _script("raise_limit")
    monkeypatch.setattr(raise_limit, "load_dotenv", lambda *_: None)
    cli.run(args)
    common = ["--workers", "1", "--data", args.data, "--checker", args.checker]
    before = len(observed["llm"]), len(observed["jev"])
    argv = [args.output, "--max-repairs", "2", "--output", str(tmp_path / "x")] + common
    monkeypatch.setenv("LLM_REASONING_EFFORT", "low")
    with pytest.raises(ValueError, match="reasoning_effort"):
        raise_limit.main(argv)
    monkeypatch.setenv("LLM_REASONING_EFFORT", "none")
    with pytest.raises(ValueError, match="larger"):
        raise_limit.main([args.output, "--max-repairs", "1", "--output", str(tmp_path / "y")] + common)
    other = deepcopy(args)
    other.output, other.max_repairs = str(tmp_path / "other"), 2
    observed["repairs_until_correct"] = 0  # another run: different first proposals
    assert (len(observed["llm"]), len(observed["jev"])) == before  # refused before any request
    cli.run(other)
    before = len(observed["llm"]), len(observed["jev"])
    with pytest.raises(ValueError, match="first proposal"):
        raise_limit.main(argv + ["--reuse", f"jev_joint={other.output}"])
    monkeypatch.setattr(cli, "_source_hash", lambda: "other-code")
    with pytest.raises(ValueError, match="different code"):
        raise_limit.main(argv)
    assert (len(observed["llm"]), len(observed["jev"])) == before and not (tmp_path / "x").exists()
