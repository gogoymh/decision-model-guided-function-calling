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
from jev_fc.reporting import build_summary, read_jsonl, update_paper


@pytest.fixture
def offline_run(tmp_path, monkeypatch):
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
    observed = {"llm": [], "jev": [], "fail_next_jev": False}

    def check_wire(body):
        text = json.dumps(body)
        assert "ORACLE_MUST_NOT_APPEAR_ON_WIRE" not in text
        for secret in ("ground_truth", "bfcl_functions", "dataset_revision", "name_map", "private_canary"):
            assert secret not in text
        assert "category" not in body and "id" not in body

    def generator_http(request):
        assert request.url.host == "mock-llm.invalid"
        body = json.loads(request.content)
        check_wire(body)
        observed["llm"].append(deepcopy(body))
        user_messages = [message for message in body["messages"] if message["role"] == "user"]
        example_id = by_prompt[user_messages[0]["content"]]
        calls = deepcopy(oracle_calls[example_id])
        if len(user_messages) == 1:
            # The initial proposal has valid JSON but missing required arguments.
            for call in calls:
                call["arguments"] = {}
        native = [{"id": f"call_{i}", "type": "function", "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}} for i, call in enumerate(calls)]
        return httpx.Response(200, json={
            "model": "offline-model", "choices": [{"finish_reason": "tool_calls" if native else "stop", "message": {"role": "assistant", "content": None, "tool_calls": native}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10},
        })

    def verifier_http(request):
        assert request.url.host == "mock-jev.invalid"
        body = json.loads(request.content)
        check_wire(body)
        assert set(body["state"]) == {"messages", "tools", "proposed_calls"}
        observed["jev"].append(deepcopy(body))
        if observed["fail_next_jev"]:
            observed["fail_next_jev"] = False
            return httpx.Response(401, json={"error": "fixture failure"})
        valid_arguments = all(call["arguments"] for call in body["state"]["proposed_calls"])
        return httpx.Response(200, json={
            "model": "jev-1.13.0",
            "answers": {"function_correct": {"type": "noul", "noul": .99}, "arguments_correct": {"type": "noul", "noul": .99 if valid_arguments else .01}},
            "usage": {"input_tokens": 300, "output_tokens": 0},
        })

    llm_http = httpx.Client(transport=httpx.MockTransport(generator_http))
    jev_http = httpx.Client(transport=httpx.MockTransport(verifier_http))
    generator = FunctionCallingLLM(api_key="offline-fixture-key", model="offline-model", base_url="https://mock-llm.invalid/v1", client=llm_http, max_retries=0)
    verifier = JevVerifier(api_key="offline-fixture-key", base_url="https://mock-jev.invalid", client=jev_http, max_retries=0)
    monkeypatch.setattr(cli, "FunctionCallingLLM", lambda: generator)
    monkeypatch.setattr(cli, "JevVerifier", lambda **_: verifier)
    monkeypatch.setattr(cli, "_source_hash", lambda: "offline-source-fingerprint")
    args = Namespace(
        data=str(data_path), checker=str(prepared / "upstream/ast_checker.py"), output=str(tmp_path / "run"),
        categories=None, split="all", split_seed=1729, limit=None,
        methods="direct,jev_joint", max_repairs=1, replicates=1, workers=1,
        threshold=.8, exhaustion_policy="execute_last", resume=False,
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
    with pytest.raises(ValueError, match="complete test runs"):
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
