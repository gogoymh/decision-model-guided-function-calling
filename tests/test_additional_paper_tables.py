"""Offline export checks use tiny fixtures, never empirical or model API results."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("additional_paper_tables", ROOT / "scripts/additional_paper_tables.py")
tables = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tables
spec.loader.exec_module(tables)


def case(identity="a"):
    return {"id": identity, "category": "apibank_firstcall", "messages": [{"role": "user", "content": "Book a room."}],
            "scorable": True, "tools": [{"type": "function", "function": {"name": "Book", "description": "Book.",
            "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"],
                           "additionalProperties": False}}}], "ground_truth": [{"name": "Book", "arguments": {"name": "A"}}]}


def proposal(name, repetition, index):
    return {"calls": [{"name": "Book", "arguments": {"name": name}}], "incomplete": None,
            "requested_model": "gpt-5-nano", "model": tables.ACTUAL_MODELS["gpt-5-nano"],
            "latency_s": 1., "retry_wait_s": .1, "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            "request_id": f"fixture-{repetition}-{index}"}


def fixture(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    examples = {identity: case(identity) for identity in ("a", "b")}
    path = data / "examples.jsonl"
    path.write_text("".join(json.dumps(example) + "\n" for example in examples.values()))
    prepared = {"preparer_sha256": tables.sha256(ROOT / "scripts/prepare_apibank.py")}
    (data / "manifest.json").write_text(json.dumps(prepared))
    dataset = tables.Dataset("apibank", path, prepared, examples, tables.prepare_apibank.score_example)
    manifest = {"format_version": "additional-matched-v1", "benchmark": "apibank", "source_sha256": tables.CORE_SHA,
                "llm_model": "gpt-5-nano", "synthetic": False, "limit": None, "max_repairs": 3, "threshold": .5,
                "final_rule": "keep_first", "exhaustion_policy": "execute_last", "temperature": None,
                "max_completion_tokens": 4096, "jev_model": "jev-1.13.0", "llm_gateway": "apim", "jev_gateway": "direct",
                "dataset_sha256": dataset.digest, "split": "test", "dataset_revisions": [tables.prepare_apibank.REVISION],
                "replicates": 3, "reasoning_effort": "none", "llm_provider": "openai", "llm_api_style": "azure_chat_completions",
                "methods": list(tables.ARMS), "shared_initial_methods": list(tables.ARMS), "ids": list(examples),
                "expected_records": 12, "external_tool_execution": False,
                "runner_sha256": tables.sha256(ROOT / "scripts/run_additional_evaluation.py"),
                "prepared_manifest_sha256": dataset.manifest_sha, "dataset_preparer_sha256": prepared["preparer_sha256"],
                "scorer_sha256": prepared["preparer_sha256"],
                "prices_usd_per_million": {"llm_input": .05, "llm_cached_input": .005, "llm_output": .4, "jev_input": .042}}
    manifest["config_id"] = tables._fingerprint(manifest)
    manifest["created_at"] = "fixture-not-an-experiment"
    rows, initials = [], []
    for repetition in range(3):
        for identity, example in examples.items():
            initial = proposal("A" if repetition == 2 else "wrong", repetition, 0)
            initials.append({"id": identity, "replicate": repetition, "config_id": manifest["config_id"],
                             "proposal": initial, "initial_wall_time_s": 1.1})
            for method in tables.ARMS:
                trace = []
                attempts = [initial] if method == "direct" or repetition == 2 else [initial, proposal("A", repetition, 1)]
                for index, current in enumerate(attempts):
                    passed = method == "direct" or index == len(attempts) - 1
                    verification = {"api_calls": 0, "latency_s": 0} if method == "direct" else {
                        "api_calls": 1, "latency_s": .2, "retry_wait_s": .05, "usage": {"input_tokens": 20},
                        "function_probability": .9, "arguments_probability": .9 if passed else .1,
                        "requested_model": "jev-1.13.0", "model": "jev-1.13.0", "function_pass": True,
                        "arguments_pass": passed, "arguments_skipped": False, "accepted": passed,
                        "feedback": {"checks": {"function_selection": {"threshold": .5, "pass": True, "probability_yes": .9, "schema_pass": True},
                                                "arguments": {"threshold": .5, "pass": passed, "probability_yes": .9 if passed else .1, "schema_pass": True, "skipped": False}}}}
                    trace.append({"attempt": index, "proposal": deepcopy(current), "verification": verification,
                                  "score": dataset.score(example, current["calls"]), "gate_pass": passed,
                                  "feedback": deepcopy(verification.get("feedback", {})),
                                  "schema": tables.validate_calls({"messages": example["messages"], "tools": example["tools"]}, current["calls"])})
                row = {"id": identity, "category": example["category"], "method": method, "replicate": repetition,
                       "trace": trace, "score": trace[-1]["score"], "calls": trace[-1]["proposal"]["calls"],
                       "final_attempt": len(trace) - 1, "final_rule": "keep_first", "max_repairs": 3,
                       "exhaustion_policy": "execute_last", "initial_proposal_shared": True, "synthetic": False,
                       "config_id": manifest["config_id"], "accepted": None if method == "direct" else True,
                       "exhausted": False, "dispatched": True, "abstained": False, "dispatch_mode": "simulated",
                       "execution_result": None, "repairs": len(trace) - 1, "wall_time_s": len(trace) * 1.3,
                       "logical_latency_s": sum(step["proposal"]["latency_s"] + step["verification"].get("latency_s", 0) for step in trace)}
                row["accounting"] = tables.token_cost(row, manifest["prices_usd_per_million"])
                rows.append(row)
    directory = tmp_path / "run"
    directory.mkdir()
    def write(name, value):
        (directory / name).write_text(json.dumps(value))
    write("manifest.json", manifest)
    (directory / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (directory / "initial.jsonl").write_text("".join(json.dumps(row) + "\n" for row in initials))
    write("summary.json", {"status": "complete", "manifest": manifest, "completed_records": 12, "expected_records": 12,
                           "methods": {method: tables.summarize_group([row for row in rows if row["method"] == method]) for method in tables.ARMS}})
    return directory, dataset, manifest, rows


def test_complete_report_statistics_and_source_immutability(tmp_path):
    directory, dataset, _, _ = fixture(tmp_path)
    before = {path.name: tables.sha256(path) for path in directory.iterdir()}
    run = tables.validate_run(directory, "gpt-5-nano", dataset)
    combined = tables.combine("apibank", "gpt-5-nano", [run], dataset)
    direct = combined["methods"]["direct"]["across_repetitions"]
    assert direct["accuracy"]["runs"] == [0, 0, 1]
    assert direct["accuracy"]["mean"] == pytest.approx(1 / 3)
    assert direct["accuracy"]["sd"] == pytest.approx(3 ** -.5)
    assert combined["methods"]["jev_joint"]["across_repetitions"]["accuracy"]["mean"] == 1
    assert combined["paired_dmfc_minus_direct"]["n_paired_ids"] == 2
    assert combined["paired_dmfc_minus_direct"]["n_paired_records"] == 6
    assert combined["paired_dmfc_minus_direct"]["delta"] == pytest.approx(2 / 3)
    assert all(key in run["provenance"] for key in ("records_sha256", "manifest_sha256", "summary_sha256", "initial_sha256"))
    assert before == {path.name: tables.sha256(path) for path in directory.iterdir()}


@pytest.mark.parametrize("failure", ["partial", "duplicate", "stale", "score", "initial", "cost", "usage", "timing", "fallback", "model", "verifier", "gate", "executed"])
def test_consequential_inconsistencies_are_rejected(tmp_path, failure):
    directory, dataset, manifest, rows = fixture(tmp_path)
    if failure == "partial":
        rows.pop()
    elif failure == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif failure == "stale":
        summary = tables._json(directory / "summary.json")
        summary["methods"]["direct"]["accuracy"] = .9
        (directory / "summary.json").write_text(json.dumps(summary))
    elif failure == "score":
        rows[0]["score"]["correct"] = True
    elif failure == "initial":
        rows[1]["trace"][0]["proposal"]["request_id"] = "foreign-initial"
    elif failure == "cost":
        rows[0]["accounting"]["estimated_usd"] = 20
    elif failure == "usage":
        rows[0]["trace"][0]["proposal"]["usage"]["completion_tokens"] = -100
        rows[0]["accounting"] = tables.token_cost(rows[0], manifest["prices_usd_per_million"])
    elif failure == "timing":
        rows[0]["logical_latency_s"] = 0
    elif failure == "fallback":
        rows[1]["final_attempt"] = 0
    elif failure == "model":
        rows[0]["trace"][0]["proposal"]["model"] = "different-model"
    elif failure == "verifier":
        rows[1]["trace"][0]["verification"]["model"] = "different-judge"
    elif failure == "gate":
        rows[1]["trace"][0]["verification"]["arguments_probability"] = .99
    else:
        rows[0]["execution_result"] = "executed"
    (directory / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        tables.validate_run(directory, "gpt-5-nano", dataset)
    assert tables._json(directory / "manifest.json") == manifest


def test_repetitions_cannot_reuse_entire_initial_cache(tmp_path):
    directory, dataset, _, _ = fixture(tmp_path)
    run = tables.validate_run(directory, "gpt-5-nano", dataset)
    run["firsts"] = {(identity, repetition): run["firsts"][(identity, 0)] for identity, repetition in run["firsts"]}
    with pytest.raises(ValueError, match="reused"):
        tables.combine("apibank", "gpt-5-nano", [run], dataset)


def test_prices_cannot_be_silently_replaced_with_free_rates(tmp_path):
    _, dataset, manifest, _ = fixture(tmp_path)
    manifest["prices_usd_per_million"] = dict.fromkeys(manifest["prices_usd_per_million"], 0)
    with pytest.raises(ValueError, match="pricing"):
        tables._check_manifest(manifest, "gpt-5-nano", dataset, False)


def test_no_call_skips_argument_judgment_and_retains_false_score(tmp_path):
    _, dataset, manifest, rows = fixture(tmp_path)
    row = rows[1]
    row["trace"] = [deepcopy(row["trace"][0]) for _ in range(4)]
    for index, step in enumerate(row["trace"]):
        step["attempt"] = index
        step["proposal"]["calls"] = []
        step["score"] = dataset.score(dataset.examples[row["id"]], [])
        step["schema"] = tables.validate_calls(dataset.examples[row["id"]], [])
        check = step["verification"]
        check.update(function_pass=False, arguments_pass=None, arguments_skipped=True, accepted=False,
                     function_probability=.1, arguments_probability=None)
        check["feedback"]["checks"] = {
            "function_selection": {"threshold": .5, "pass": False, "probability_yes": .1, "schema_pass": True},
            "arguments": {"threshold": .5, "pass": None, "probability_yes": None, "schema_pass": True, "skipped": True}}
        step["feedback"] = deepcopy(check["feedback"])
        step["gate_pass"] = False
    row.update(accepted=False, exhausted=True, repairs=3, final_attempt=0, calls=[], score=row["trace"][0]["score"], logical_latency_s=4.8)
    row["accounting"] = tables.token_cost(row, manifest["prices_usd_per_million"])
    assert tables._validate_record(row, manifest, dataset.examples[row["id"]], dataset, {})["score"]["correct"] is False
    row["trace"][0]["verification"]["arguments_probability"] = .5
    with pytest.raises(ValueError, match="skip"):
        tables._validate_record(row, manifest, dataset.examples[row["id"]], dataset, {})


def test_continuation_preserves_checkpoint_and_discloses_workers(tmp_path):
    source, _, original, _ = fixture(tmp_path)
    child = tmp_path / "child"
    child.mkdir()
    (child / "provenance").mkdir()
    settings = {key: value for key, value in original.items() if key not in {"config_id", "created_at"}}
    settings["workers"] = 8
    original = {**settings, "config_id": tables._fingerprint(settings), "created_at": "original"}
    (source / "manifest.json").write_text(json.dumps(original))
    amended_settings = {**settings, "workers": 32}
    manifest = {**amended_settings, "config_id": tables._fingerprint(amended_settings), "created_at": "child"}
    (child / "manifest.json").write_text(json.dumps(manifest))
    for name in ("records.jsonl", "initial.jsonl"):
        rows = list(tables._rows(source / name))
        (child / name).write_text("".join(json.dumps({**row, "config_id": manifest["config_id"],
            "original_config_id": original["config_id"], "recorded_workers": 8}) + "\n" for row in rows))
    (child / "provenance/original_manifest.json").write_bytes((source / "manifest.json").read_bytes())
    history = {"format_version": "additional-concurrency-continuation-v1", "child_config_id": manifest["config_id"],
               "original_config_id": original["config_id"], "source_directory": str(source),
               "child_manifest_sha256": tables.sha256(child / "manifest.json"),
               "script_sha256": tables.sha256(ROOT / "scripts/prepare_concurrency_continuation.py"),
               "source_files_sha256": {name: tables.sha256(source / name) for name in ("manifest.json", "records.jsonl", "initial.jsonl")},
               "settings_changes": {"workers": {"before": 8, "after": 32}},
               "record_amendments": ["config_id", "original_config_id", "recorded_workers"],
               "counts": {"completed_method_records": 12, "cached_initials": 6},
               "runtime_history": [{"workers": 8}, {"workers": 32}]}
    (child / "provenance/concurrency_continuation.json").write_text(json.dumps(history))
    result = tables.validate_continuation(child, manifest)
    assert result["runtime_history"] == [{"workers": 8}, {"workers": 32}]
    amended = list(tables._rows(child / "records.jsonl"))
    amended[0]["logical_latency_s"] += 1
    (child / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in amended))
    with pytest.raises(ValueError, match="preserved"):
        tables.validate_continuation(child, manifest)


def test_unknown_usage_cost_is_explicit_and_not_zero(tmp_path):
    directory, dataset, _, _ = fixture(tmp_path)
    run = tables.validate_run(directory, "gpt-5-nano", dataset)
    direct = [row for row in run["rows"] if row["method"] == "direct"]
    direct[0]["accounting"]["complete"] = False
    direct[0]["accounting"]["estimated_usd"] = None
    result = tables.metrics(direct)
    assert result["mean_cost_usd"] is None
    assert result["unknown_cost_n"] == 1
    assert result["mean_known_cost_usd"] > 0
    for row in direct:
        row["accounting"]["complete"] = False
        row["accounting"]["estimated_usd"] = None
    assert tables.metrics(direct)["mean_known_cost_usd"] is None


def test_keep_first_retains_all_repair_cost_and_latency(tmp_path):
    _, dataset, manifest, rows = fixture(tmp_path)
    row = rows[1]
    row["trace"] = [deepcopy(row["trace"][0]) for _ in range(4)]
    for index, step in enumerate(row["trace"]):
        step["attempt"] = index
        step["gate_pass"] = False
    row.update(accepted=False, exhausted=True, repairs=3, final_attempt=0,
               calls=row["trace"][0]["proposal"]["calls"], score=row["trace"][0]["score"], logical_latency_s=4.8)
    row["accounting"] = tables.token_cost(row, manifest["prices_usd_per_million"])
    compact = tables._validate_record(row, manifest, dataset.examples[row["id"]], dataset, {})
    assert compact["logical_latency_s"] == 4.8
    assert compact["score"]["correct"] is False
    assert compact["accounting"]["estimated_usd"] == row["accounting"]["estimated_usd"]
    assert len(compact["trace"]) == 4


def test_anthropic_cache_reads_and_writes_count_once(tmp_path):
    _, _, manifest, rows = fixture(tmp_path)
    row = rows[0]
    row["trace"][0]["proposal"]["usage"] = {"prompt_tokens": 300, "completion_tokens": 10,
        "prompt_tokens_details": {"cached_tokens": 100}, "cache_creation_input_tokens": 150,
        "cache_creation": {"ephemeral_5m_input_tokens": 150, "ephemeral_1h_input_tokens": 0}}
    rates = {**manifest["prices_usd_per_million"], "llm_input": 2, "llm_cached_input": .1,
             "llm_output": 10, "llm_cache_write_5m": 2.5, "llm_cache_write_1h": 4}
    result = tables.token_cost(row, rates)
    assert result["estimated_usd"] == pytest.approx((50 * 2 + 100 * .1 + 150 * 2.5 + 10 * 10) / 1e6)
    assert result["complete"] is True
    row["trace"][0]["proposal"]["usage"].pop("cache_creation")
    assert tables.token_cost(row, rates)["estimated_usd"] is None


def table_report(tmp_path):
    directory, dataset, _, _ = fixture(tmp_path)
    row = tables.combine("apibank", "gpt-5-nano", [tables.validate_run(directory, "gpt-5-nano", dataset)], dataset)
    rows = []
    for benchmark in ("bfcl", "apibank"):
        for model in tables.MODELS:
            rows.append({**deepcopy(row), "benchmark": benchmark, "generator": model, "generator_display": tables.DISPLAYS[model]})
    return {"status": "complete", "rows": rows}


def test_six_row_table_standard_font_no_scaling_and_unknown_dash(tmp_path):
    report = table_report(tmp_path)
    for method in tables.ARMS:
        report["rows"][-1]["methods"][method]["across_repetitions"]["mean_known_cost_usd"]["mean"] = None
    table = tables.latex_table(report)
    assert r"\begin{table*}" in table and r"\centering\footnotesize" in table
    assert "resizebox" not in table and "scriptsize" not in table
    assert table.count(" & nano & ") == 2 and table.count(" & mini & ") == 2 and table.count(" & Sonnet 5.5 & ") == 2
    assert r"\textemdash / \textemdash" in table
    assert "156 cases" in table and "exploratory adapted exact match" in table


def test_optional_paper_update_changes_categories_only(tmp_path):
    table = tables.latex_table(table_report(tmp_path))
    source = "ABSTRACT unchanged\n% BEGIN AUTO CATEGORIES\nold\n% END AUTO CATEGORIES\n% BEGIN AUTO RESULTS\nTable2\n% END AUTO RESULTS\n% BEGIN AUTO REPAIRS\nFigure2\n% END AUTO REPAIRS\n"
    paper = tmp_path / "main.tex"
    paper.write_text(source)
    tables.update_paper(paper, table)
    updated = paper.read_text()
    assert updated.split("% BEGIN AUTO CATEGORIES")[0] == source.split("% BEGIN AUTO CATEGORIES")[0]
    assert updated.split("% END AUTO CATEGORIES")[1] == source.split("% END AUTO CATEGORIES")[1]
    assert tables.MARKER in updated


def test_incomplete_main_creates_no_report_or_paper_change(tmp_path, monkeypatch):
    paper = tmp_path / "main.tex"
    paper.write_text("preserve")
    def unavailable(_):
        raise ValueError("Actual records are incomplete")
    monkeypatch.setattr(tables, "build_report", unavailable)
    with pytest.raises(ValueError, match="incomplete"):
        tables.main(["--root", str(tmp_path), "--paper", str(paper)])
    assert paper.read_text() == "preserve" and not (tmp_path / "results").exists()


def test_report_paths_cannot_overwrite_source_artifacts(tmp_path):
    source = tmp_path / "results/additional/live"
    source.mkdir(parents=True)
    (source / "manifest.json").write_text("preserve")
    for target in (source / "records.jsonl", tmp_path / "data/manifest.json", tmp_path / "paper/main.tex", tmp_path / "results/final/final.json"):
        with pytest.raises(ValueError, match="overwrite"):
            tables._output_paths_safe((target, tmp_path / "table.tex"), tmp_path, None)
    tables._output_paths_safe((tmp_path / "results/additional/combined-main.json", tmp_path / "results/additional/combined-main.tex"), tmp_path, None)


def test_pinned_data_load_is_offline_and_adapter_preserves_labels():
    if not (ROOT / "data/apibank-firstcall-candidate").exists():
        pytest.skip("Pinned local data unavailable; tests never download it.")
    datasets = tables.load_datasets(ROOT)
    assert len(datasets["bfcl"].ids) == 3051
    assert len(datasets["apibank"].ids) == 156
    assert all(example["messages"][-1] == {"role": "user", "content": tables.input_adapter.INSTRUCTION}
               for example in datasets["apibank"].examples.values())
