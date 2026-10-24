"""Read-only validation and six-row export of completed cross-benchmark results.

Historical BFCL runs remain immutable. New results must use the common native-candidate
API-Bank adapter, all prespecified cases, and three paired repetitions. The
default writes a JSON report and a standalone LaTeX table only; --paper changes
only the CATEGORIES auto block after every source has passed validation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from jev_fc.benchmarks import BFCL_REVISION, load_examples  # noqa: E402
from jev_fc.cli import _fingerprint, _source_hash  # noqa: E402
from jev_fc.jev import validate_calls  # noqa: E402
from jev_fc.pipeline import METHODS  # noqa: E402
from jev_fc.reporting import paired_delta, percentile, summarize_group, token_cost  # noqa: E402
from jev_fc.scoring import score_example as score_bfcl  # noqa: E402
import prepare_apibank  # noqa: E402
import prepare_apibank_candidate_adapter as input_adapter  # noqa: E402
import run_claude_additional_evaluation as claude_adapter  # noqa: E402

CORE_SHA = "c167522df5f566dc0797b8124cb2b1d8ebbad0ab25133b85f58343f5228c8342"
BFCL_SHA = "4cad37e0a68506c8ce6369a454796a3695327eb860e9645430b120e5f3ef28e3"
ADAPTED_MANIFEST_SHA = "a931e2c5ad3c854c5638145d273da3e7925ff3261072340c5a8a9464bc7953fd"
MODELS = ("gpt-5-nano", "gpt-5.4-mini", "claude-sonnet-5-5")
DISPLAYS = dict(zip(MODELS, ("nano", "mini", "Sonnet 5.5")))
ACTUAL_MODELS = dict(zip(MODELS, ("gpt-5-nano-2025-08-07", "gpt-5.4-mini-2026-03-17", "claude-sonnet-5-5")))
PRICES = {
    "gpt-5-nano": {"llm_input": .05, "llm_cached_input": .005, "llm_output": .4, "jev_input": .042},
    "gpt-5.4-mini": {"llm_input": .75, "llm_cached_input": .075, "llm_output": 4.5, "jev_input": .042},
    "claude-sonnet-5-5": {"llm_input": 2., "llm_cached_input": .1, "llm_output": 10.,
                           "llm_cache_write_5m": 2.5, "llm_cache_write_1h": 4., "jev_input": .042},
}
ARMS = ("direct", "jev_joint")
MARKER = "% COMBINED MAIN TABLE: V1"
SCORE_KEYS = ("correct", "function_correct", "arguments_correct", "presence_correct", "metric", "checker_revision")
CHECK_SUMMARY = ("n", "exact_n", "accuracy", "function_accuracy", "arguments_accuracy_conditional",
                 "mean_latency_s", "p95_latency_s", "mean_cost_usd", "cost_observed_n", "mean_repairs",
                 "coverage", "accepted_precision", "exhaustion_rate", "incomplete_proposal_n")
SAME_MODEL = ("source_sha256", "dataset_sha256", "dataset_revisions", "llm_model", "llm_provider",
              "llm_api_style", "llm_gateway", "llm_endpoint", "reasoning_effort", "temperature",
              "max_completion_tokens", "jev_model", "jev_endpoint", "threshold", "max_repairs",
              "final_rule", "exhaustion_policy", "prices_usd_per_million")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return _fingerprint(value)


def _json(path):
    return json.loads(Path(path).read_text())


def _rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (float, int)) and math.isfinite(value) and value >= 0


def _usage_counts(usage):
    _require(isinstance(usage, dict), "Usage must be a dictionary when reported.")
    for key, value in usage.items():
        if key in {"prompt_tokens_details", "input_tokens_details", "cache_creation"}:
            if value is not None:
                _usage_counts(value)
        elif key.endswith("tokens") or key in {"prompt_tokens", "completion_tokens", "input_tokens", "output_tokens"}:
            _require(value is None or (not isinstance(value, bool) and isinstance(value, int) and value >= 0),
                     "Token usage must be a nonnegative integer or unknown.")


def _equal(left, right):
    if isinstance(left, (float, int)) and not isinstance(left, bool) and isinstance(right, (float, int)) and not isinstance(right, bool):
        return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)
    return left == right


def spread(values):
    known = bool(values) and all(value is not None for value in values)
    return {"runs": values, "mean": statistics.fmean(values) if known else None,
            "sd": statistics.stdev(values) if known and len(values) > 1 else None}


@dataclass
class Dataset:
    benchmark: str
    path: Path
    prepared_manifest: dict
    examples: dict
    score: object

    @property
    def digest(self):
        return sha256(self.path)

    @property
    def manifest_sha(self):
        return sha256(self.path.parent / "manifest.json")

    @property
    def ids(self):
        return set(self.examples)


def load_datasets(root):
    """Validate canonical data, the audited adapter, and public schemas without inference."""
    _require(_source_hash() == CORE_SHA, "Frozen core code changed; existing results cannot populate this table.")
    bfcl_path = root / "data/bfcl/examples.jsonl"
    bfcl_manifest = _json(bfcl_path.parent / "manifest.json")
    _require(sha256(bfcl_path) == BFCL_SHA == bfcl_manifest["normalized_sha256"][bfcl_path.name], "BFCL data digest changed.")
    bfcl = load_examples(bfcl_path)
    _require(len(bfcl) == 3051 and sum(row["category"] != "live_relevance" for row in bfcl) == 3035,
             "BFCL canonical accuracy/timing denominators changed.")
    checker = root / "data/bfcl/upstream/ast_checker.py"
    _require(sha256(checker) == bfcl_manifest["sources"]["upstream/ast_checker.py"]["sha256"], "Official BFCL checker changed.")
    apibank_path = root / "data/apibank-firstcall-candidate/examples.jsonl"
    adapted = _json(apibank_path.parent / "manifest.json")
    _require(sha256(apibank_path.parent / "manifest.json") == ADAPTED_MANIFEST_SHA, "API-Bank common-input manifest changed.")
    adapter = adapted.get("input_adapter") or {}
    _require(adapted.get("revision") == prepare_apibank.REVISION and adapted.get("format_version") == "apibank-firstcall-v1"
             and adapted.get("preparer_sha256") == sha256(Path(prepare_apibank.__file__)), "API-Bank preparation/scorer changed.")
    _require(adapter.get("format_version") == "apibank-native-candidate-v1"
             and adapter.get("script_sha256") == sha256(Path(input_adapter.__file__))
             and adapter.get("instruction") == input_adapter.INSTRUCTION
             and adapter.get("amended_after_test_outcomes_inspected") is True,
             "API-Bank requires the common native-candidate adapter; protocol pilots cannot enter the table.")
    original_path = root / "data/apibank-firstcall/examples.jsonl"
    original_manifest_path = original_path.parent / "manifest.json"
    _require(sha256(original_manifest_path) == adapter["source_manifest_sha256"]
             == sha256(apibank_path.parent / adapter["source_manifest_copy"]), "API-Bank original preparation provenance changed.")
    original = load_examples(original_path)
    adapted_examples = load_examples(apibank_path)
    _require(sha256(apibank_path) == adapted["files_sha256"][apibank_path.name], "API-Bank adapted examples changed.")
    _require(len(original) == len(adapted_examples) == 204 and adapted["counts"] == {
        "dialogue_files": 214, "first_api_targets": 212, "no_api_files": 2, "schema_excluded": 8,
        "scorable": 204, "dev": 48, "test": 156, "catalog_tools": 53}, "API-Bank audited denominators changed.")
    for before, after in zip(original, adapted_examples):
        expected = {**before, "messages": before["messages"] + [{"role": "user", "content": input_adapter.INSTRUCTION}]}
        _require(expected == after, "API-Bank adapter changed labels, catalog, or original conversation.")
    catalog = _json(apibank_path.parent / "tools.json")
    _require(sha256(apibank_path.parent / "tools.json") == adapted["files_sha256"]["tools.json"] and len(catalog) == 53,
             "API-Bank fixed catalog changed.")
    test = [row for row in adapted_examples if row["split"] == "test"]
    _require(len(test) == 156 and all(row["scorable"] and row["tools"] == catalog
                                    and row["split"] == prepare_apibank.family_split(row["source_file"])[1] for row in adapted_examples),
             "API-Bank test split/catalog changed.")
    for row in (*bfcl, *test):
        validate_calls({"messages": row["messages"], "tools": row["tools"]}, [])
    return {
        "bfcl": Dataset("bfcl", bfcl_path, bfcl_manifest, {row["id"]: row for row in bfcl},
                        lambda example, calls: score_bfcl(example, calls, checker_path=checker)),
        "apibank": Dataset("apibank", apibank_path, adapted, {row["id"]: row for row in test}, prepare_apibank.score_example),
    }


def _check_manifest(manifest, model, dataset, legacy):
    _require(model in MODELS, "Unsupported generator for the predeclared matrix.")
    expected = {"source_sha256": CORE_SHA, "llm_model": model, "synthetic": False, "limit": None,
                "max_repairs": 3, "threshold": .5, "final_rule": "keep_first", "exhaustion_policy": "execute_last",
                "temperature": None, "max_completion_tokens": 4096, "jev_model": "jev-1.13.0",
                "llm_gateway": "apim", "jev_gateway": "direct", "dataset_sha256": dataset.digest,
                "split": "all" if dataset.benchmark == "bfcl" else "test",
                "dataset_revisions": [BFCL_REVISION if dataset.benchmark == "bfcl" else prepare_apibank.REVISION],
                "replicates": 1 if legacy else 3, "reasoning_effort": None if model == MODELS[2] else "none",
                "llm_provider": "anthropic" if model == MODELS[2] else "openai",
                "llm_api_style": "anthropic_messages" if model == MODELS[2] else "azure_chat_completions"}
    for key, value in expected.items():
        _require(manifest.get(key) == value, f"Run differs from the predeclared protocol: {key}.")
    _require(all(manifest.get("prices_usd_per_million", {}).get(key) == value for key, value in PRICES[model].items()),
             "Run pricing differs from the declared generator/verifier rates.")
    methods = manifest.get("methods")
    _require(isinstance(methods, list) and len(methods) == len(set(methods)) and set(ARMS) <= set(methods) <= set(METHODS),
             "Run methods do not contain the required paired arms.")
    _require(set(manifest.get("shared_initial_methods", [])) == set(methods), "Initial proposals must be shared across run arms.")
    _require(len(manifest.get("ids", [])) == len(dataset.ids) and set(manifest["ids"]) == dataset.ids,
             "Run does not cover every prespecified dataset ID.")
    _require(manifest.get("expected_records") == len(dataset.ids) * manifest["replicates"] * len(methods), "Run expected grid changed.")
    settings = {key: value for key, value in manifest.items() if key != "config_id" and (legacy or key != "created_at")}
    _require(manifest.get("config_id") == _fingerprint(settings), "Run manifest fingerprint does not match its settings.")
    if legacy:
        _require(dataset.benchmark == "bfcl" and not manifest.get("format_version")
                 and manifest.get("derived", {}).get("rule") == "keep_first", "Not a historical final BFCL run.")
    else:
        _require(methods == list(ARMS) and manifest.get("format_version") == "additional-matched-v1"
                 and manifest.get("benchmark") == dataset.benchmark and manifest.get("external_tool_execution") is False,
                 "Only the new nonexecuting matched experiment format is supported.")
        _require(manifest.get("runner_sha256") == sha256(ROOT / "scripts/run_additional_evaluation.py")
                 and manifest.get("prepared_manifest_sha256") == dataset.manifest_sha, "Runner/prepared-data provenance changed.")
        scorer = ROOT / ("src/jev_fc/scoring.py" if dataset.benchmark == "bfcl" else "scripts/prepare_apibank.py")
        _require(manifest.get("scorer_sha256") == sha256(scorer), "Scoring code changed.")
        if dataset.benchmark == "apibank":
            _require(manifest.get("dataset_preparer_sha256") == dataset.prepared_manifest["preparer_sha256"], "API-Bank preparer changed.")
        else:
            checker = dataset.path.parent / "upstream/ast_checker.py"
            _require(manifest.get("checker_sha256") == sha256(checker), "BFCL checker provenance changed.")
            if model == MODELS[2]:
                _require(manifest.get("provider_adapter_sha256") == sha256(Path(claude_adapter.__file__))
                         and manifest.get("provider_input_rule") == claude_adapter.PROVIDER_INPUT_RULE,
                         "BFCL Claude must disclose its authenticated native-schema compatibility wrapper.")


def _score_schema(score, dataset, category):
    _require(isinstance(score, dict) and isinstance(score.get("correct"), bool), "Missing boolean correctness score.")
    for key in ("function_correct", "arguments_correct"):
        _require(score.get(key) is None or isinstance(score[key], bool), "Invalid conditional score schema.")
    if dataset.benchmark == "bfcl":
        metric = "bfcl_call_presence" if category in {"irrelevance", "live_irrelevance", "live_relevance"} else "bfcl_python_ast"
        _require(score.get("metric") == metric and score.get("checker_revision") == BFCL_REVISION, "BFCL score metric/revision changed.")
    else:
        _require(score.get("metric") == "apibank_firstcall_exact", "API-Bank score must be adapted exact first-call match.")


def _validate_record(row, manifest, example, dataset, score_cache):
    """Validate the whole actual trajectory, then retain only fields needed for offline statistics."""
    _require(row.get("synthetic") is False and row.get("config_id") == manifest["config_id"]
             and row.get("category") == example["category"], "Record identity/category/provenance changed.")
    _require(row.get("dispatch_mode") == "simulated" and row.get("execution_result") is None
             and row.get("dispatched") is True and row.get("abstained") is False, "Benchmark must not execute tools or abstain.")
    _require(row.get("initial_proposal_shared") is True and row.get("max_repairs") == 3
             and row.get("exhaustion_policy") == "execute_last" and row.get("final_rule") == "keep_first", "Record policy changed.")
    trace = row.get("trace")
    _require(isinstance(trace, list) and 1 <= len(trace) <= 4 and row.get("repairs") == len(trace) - 1,
             "Record repair count/trace length changed.")
    direct = row["method"] == "direct"
    _require((len(trace) == 1 and row.get("accepted") is None and row.get("exhausted") is False) if direct else
             (isinstance(row.get("accepted"), bool) and all(step.get("gate_pass") is False for step in trace[:-1])
              and trace[-1].get("gate_pass") is row["accepted"]
              and row.get("exhausted") is (not row["accepted"])
              and (row["accepted"] or len(trace) == 4)), "Record stopping rule changed.")
    final = 0 if not direct and row["accepted"] is False else len(trace) - 1
    _require(not isinstance(row.get("final_attempt"), bool) and row.get("final_attempt") == final
             and row.get("calls") == trace[final]["proposal"]["calls"] and row.get("score") == trace[final]["score"],
             "Returned call/score does not follow first-passing or keep_first fallback.")
    logical = 0.
    for index, step in enumerate(trace):
        _require(step.get("attempt") == index, "Trace attempt order changed.")
        proposal, verification = step["proposal"], step["verification"]
        _require(proposal.get("requested_model") == manifest["llm_model"]
                 and proposal.get("model") in {ACTUAL_MODELS[manifest["llm_model"]], None}
                 and (proposal.get("model") is not None or proposal.get("incomplete")), "Emitted/requested generator identity changed.")
        local = validate_calls({"messages": example["messages"], "tools": example["tools"]}, proposal["calls"])
        _require(step.get("schema") == local, "Recorded local schema decision differs from public schema validation.")
        if not direct:
            _require(verification.get("requested_model") == "jev-1.13.0"
                     and verification.get("model") == ("jev-1.13.0" if local["well_formed"] else None),
                     "Actual/requested verifier identity changed.")
            fp, ap = verification.get("function_probability"), verification.get("arguments_probability")
            for probability in (fp, ap):
                _require(probability is None or (_finite(probability) and probability <= 1), "Invalid verifier probability.")
            skipped = proposal["calls"] == []
            _require(not skipped or ap is None, "No-call candidates must skip the argument probability.")
            function_pass = bool(local["function_pass"] and fp is not None and fp >= .5)
            arguments_pass = None if skipped else bool(local["arguments_pass"] and ap is not None and ap >= .5)
            accepted = bool(local["accepted"] and function_pass and (skipped or arguments_pass))
            _require(verification.get("function_pass") is function_pass and verification.get("arguments_pass") is arguments_pass
                     and verification.get("arguments_skipped") is skipped and verification.get("accepted") is accepted
                     and step.get("gate_pass") is accepted, "Recorded acceptance does not apply both .5 checks to the same proposal.")
            checks = verification.get("feedback", {}).get("checks", {})
            _require(checks.get("function_selection", {}).get("threshold") == .5
                     and checks.get("arguments", {}).get("threshold") == .5, "Recorded feedback threshold changed.")
            _require(step.get("feedback") == verification.get("feedback"), "Recorded repair feedback differs from the actual verifier feedback.")
            for kind, values in (("function_selection", {"pass": function_pass, "probability_yes": fp, "schema_pass": local["function_pass"]}),
                                 ("arguments", {"pass": arguments_pass, "probability_yes": ap, "schema_pass": local["arguments_pass"], "skipped": skipped})):
                _require(all(checks.get(kind, {}).get(key) == value for key, value in values.items()), "Recorded repair feedback decisions differ from the actual checks.")
        if proposal.get("incomplete") == "unsupported_schema":
            error = proposal.get("provider_input_error") or {}
            _require(manifest.get("provider_adapter_sha256") == sha256(Path(claude_adapter.__file__))
                     and claude_adapter.unsupported_property_names(example) and error.get("status_code") == 400
                     and error.get("request_sent") is True and proposal.get("provider_adapter") == "claude-native-schema-v1",
                     "Unsupported-schema proposal lacks actual authenticated provider-rejection evidence.")
        for event in (proposal, verification):
            _require(_finite(event.get("latency_s", 0)) and _finite(event.get("retry_wait_s", 0)), "Invalid API timing.")
            _usage_counts(event.get("usage") or {})
            logical += event.get("latency_s", 0)
        _score_schema(step.get("score"), dataset, example["category"])
        cache_key = (example["id"], _digest(proposal["calls"]))
        if cache_key not in score_cache:
            score_cache[cache_key] = dataset.score(example, proposal["calls"])
        _require(all(step["score"].get(key) == score_cache[cache_key].get(key) for key in SCORE_KEYS), "Recorded score differs from offline canonical scoring.")
    _require(_finite(row.get("logical_latency_s")) and _equal(logical, row["logical_latency_s"]), "API time must include every actual attempt.")
    _require(_finite(row.get("wall_time_s")), "Invalid recorded wall time.")
    accounting = token_cost(row, manifest["prices_usd_per_million"])
    _require(set(row.get("accounting", {})) == set(accounting)
             and all(_equal(row["accounting"][key], value) for key, value in accounting.items()), "Recorded token/cost accounting changed.")
    compact_trace = [{"score": step["score"], "gate_pass": step["gate_pass"],
                      "proposal": {key: step["proposal"].get(key) for key in ("calls", "incomplete", "provider_input_error")},
                      "verification": {key: step["verification"][key] for key in
                                       ("function_probability", "arguments_probability", "api_calls")
                                       if key in step["verification"]}}
                     for step in trace]
    return {**{key: row[key] for key in ("id", "category", "method", "replicate", "score", "accepted", "exhausted",
                                       "dispatched", "repairs", "logical_latency_s", "wall_time_s", "calls", "final_attempt")},
            "accounting": accounting, "trace": compact_trace,
            "retry_wait_s": sum(step["proposal"].get("retry_wait_s", 0) + step["verification"].get("retry_wait_s", 0) for step in trace),
            "schema_error_n": sum(not step["schema"]["accepted"] for step in trace),
            "provider_input_error_n": sum(step["proposal"].get("incomplete") == "unsupported_schema" for step in trace),
            "retry_timing_unknown_n": sum((step["proposal"].get("provider_input_error") or {}).get("retry_timing_known") is False for step in trace),
            "recorded_workers": row.get("recorded_workers", manifest.get("workers")),
            "empty_final": row["calls"] == [], "extra_final": isinstance(row["calls"], list) and len(row["calls"]) > 1}


def metrics(rows, *, legacy=False):
    result = summarize_group(rows)
    exact = [row for row in rows if row["category"] != "live_relevance"]
    known = [row["accounting"]["estimated_usd"] for row in rows if row["accounting"]["complete"]]
    result.update(p50_latency_s=percentile([row["logical_latency_s"] for row in rows], .5),
                  mean_known_cost_usd=statistics.fmean(known) if known else None,
                  unknown_cost_n=len(rows) - len(known),
                  mean_retry_wait_s=statistics.fmean(row["retry_wait_s"] for row in rows) if rows else None,
                  schema_error_proposal_n=sum(row["schema_error_n"] for row in rows),
                  provider_input_error_proposal_n=sum(row["provider_input_error_n"] for row in rows),
                  retry_timing_unknown_proposal_n=sum(row["retry_timing_unknown_n"] for row in rows),
                  empty_call_rate=statistics.fmean(row["empty_final"] for row in rows) if rows else None,
                  extra_call_rate=statistics.fmean(row["extra_final"] for row in rows) if rows else None,
                  incomplete_final_n=sum(bool(row["trace"][row["final_attempt"]]["proposal"]["incomplete"]) for row in rows),
                  function_change_rate=statistics.fmean(_names(row["trace"][0]["proposal"]["calls"]) != _names(row["calls"]) for row in exact) if exact else None,
                  mean_wall_time_s=None if legacy or not rows else statistics.fmean(row["wall_time_s"] for row in rows),
                  p50_wall_time_s=None if legacy else percentile([row["wall_time_s"] for row in rows], .5),
                  p95_wall_time_s=None if legacy else percentile([row["wall_time_s"] for row in rows], .95),
                  wall_scope="Unavailable: historical method wall excludes cached initial generation" if legacy else "Standalone arm including shared initial; scoring/queue excluded")
    return result


def _names(calls):
    return sorted(str(call.get("name")) for call in calls) if isinstance(calls, list) else None


def validate_continuation(path, manifest):
    """Authenticate changed-concurrency metadata and every preserved raw checkpoint observation."""
    history_path = path / "provenance/concurrency_continuation.json"
    if not history_path.exists():
        return None
    history = _json(history_path)
    source = Path(history["source_directory"])
    _require(history.get("format_version") == "additional-concurrency-continuation-v1"
             and history.get("child_config_id") == manifest["config_id"]
             and history.get("child_manifest_sha256") == sha256(path / "manifest.json")
             and history.get("script_sha256") == sha256(ROOT / "scripts/prepare_concurrency_continuation.py"),
             "Continuation provenance/helper/child fingerprint changed.")
    original = _json(source / "manifest.json")
    _require(original["config_id"] == history["original_config_id"]
             and sha256(source / "manifest.json") == sha256(path / "provenance/original_manifest.json"),
             "Continuation original manifest copy changed.")
    for name, digest in history["source_files_sha256"].items():
        _require((digest is None and not (source / name).exists()) or (digest is not None and sha256(source / name) == digest),
                 "Continuation source checkpoint changed.")
    before = {key: value for key, value in original.items() if key not in {"config_id", "created_at"}}
    after = {key: value for key, value in manifest.items() if key not in {"config_id", "created_at"}}
    actual_changes = {key: {"before": before.get(key), "after": after.get(key)} for key in set(before) | set(after) if before.get(key) != after.get(key)}
    _require(actual_changes == history["settings_changes"] and set(actual_changes) <= {"workers", "provider_adapter_sha256", "provider_input_rule"}
             and history.get("record_amendments") == ["config_id", "original_config_id", "recorded_workers"], "Continuation changed undeclared experiment settings.")
    counts = {}
    for name in ("records.jsonl", "initial.jsonl"):
        parent = list(_rows(source / name))
        child = iter(_rows(path / name))
        for row in parent:
            amended = next(child, None)
            expected = {**row, "config_id": manifest["config_id"], "original_config_id": original["config_id"], "recorded_workers": original["workers"]}
            _require(amended == expected, "Continuation altered an actual preserved initial/trajectory observation.")
        counts[name] = len(parent)
        _require(all("original_config_id" not in row and "recorded_workers" not in row for row in child), "Continuation has extra undeclared preserved observations.")
    _require(counts["records.jsonl"] == history["counts"]["completed_method_records"]
             and counts["initial.jsonl"] == history["counts"]["cached_initials"], "Continuation checkpoint count changed.")
    return {**history, "provenance_sha256": sha256(history_path)}


def validate_run(path, model, dataset, *, legacy=False, root=ROOT):
    path = Path(path)
    manifest, summary = _json(path / "manifest.json"), _json(path / "summary.json")
    _check_manifest(manifest, model, dataset, legacy)
    continuation = validate_continuation(path, manifest) if not legacy else None
    if not legacy and dataset.benchmark == "bfcl" and model == MODELS[2]:
        _require(continuation is not None, "BFCL Claude publication requires its preserved checkpoint/concurrency provenance.")
    _require(summary.get("status") == "complete" and summary.get("manifest") == manifest
             and summary.get("expected_records") == summary.get("completed_records") == manifest["expected_records"],
             "Result summary is missing, partial, or stale.")
    expected = {(identity, repetition, method) for identity in dataset.ids
                for repetition in range(manifest["replicates"]) for method in manifest["methods"]}
    seen, firsts, compact, cache, actual_models, recorded_observations = set(), {}, [], {}, set(), {}
    for row in _rows(path / "records.jsonl"):
        key = (row.get("id"), row.get("replicate"), row.get("method"))
        _require(key in expected and key not in seen and not isinstance(row.get("replicate"), bool), "Duplicate, foreign, or malformed actual result grid.")
        seen.add(key)
        _require(row.get("config_id") == manifest["config_id"] and row.get("synthetic") is False, "Foreign or synthetic record.")
        _require(isinstance(row.get("trace"), list) and row["trace"], "Missing or malformed actual proposal trace.")
        first = row["trace"][0]
        initial_key, initial_value = key[:2], (_digest(first["proposal"]), _digest(first["score"]))
        _require(firsts.setdefault(initial_key, initial_value) == initial_value, "Methods do not share the same initial proposal and score.")
        if row["method"] in ARMS:
            compact.append(_validate_record(row, manifest, dataset.examples[row["id"]], dataset, cache))
            actual_models.update(step["proposal"]["model"] for step in row["trace"] if step["proposal"].get("model"))
            recorded_observations[key] = _digest({name: row[name] for name in ("trace", "accounting", "logical_latency_s", "wall_time_s", "repairs", "accepted", "exhausted")})
    _require(seen == expected, "Actual records are incomplete despite their summary; publication requires the complete grid.")
    provenance = {"run_dir": str(path.resolve()), "manifest_sha256": sha256(path / "manifest.json"),
                  "records_sha256": sha256(path / "records.jsonl"), "summary_sha256": sha256(path / "summary.json"),
                  "config_id": manifest["config_id"], "declared_run_dir": summary.get("run_dir"),
                  "emitted_models": sorted(actual_models), "dataset_sha256": dataset.digest,
                  "prepared_manifest_sha256": dataset.manifest_sha, "source_sha256": manifest["source_sha256"]}
    if legacy:
        parts, methods_in_parts, source_observations = [], [], {}
        _require(isinstance(manifest["derived"].get("parts"), list) and manifest["derived"]["parts"], "Historical derivation has no measured source parts.")
        for part in manifest["derived"]["parts"]:
            methods_in_parts.extend(part["methods"])
            source = root / part["run_dir"]
            _require(sha256(source / "records.jsonl") == part["records_sha256"]
                     and _json(source / "manifest.json")["config_id"] == part["config_id"], "Historical source part changed after final derivation.")
            parts.append({**part, "manifest_sha256": sha256(source / "manifest.json")})
            for row in _rows(source / "records.jsonl"):
                if row["method"] in part["methods"] and row["method"] in ARMS:
                    key = (row["id"], row["replicate"], row["method"])
                    _require(key not in source_observations, "Duplicate measured source arm in historical derivation.")
                    source_observations[key] = _digest({name: row[name] for name in ("trace", "accounting", "logical_latency_s", "wall_time_s", "repairs", "accepted", "exhausted")})
        _require(sorted(methods_in_parts) == sorted(manifest["methods"]), "Historical source parts do not uniquely cover every arm.")
        _require(recorded_observations == source_observations, "Final trajectory differs from its actual measured historical source.")
        provenance["derived_parts"] = parts
    else:
        initials = {}
        for initial in _rows(path / "initial.jsonl"):
            key = (initial.get("id"), initial.get("replicate"))
            _require(key in firsts and key not in initials and initial.get("config_id") == manifest["config_id"]
                     and _finite(initial.get("initial_wall_time_s")), "Invalid/foreign initial cache.")
            initials[key] = _digest(initial["proposal"])
            _require(initials[key] == firsts[key][0], "Record initial differs from the saved shared cache.")
        _require(set(initials) == set(firsts), "Initial cache is incomplete.")
        provenance["initial_sha256"] = sha256(path / "initial.jsonl")
        provenance["runner_sha256"] = manifest["runner_sha256"]
        provenance["scorer_sha256"] = manifest["scorer_sha256"]
        provenance["workers"] = manifest.get("workers")
        provenance["recorded_workers_counts"] = dict(sorted(Counter(row["recorded_workers"] for row in compact).items()))
        if continuation:
            provenance["continuation"] = continuation
        if manifest.get("provider_adapter_sha256"):
            provenance["provider_adapter_sha256"] = manifest["provider_adapter_sha256"]
            provenance["provider_input_rule"] = manifest["provider_input_rule"]
        if (path / "errors.jsonl").exists():
            provenance["errors_sha256"] = sha256(path / "errors.jsonl")
    for method in ARMS:
        observed = metrics([row for row in compact if row["method"] == method], legacy=legacy)
        _require(all(_equal(summary["methods"][method].get(key), observed.get(key)) for key in CHECK_SUMMARY),
                 "Saved method summary differs from the actual recorded outcomes/accounting.")
    return {"manifest": manifest, "rows": compact, "firsts": firsts, "provenance": provenance,
            "error_events": sum(1 for _ in _rows(path / "errors.jsonl")) if (path / "errors.jsonl").exists() else 0,
            "legacy": legacy}


def combine(benchmark, model, runs, dataset):
    _require((len(runs) == 3 and all(run["manifest"]["replicates"] == 1 for run in runs))
             or (len(runs) == 1 and runs[0]["manifest"]["replicates"] == 3), "Three independent repetitions are required.")
    reference = runs[0]["manifest"]
    for run in runs[1:]:
        _require(all(run["manifest"].get(key) == reference.get(key) for key in SAME_MODEL), "Repetitions differ in model/data/interface/protocol/prices.")
    rows, initial_maps = [], defaultdict(dict)
    for run_index, run in enumerate(runs):
        for (identity, repetition), first in run["firsts"].items():
            initial_maps[run_index if len(runs) > 1 else repetition][identity] = first[0]
        rows.extend({**row, "replicate": run_index if len(runs) > 1 else row["replicate"]} for row in run["rows"])
    _require(len(initial_maps) == 3 and len({_digest(value) for value in initial_maps.values()}) == 3,
             "A repetition reused another run's entire initial-proposal cache.")
    categories = sorted({example["category"] for example in dataset.examples.values()})
    methods = {}
    for method in ARMS:
        groups = [[row for row in rows if row["method"] == method and row["replicate"] == repetition] for repetition in range(3)]
        per_run = [metrics(group, legacy=runs[0]["legacy"]) for group in groups]
        numeric = [key for key, value in per_run[0].items() if isinstance(value, (int, float)) or value is None]
        methods[method] = {"runs": per_run, "across_repetitions": {key: spread([item[key] for item in per_run]) for key in numeric},
                           "pooled": metrics([row for row in rows if row["method"] == method], legacy=runs[0]["legacy"]),
                           "categories": {category: {"accuracy": spread([metrics([row for row in group if row["category"] == category], legacy=runs[0]["legacy"])["accuracy"] for group in groups]),
                                                       "runs": [metrics([row for row in group if row["category"] == category], legacy=runs[0]["legacy"]) for group in groups]}
                                          for category in categories}}
    paired = paired_delta(rows, "jev_joint", seed=1729, samples=2000)
    accuracy_n = sum(example["category"] != "live_relevance" for example in dataset.examples.values())
    _require(paired["n_paired_ids"] == accuracy_n and paired["n_paired_records"] == accuracy_n * 3, "Paired interval lacks all prespecified repetitions/cases.")
    return {"benchmark": benchmark, "generator": model, "generator_display": DISPLAYS[model],
            "emitted_models": sorted({identity for run in runs for identity in run["provenance"]["emitted_models"]}),
            "accuracy_examples_per_repetition": accuracy_n, "timing_cost_examples_per_repetition": len(dataset.ids),
            "repetitions": 3, "methods": methods, "paired_dmfc_minus_direct": paired,
            "provenance": [run["provenance"] for run in runs], "api_error_events": sum(run["error_events"] for run in runs),
            "generator_interface": {key: reference.get(key) for key in ("llm_provider", "llm_api_style", "llm_gateway", "llm_endpoint", "reasoning_effort", "temperature", "max_completion_tokens", "prices_usd_per_million")}}


def build_report(root=ROOT):
    root = Path(root).resolve()
    datasets = load_datasets(root)
    rows = []
    for benchmark in ("bfcl", "apibank"):
        for model in MODELS:
            legacy = benchmark == "bfcl" and model != MODELS[2]
            paths = ([root / f"results/final/{model}_run{repetition}" for repetition in range(1, 4)] if legacy else
                     [root / ("results/additional/bfcl_claude-sonnet-5-5_compatible32" if benchmark == "bfcl" else
                              f"results/additional/apibank_candidate_{model}")])
            runs = [validate_run(path, model, datasets[benchmark], legacy=legacy, root=root) for path in paths]
            rows.append(combine(benchmark, model, runs, datasets[benchmark]))
    return {"format_version": "additional-paper-table-v1", "status": "complete", "source_sha256": CORE_SHA,
            "exporter_sha256": sha256(Path(__file__)), "settings": {"max_repairs": 3, "threshold": .5, "final_rule": "keep_first", "repetitions": 3},
            "aggregation": {"accuracy": "Mean and sample SD of three repetition accuracies",
                            "api_time": "Mean of three repetition means, entire actual trajectory; shared initial once per standalone arm",
                            "cost": "Mean of three per-repetition known-usage mean costs; unknown counts retained; missing all-known basis remains unavailable",
                            "strict_cost": "Also reports full-run mean cost only when every record has complete usage/rates",
                            "paired_ci": "Mean differences per example ID, then 2000 ID bootstrap samples with seed 1729",
                            "quantiles": "Per-repetition and pooled p50/p95 reported separately; neither is a quantile of repetition means",
                            "retry_errors": "Retry waits excluded from API time; failed unsaved attempts may add invoice spend beyond recorded standalone cost"},
            "benchmark_notes": {"bfcl": "Pinned single-turn subset; main accuracy excludes 16 live-relevance cases; timings/cost include all 3051; exploratory settings selected on these examples",
                                "apibank": "Exploratory adapted first-call exact match on 156 test cases; common native-candidate instruction amended after format-pilot test outcomes were inspected, then checked on four common development cases; r/threshold/fallback unchanged. Fixed 53-tool catalog; not native execution accuracy and no no-call negatives. Earlier original/user-turn format pilots excluded and retained."},
            "datasets": {name: {"data_path": str(dataset.path), "dataset_sha256": dataset.digest, "prepared_manifest_sha256": dataset.manifest_sha,
                                "counts": dataset.prepared_manifest.get("counts"), "input_adapter": dataset.prepared_manifest.get("input_adapter")}
                         for name, dataset in datasets.items()}, "rows": rows}


def latex_table(report):
    _require(report.get("status") == "complete" and len(report.get("rows", [])) == 6, "A complete six-row report is required.")
    lines = [MARKER, r"\begin{table*}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{5pt}",
             r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"Benchmark & Generator & Direct (\%) & \textbf{DMGFC} (\%) & API s (D / M) & USD/1k (D / M)\\", r"\midrule"]
    for index, row in enumerate(report["rows"]):
        values = [row["methods"][method]["across_repetitions"] for method in ARMS]
        accuracy = []
        for current, other in ((values[0], values[1]), (values[1], values[0])):
            value = current["accuracy"]
            mean = f"{100 * value['mean']:.1f}"
            if value["mean"] > other["accuracy"]["mean"]:
                mean = rf"\mathbf{{{mean}}}"
            accuracy.append(rf"${mean}\pm{100 * value['sd']:.1f}$")
        def pair(key, scale, digits):
            return " / ".join(r"\textemdash" if value[key]["mean"] is None else f"{scale * value[key]['mean']:.{digits}f}" for value in values)
        benchmark = "BFCL" if row["benchmark"] == "bfcl" else r"API-Bank$^{\dagger}$"
        if index == 3:
            lines.append(r"\midrule")
        lines.append(f"{benchmark} & {row['generator_display']} & " + " & ".join(accuracy)
                     + " & " + pair("mean_latency_s", 1, 2) + " & " + pair("mean_known_cost_usd", 1000, 3) + r"\\")
    unknown = sum(row["methods"][method]["pooled"]["unknown_cost_n"] for row in report["rows"] for method in ARMS)
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Direct/DMGFC accuracy: mean $\pm$ sample s.d. of three runs, $r=3$; bold compares methods per row. "
              r"API seconds and USD/1k are Direct/DMGFC pairs. BFCL: 3,035 accuracy cases, 3,051 time/cost cases; "
              r"$^{\dagger}$API-Bank: 156 cases, exploratory adapted exact match. Time includes generation/verification only; "
              rf"cost omits {unknown} records with unavailable usage/prices. \textemdash\ means unavailable.}}",
              r"\label{tab:categories}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def update_paper(path, table):
    path = Path(path)
    source = path.read_text()
    if "% BEGIN AUTO CATEGORIES" not in source and r"\input{sec/experiment_setting_and_results}" in source:
        path = path.parent / "sec/experiment_setting_and_results.tex"
        source = path.read_text()
    pattern = r"(% BEGIN AUTO CATEGORIES\s*\n)(.*?)(\n% END AUTO CATEGORIES)"
    _require(len(re.findall(pattern, source, re.DOTALL)) == 1, "Exactly one CATEGORIES auto block is required.")
    updated = re.sub(pattern, lambda match: match.group(1) + table.rstrip() + match.group(3), source, flags=re.DOTALL)
    path.write_text(updated)
    return path


def _output_paths_safe(paths, root, paper):
    root = Path(root).resolve()
    protected_dirs = [root / "data", root / "results/final", root / "paper"]
    additional = root / "results/additional"
    if additional.exists():
        protected_dirs.extend(path for path in additional.iterdir() if path.is_dir() and (path / "manifest.json").exists())
    resolved = [Path(path).resolve() for path in paths]
    _require(len(set(resolved)) == len(resolved), "JSON and LaTeX outputs must be different files.")
    for path in resolved:
        _require(not any(path == directory or directory in path.parents for directory in protected_dirs)
                 and (paper is None or path != Path(paper).resolve()), "Report outputs must not overwrite source data, runs, or manuscript files.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", type=Path, help="Default: results/additional/combined-main.json")
    parser.add_argument("--tex", type=Path, help="Default: results/additional/combined-main.tex")
    parser.add_argument("--paper", type=Path, help="Optional: replace only this manuscript's CATEGORIES auto block")
    args = parser.parse_args(argv)
    json_path = args.json or args.root / "results/additional/combined-main.json"
    tex_path = args.tex or args.root / "results/additional/combined-main.tex"
    _output_paths_safe((json_path, tex_path), args.root, args.paper)
    report = build_report(args.root)
    table = latex_table(report)
    for path in (json_path, tex_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tex_path.write_text(table)
    if args.paper:
        update_paper(args.paper, table)
    print(json.dumps({"status": "complete", "rows": len(report["rows"]), "json": str(json_path), "tex": str(tex_path), "model_requests": 0}, indent=2))
    return report


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print("Combined table unavailable: " + str(exc), file=sys.stderr)
        raise SystemExit(1) from None
