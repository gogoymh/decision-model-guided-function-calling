"""Additional evaluation stays paired, oracle-free, resumable, and offline until explicitly run."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("additional", ROOT / "scripts/run_additional_evaluation.py")
additional = importlib.util.module_from_spec(spec)
spec.loader.exec_module(additional)


def example(identity="a", split="test"):
    return {"id": identity, "category": "apibank_firstcall", "split": split, "family": identity,
            "source_file": "ORACLE_FILE", "dataset_revision": additional.prepare_apibank.REVISION,
            "scorable": True, "messages": [{"role": "user", "content": "Book a room."}],
            "tools": [{"type": "function", "function": {"name": "Book", "description": "Book a room.",
                       "parameters": {"type": "object", "properties": {"name": {"type": "string"}},
                                      "required": ["name"], "additionalProperties": False}}}],
            "ground_truth": [{"name": "Book", "arguments": {"name": "ORACLE_GOLD"}}],
            "raw_ground_truth": {"api_name": "Book", "param_dict": {"name": "ORACLE_GOLD"}}}


def proposal(name="ORACLE_GOLD"):
    return {"calls": [{"name": "Book", "arguments": {"name": name}}], "content": None,
            "latency_s": 2., "retry_wait_s": .25, "usage": {"prompt_tokens": 100, "completion_tokens": 10}}


class Generator:
    def __init__(self, events, initial="wrong", repairs=None):
        self.events, self.initial = events, initial
        self.repairs = iter(repairs or ["ORACLE_GOLD"])
        self.calls = []
        self.model, self.provider, self.api_style = "new-generator", "anthropic", "anthropic_messages"
        self.gateway, self.base_url, self.endpoint = "apim", "https://example.invalid/v1", "https://example.invalid/v1/messages"
        self.auth_header, self.timeout, self.max_retries = "api-key", 120, 3
        self.reasoning_effort, self.temperature, self.max_completion_tokens = None, None, 4096

    def generate(self, public, history):
        assert set(public) == {"messages", "tools"}
        assert "ORACLE" not in json.dumps(public)
        assert all("score" not in item for item in history)
        self.calls.append((deepcopy(public), deepcopy(history)))
        self.events.append("generate")
        return proposal(next(self.repairs) if history else self.initial)

    def close(self):
        pass


class Verifier:
    def __init__(self, events, fail=False, accepts=None):
        self.events, self.fail = events, fail
        self.accepts = iter(accepts or [False, True])
        self.model, self.gateway, self.base_url = "jev-1.13.0", "direct", "https://example.invalid"
        self.timeout, self.max_retries = 60, 3

    def verify(self, public, calls):
        assert set(public) == {"messages", "tools"}
        self.events.append("verify")
        if self.fail:
            error = RuntimeError("SECRET_HTTP_BODY_AND_KEY")
            error.status_code = 503
            raise error
        passed = next(self.accepts)
        return {"function_pass": True, "arguments_pass": passed, "arguments_skipped": False,
                "function_probability": .99, "arguments_probability": .9 if passed else .1,
                "accepted": passed, "latency_s": .5, "retry_wait_s": .1, "api_calls": 1,
                "usage": {"input_tokens": 20},
                "feedback": {"checks": {"arguments": {"pass": passed, "probability_yes": .9 if passed else .1}},
                             "instruction": "Check arguments."}}

    def close(self):
        pass


def args(tmp_path, **override):
    values = dict(benchmark="apibank", model="new-generator", api_style="anthropic_messages",
                  gateway="apim", jev_gateway="direct", jev_model="jev-1.13.0", base_url=None,
                  jev_base_url=None, data=tmp_path / "data", checker=tmp_path / "checker.py", split="test",
                  split_seed=1729, replicates=1, workers=1, limit=None, output=tmp_path / "run", resume=False,
                  run=True, threshold=.5, reasoning_effort="none", temperature=None,
                  max_completion_tokens=4096, timeout=120., jev_timeout=60., max_retries=3)
    values.update({name + "_price": None for name in additional.PRICE_NAMES})
    values.update(override)
    return SimpleNamespace(**values)


def config(arguments, examples, generator, verifier):
    return additional.settings_for(arguments, examples,
                                   {"dataset_sha256": "fixed-data", "dataset_preparer_sha256": "prepared"},
                                   generator, verifier)


def test_pairing_scoring_after_requests_cost_and_initial_timing(tmp_path, monkeypatch):
    events = []
    generator, verifier = Generator(events), Verifier(events)
    arguments = args(tmp_path, llm_input_price=2., llm_output_price=10., jev_input_price=.042)
    cases = [example()]
    settings = config(arguments, cases, generator, verifier)

    def score(case, calls):
        events.append("score")
        return additional.prepare_apibank.score_example(case, calls)

    summary = additional.execute(arguments, cases, score, settings, generator, verifier)
    rows = additional.read_jsonl(arguments.output / "records.jsonl")
    direct, dmfc = rows
    assert events == ["generate", "verify", "generate", "verify", "score", "score", "score"]
    assert len(generator.calls) == 2 and generator.calls[0][1] == []
    assert direct["trace"][0]["proposal"] == dmfc["trace"][0]["proposal"]
    assert direct["logical_latency_s"] == 2 and dmfc["logical_latency_s"] == 5
    assert direct["score"]["correct"] is False and dmfc["score"]["correct"] is True
    assert direct["accounting"]["estimated_usd"] == pytest.approx(.0003)
    assert dmfc["accounting"]["estimated_usd"] == pytest.approx(.00060168)
    initial = additional.read_jsonl(arguments.output / "initial.jsonl")[0]
    assert direct["wall_time_s"] >= initial["initial_wall_time_s"]
    assert dmfc["wall_time_s"] >= initial["initial_wall_time_s"]
    assert all(row["dispatch_mode"] == "simulated" and row["execution_result"] is None for row in rows)
    assert summary["status"] == "complete" and summary["paired_dmfc_minus_direct"]["delta"] == 1
    # The repair changes Book's argument only; it must not count as a function-selection change.
    assert summary["methods"]["jev_joint"]["function_change_rate"] == 0
    assert summary["methods"]["jev_joint"]["mean_retry_wait_s"] == pytest.approx(.7)


def test_failed_verifier_stops_sanitized_and_resume_reuses_initial(tmp_path):
    events = []
    arguments = args(tmp_path)
    generator, verifier = Generator(events), Verifier(events, fail=True)
    cases = [example()]
    settings = config(arguments, cases, generator, verifier)
    with pytest.raises(RuntimeError, match="sanitized error"):
        additional.execute(arguments, cases, additional.prepare_apibank.score_example, settings, generator, verifier)
    assert len(generator.calls) == 1
    assert "SECRET" not in (arguments.output / "errors.jsonl").read_text()
    arguments.resume = True
    events.clear()
    generator, verifier = Generator(events), Verifier(events)
    summary = additional.execute(arguments, cases, additional.prepare_apibank.score_example, settings, generator, verifier)
    assert len(generator.calls) == 1 and generator.calls[0][1]  # only repair generation; cached initial reused
    assert summary["status"] == "complete"
    events.clear()
    additional.execute(arguments, cases, additional.prepare_apibank.score_example, settings, generator, verifier)
    assert events == []  # complete resume never generates or verifies again
    changed = {**settings, "llm_model": "other-generator"}
    with pytest.raises(ValueError, match="fingerprint"):
        additional.execute(arguments, cases, additional.prepare_apibank.score_example, changed, generator, verifier)


def test_keep_first_and_three_repetitions_id_clustered(tmp_path):
    events = []
    arguments = args(tmp_path, replicates=3)
    generator = Generator(events, initial="ORACLE_GOLD", repairs=["wrong"] * 9)
    verifier = Verifier(events, accepts=[False] * 12)
    summary = additional.execute(arguments, [example()], additional.prepare_apibank.score_example,
                                 config(arguments, [example()], generator, verifier), generator, verifier)
    rows = additional.read_jsonl(arguments.output / "records.jsonl")
    joint = [row for row in rows if row["method"] == "jev_joint"]
    assert len(generator.calls) == 12 and len(joint) == 3
    assert all(row["final_attempt"] == 0 and row["score"]["correct"] for row in joint)
    assert all(row["trace"][-1]["score"]["correct"] is False and row["logical_latency_s"] == 10 for row in joint)
    assert summary["repetition_report"]["jev_joint"]["accuracy_mean"] == 1
    assert summary["repetition_report"]["jev_joint"]["accuracy_sample_sd"] == 0
    assert summary["paired_dmfc_minus_direct"]["n_paired_ids"] == 1
    assert summary["paired_dmfc_minus_direct"]["n_paired_records"] == 3
    assert summary["methods"]["jev_joint"]["known_usage_usd_per_1k"] is None


def write_prepared(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    path = folder / "examples.jsonl"
    cases = [example("a", "test"), example("b", "dev")]
    path.write_text("".join(json.dumps(case) + "\n" for case in cases))
    manifest = {"revision": additional.prepare_apibank.REVISION, "format_version": "apibank-firstcall-v1",
                "preparer_sha256": additional.sha256(Path(additional.prepare_apibank.__file__)),
                "files_sha256": {"examples.jsonl": additional.sha256(path)}, "split": {"unit": "family"}}
    additional._write_json(folder / "manifest.json", manifest)
    return folder


def test_prepared_data_hash_split_and_default_offline_preflight(tmp_path, monkeypatch):
    folder = write_prepared(tmp_path)
    events = []
    generator, verifier = Generator(events), Verifier(events)
    monkeypatch.setattr(additional, "make_clients", lambda *args, **kwargs: (generator, verifier))
    result = additional.main(["--benchmark", "apibank", "--model", "new-generator",
                              "--api-style", "anthropic_messages", "--data", str(folder),
                              "--output", str(tmp_path / "offline")])
    assert result["network_requests"] == 0 and result["selected_examples"] == 1
    assert result["run_enabled"] is False and events == [] and not (tmp_path / "offline").exists()
    arguments = args(tmp_path)
    selected, _, _ = additional.load_data(arguments)
    assert [case["id"] for case in selected] == ["a"]
    (folder / "examples.jsonl").write_text("edited data")
    with pytest.raises(ValueError, match="checksum"):
        additional.load_data(arguments)


def test_unknown_prices_do_not_inherit_other_model_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_INPUT_PRICE", ".05")
    monkeypatch.setenv("LLM_OUTPUT_PRICE", ".4")
    for name in additional.PRICE_NAMES:
        monkeypatch.delenv("ADDITIONAL_" + name.upper() + "_PRICE", raising=False)
    assert set(additional.explicit_prices(args(tmp_path)).values()) == {None}
    monkeypatch.setenv("ADDITIONAL_LLM_INPUT_PRICE", "2")
    assert additional.explicit_prices(args(tmp_path))["llm_input"] == 2
    assert additional.explicit_prices(args(tmp_path, llm_input_price=3))["llm_input"] == 3


def test_existing_paper_output_rejected_before_any_operation(tmp_path):
    arguments = args(tmp_path)
    arguments.output.mkdir()
    additional._write_json(arguments.output / "manifest.json", {"config_id": "paper-run", "methods": ["direct"]})
    events = []
    generator, verifier = Generator(events), Verifier(events)
    with pytest.raises(ValueError, match="fingerprint"):
        additional.execute(arguments, [example()], additional.prepare_apibank.score_example,
                           config(arguments, [example()], generator, verifier), generator, verifier)
    assert events == []


def test_bfcl_empty_tool_irrelevance_is_preserved_in_full_preflight(tmp_path):
    arguments = args(tmp_path, benchmark="bfcl", data=ROOT / "data/bfcl", split="all",
                     checker=ROOT / "data/bfcl/upstream/ast_checker.py")
    if not arguments.data.exists():
        pytest.skip("Prepared canonical BFCL data unavailable; tests do not download it.")
    cases, _, _ = additional.load_data(arguments)
    assert len(cases) == 3051
    empty_tools = [case for case in cases if not case["tools"]]
    assert len(empty_tools) == 4 and all(case["category"] == "live_irrelevance" for case in empty_tools)
