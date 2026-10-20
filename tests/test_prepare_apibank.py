"""Offline extraction preserves first-call boundaries, independent tools, and audited source anomalies."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_apibank", ROOT / "scripts/prepare_apibank.py")
prepare_apibank = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_apibank)


def write_dialogue(root, filename, turns):
    path = root / prepare_apibank.DIALOGUE_PATH / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(turn) + "\n" for turn in turns))


@pytest.fixture
def toy_upstream(tmp_path):
    folder = tmp_path / prepare_apibank.API_PATH
    folder.mkdir(parents=True)
    # This would fail if imported. The preparer must read literal AST fields only.
    (folder / "tools.py").write_text('''
raise RuntimeError("UPSTREAM MUST NOT EXECUTE")
class API: pass
class Book(API):
    description = "Book a room."
    input_parameters = {"name": {"type": "str", "description": "Name"},
                        "count": {"type": "int", "description": "Count"}}
    output_parameters = {"order": {"type": "str"}}
    def call(self, name, count=1): pass
class Reset(API):
    description = "First status requests email; second status requests code."
    input_parameters = {"status": {"type": "str"}, "email": {"type": "str"}, "code": {"type": "int"}}
    output_parameters = {}
    def call(self, status, **kwargs): pass
class Cancel(API):
    description = "Cancel by name."
    input_parameters = {"name": {"type": "str"}}
    output_parameters = {}
    def call(self, name): pass
''')
    return tmp_path


def test_ast_requiredness_keeps_optional_and_conditional_arguments(toy_upstream):
    catalog = prepare_apibank.read_catalog(toy_upstream)
    assert catalog["Book"]["required"] == ["name"]
    assert catalog["Reset"]["required"] == ["status"]
    assert "email" in catalog["Reset"]["tool"]["function"]["parameters"]["properties"]
    assert "second status" in catalog["Reset"]["tool"]["function"]["description"]


def test_first_call_never_uses_future_labels_observations_or_target_tool_selection(toy_upstream):
    prefix = [{"role": "User", "text": "Book a room named Lake"}, {"role": "AI", "text": "For tomorrow?"}]
    target = {"role": "API", "api_name": "Book", "param_dict": {"name": "SECRET_LABEL", "count": "2"},
              "result": {"output": "SECRET_CURRENT_OUTPUT"}}
    future = {"role": "API", "api_name": "Reset", "param_dict": {"status": "SECRET_FUTURE"}}
    write_dialogue(toy_upstream, "Book-family-1.jsonl", [*prefix, target, future])
    write_dialogue(toy_upstream, "Other-family-1.jsonl", [*prefix, {**target, "api_name": "Cancel",
                                                                "param_dict": {"name": "different"}}])
    catalog = prepare_apibank.read_catalog(toy_upstream)
    rows, exclusions = prepare_apibank.build_dataset(toy_upstream, catalog)
    assert not exclusions
    assert rows[0]["tools"] == rows[1]["tools"] == [entry["tool"] for entry in catalog.values()]
    public = prepare_apibank.model_input(rows[0])
    assert set(public) == {"messages", "tools"}
    assert public["messages"] == [{"role": "user", "content": prefix[0]["text"]},
                                  {"role": "assistant", "content": prefix[1]["text"]}]
    assert "SECRET" not in json.dumps(public)
    assert rows[0]["ground_truth"] == [{"name": "Book", "arguments": {"name": "SECRET_LABEL", "count": 2}}]
    assert "SECRET_CURRENT_OUTPUT" not in json.dumps(rows[0])
    public["tools"].clear()
    assert len(rows[0]["tools"]) == 3


def test_schema_issue_retained_without_renaming_and_no_api_file_is_not_negative(toy_upstream):
    write_dialogue(toy_upstream, "Cancel-family-1.jsonl", [
        {"role": "User", "text": "Cancel it."},
        {"role": "API", "api_name": "Cancel", "param_dict": {"device_id": "lamp"}},
    ])
    write_dialogue(toy_upstream, "NoAPI-1.jsonl", [{"role": "User", "text": "No API turn in this source."}])
    rows, exclusions = prepare_apibank.build_dataset(toy_upstream, prepare_apibank.read_catalog(toy_upstream))
    assert len(rows) == 1 and not rows[0]["scorable"] and rows[0]["ground_truth"] is None
    assert rows[0]["raw_ground_truth"]["param_dict"] == {"device_id": "lamp"}
    assert rows[0]["issues"] == [{"kind": "unknown_arguments", "names": ["device_id"]},
                                  {"kind": "missing_required", "names": ["name"]}]
    assert {row["reason"] for row in exclusions} == {"no_api_target", "schema_incompatible_target"}
    with pytest.raises(ValueError, match="must not enter scored runs"):
        prepare_apibank.score_example(rows[0], [{"name": "Cancel", "arguments": {"name": "lamp"}}])


def test_exact_score_is_post_generation_and_conditional_on_function(toy_upstream):
    write_dialogue(toy_upstream, "Book-family-1.jsonl", [
        {"role": "User", "text": "Book Lake for two."},
        {"role": "API", "api_name": "Book", "param_dict": {"name": "Lake", "count": "2"}},
    ])
    rows, _ = prepare_apibank.build_dataset(toy_upstream, prepare_apibank.read_catalog(toy_upstream))
    example = rows[0]
    correct = prepare_apibank.score_example(example, [{"name": "Book", "arguments": {"name": "Lake", "count": 2}}])
    assert correct["correct"] and correct["function_correct"] and correct["arguments_correct"]
    assert correct["metric"] == "apibank_firstcall_exact"
    coerced = prepare_apibank.score_example(example, [{"name": "Book", "arguments": {"name": "Lake", "count": "2"}}])
    assert coerced["correct"]
    wrong_args = prepare_apibank.score_example(example, [{"name": "Book", "arguments": {"name": "wrong", "count": 2}}])
    assert not wrong_args["correct"] and wrong_args["function_correct"] and not wrong_args["arguments_correct"]
    assert "Lake" not in json.dumps(wrong_args["errors"])
    wrong_function = prepare_apibank.score_example(example, [{"name": "Cancel", "arguments": {"name": "Lake"}}])
    assert not wrong_function["correct"] and not wrong_function["function_correct"]
    assert wrong_function["arguments_correct"] is None
    assert not prepare_apibank.score_example(example, [])["correct"]
    assert not prepare_apibank.score_example(example, None)["correct"]
    assert not prepare_apibank.score_example(example, [{"name": "Book", "arguments": {"name": "Lake", "x": 2}}])["correct"]


@pytest.mark.parametrize("raw,declared,expected", [
    ("007", "str", "007"), ("2", "int", 2), ("-3", "int", -3), ("True", "bool", True),
    ("False", "bool", False), ("['Alice', 'Bob']", "list(str)", ["Alice", "Bob"]),
    ("[{'name': 'pulse', 'value': '80'}]", "list", [{"name": "pulse", "value": "80"}]),
])
def test_declared_type_coercion(raw, declared, expected):
    assert prepare_apibank.coerce_argument(raw, declared) == expected


@pytest.mark.parametrize("raw,declared", [
    (True, "int"), ("2.5", "int"), ("true", "bool"), ("['Alice', 3]", "list(str)"),
    ("__import__('os').system('false')", "list"), ("[{1: 'non-string-key'}]", "list"),
])
def test_coercion_rejects_ambiguous_or_nonliteral_values(raw, declared):
    with pytest.raises((ValueError, SyntaxError)):
        prepare_apibank.coerce_argument(raw, declared)


def test_family_split_is_deterministic_and_keeps_variants_together():
    family, split = prepare_apibank.family_split("BookHotel-level-1-1.jsonl")
    assert family == "BookHotel-level-1"
    assert prepare_apibank.family_split("BookHotel-level-1-5.jsonl") == (family, split)
    assert split in {"dev", "test"}


def test_source_pin_and_checksum_reject_changes_before_output(toy_upstream, monkeypatch):
    monkeypatch.setattr(prepare_apibank.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(stdout="wrong-revision\n"))
    with pytest.raises(ValueError, match="audited revision"):
        prepare_apibank.prepare(toy_upstream, toy_upstream / "output")
    assert not (toy_upstream / "output").exists()
    monkeypatch.setattr(prepare_apibank.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(stdout=prepare_apibank.REVISION + "\n"))
    with pytest.raises(ValueError, match="dialogue checksum/count"):
        prepare_apibank.prepare(toy_upstream, toy_upstream / "output")
    assert not (toy_upstream / "output").exists()


def test_raw_tree_checksum_is_order_independent_but_sensitive_to_labels():
    before = {"a.jsonl": "hash-a", "b.jsonl": "hash-b"}
    reordered = {"b.jsonl": "hash-b", "a.jsonl": "hash-a"}
    assert prepare_apibank._tree_hash(before) == prepare_apibank._tree_hash(reordered)
    assert prepare_apibank._tree_hash(before) != prepare_apibank._tree_hash({**before, "a.jsonl": "edited-gold"})


def test_pinned_official_firstcall_counts_and_family_split_if_available():
    upstream = ROOT.parent / "Documents/Codex/2026-10-19/new-chat/work/api-bank-official"
    if not upstream.exists():
        pytest.skip("Pinned API-Bank clone unavailable; tests never download or call models.")
    prepare_apibank.verify_sources(upstream)
    catalog = prepare_apibank.read_catalog(upstream)
    rows, exclusions = prepare_apibank.build_dataset(upstream, catalog)
    eligible = [row for row in rows if row["scorable"]]
    dev = [row for row in eligible if row["split"] == "dev"]
    test = [row for row in eligible if row["split"] == "test"]
    assert (len(catalog), len(rows), len(eligible), len(dev), len(test)) == (53, 212, 204, 48, 156)
    assert {row["family"] for row in dev}.isdisjoint(row["family"] for row in test)
    assert {row["source_file"] for row in exclusions if row["reason"] == "no_api_target"} \
        == prepare_apibank.NO_API_FILES
    assert {row["source_file"] for row in exclusions if row["reason"] == "schema_incompatible_target"} \
        == prepare_apibank.SCHEMA_EXCLUSIONS
    assert all(row["tools"] == eligible[0]["tools"] for row in eligible)
    assert all(message["role"] in {"user", "assistant"} for row in eligible for message in row["messages"])
    assert all(prepare_apibank.score_example(row, row["ground_truth"])["correct"] for row in eligible)
    # GetToday has {} call parameters even when its recorded result.input is null.
    today = [row for row in eligible if row["raw_ground_truth"]["api_name"] == "GetToday"]
    assert len(today) == 6
    assert all(row["ground_truth"] == [{"name": "GetToday", "arguments": {}}] for row in today)


def test_local_official_preparation_is_idempotent_and_rejects_changed_output(tmp_path):
    upstream = ROOT.parent / "Documents/Codex/2026-10-19/new-chat/work/api-bank-official"
    if not upstream.exists():
        pytest.skip("Pinned API-Bank clone unavailable; tests never download or call models.")
    output = tmp_path / "prepared"
    first = prepare_apibank.prepare(upstream, output)
    assert prepare_apibank.prepare(upstream, output) == first
    assert (output / "upstream/API-Bank-LICENSE").read_bytes() == (upstream / "api-bank/LICENSE").read_bytes()
    assert len((output / "all_firstcalls.jsonl").read_text().splitlines()) == 212
    assert len((output / "examples.jsonl").read_text().splitlines()) == 204
    assert len((output / "exclusions.jsonl").read_text().splitlines()) == 10
    raw = output / "raw/SearchEngine-level-1-1.jsonl"
    raw.write_text("modified raw label\n")
    with pytest.raises(ValueError, match="Existing preparation differs"):
        prepare_apibank.prepare(upstream, output)
    assert raw.read_text() == "modified raw label\n"
