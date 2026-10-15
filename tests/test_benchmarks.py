"""Offline data isolation, split, schema, and official-checker regression tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from jev_fc.benchmarks import (
    BFCL_REVISION,
    KNOWN_ANNOTATION_ISSUES,
    _fetch_verified,
    json_schema,
    load_examples,
    normalize_example,
    prepare_bfcl,
    public_example,
    stratified_split,
)
from jev_fc.scoring import canonical_calls, load_official_checker, score_example


def toy_example(category="simple_python"):
    question = {
        "id": f"{category}_toy",
        "question": [[{"role": "user", "content": "Look up age 7 in San Diego."}]],
        "function": [{
            "name": "people.lookup",
            "description": "Look up people.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "age": {"type": "integer"},
                    "city": {"type": "string"},
                    "unit": {"type": "string", "optional": True, "default": "metric"},
                },
                "required": ["age", "city"],
            },
        }],
    }
    answer = {"id": question["id"], "ground_truth": [{"people.lookup": {"age": [7], "city": ["San Diego"], "unit": ["", "metric"]}}]}
    return normalize_example(question, category, answer)


@pytest.fixture
def checker_path():
    path = Path(__file__).resolve().parents[1] / "data/bfcl/upstream/ast_checker.py"
    if not path.exists():
        pytest.skip("Run python -m jev_fc.benchmarks once to fetch the pinned official evaluator; tests never use the network")
    return path


def test_schema_conversion_handles_property_named_type_and_nested_types():
    schema = {
        "type": "dict",
        "properties": {
            "type": {"type": "string", "enum": ["round"]},
            "points": {"type": "tuple", "items": {"type": "float"}},
            "options": {"type": "dict", "properties": {"x": {"type": "any"}}},
        },
        "required": ["type"],
    }
    original = copy.deepcopy(schema)
    converted = json_schema(schema)
    assert converted["type"] == "object"
    assert converted["properties"]["type"] == {"type": "string", "enum": ["round"]}
    assert converted["properties"]["points"] == {"type": "array", "items": {"type": "number"}}
    assert converted["properties"]["options"]["properties"]["x"]["type"] == "string"
    assert schema == original


def test_public_context_has_no_answer_or_category_label_in_messages():
    row = toy_example()
    public = public_example(row)
    assert set(public) == {"id", "category", "messages", "tools"}
    assert "ground_truth" not in json.dumps(public)
    assert "bfcl_functions" not in public and "name_map" not in public
    public["tools"][0]["function"]["description"] = "changed"
    assert row["tools"][0]["function"]["description"] == "Look up people."


def test_aliases_are_reversible_and_collision_safe():
    question = {
        "id": "multiple_alias", "question": [[{"role": "user", "content": "test"}]],
        "function": [{"name": name, "description": "", "parameters": {"type": "dict", "properties": {}, "required": []}} for name in ["a.b", "a_b", "z" * 100]],
    }
    row = normalize_example(question, "multiple", {"id": "multiple_alias", "ground_truth": [{"a.b": {}}]})
    assert len(row["name_map"]) == 3
    assert max(map(len, row["name_map"])) <= 64
    assert set(row["name_map"].values()) == {"a.b", "a_b", "z" * 100}
    calls = [{"name": alias, "arguments": {}} for alias in row["name_map"]]
    assert [next(iter(call)) for call in canonical_calls(row, calls)] == ["a.b", "a_b", "z" * 100]


def test_ground_truth_alignment_cannot_silently_shift():
    row = toy_example()
    question = {"id": row["id"], "question": [row["messages"]], "function": row["bfcl_functions"]}
    with pytest.raises(ValueError, match="mismatched"):
        normalize_example(question, "simple_python", {"id": "wrong", "ground_truth": []})


def test_split_membership_is_order_independent_and_disjoint():
    rows = [{"id": f"{cat}_{i}", "category": cat} for cat in ["a", "b"] for i in range(25)]
    dev, test = stratified_split(rows, dev_fraction=0.1, seed=42)
    reversed_dev, reversed_test = stratified_split(reversed(rows), dev_fraction=0.1, seed=42)
    dev_ids, test_ids = {x["id"] for x in dev}, {x["id"] for x in test}
    assert len(dev) == 4 and len(test) == 46  # 2.5 per stratum rounds down.
    assert dev_ids.isdisjoint(test_ids)
    assert dev_ids | test_ids == {x["id"] for x in rows}
    assert dev_ids == {x["id"] for x in reversed_dev}
    assert test_ids == {x["id"] for x in reversed_test}
    assert "split" not in rows[0]
    assert {x["id"] for x in stratified_split(rows, seed=43)[0]} != dev_ids


def test_modified_cached_file_is_rejected_without_network(tmp_path):
    cached = tmp_path / "data.json"
    cached.write_text("tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _fetch_verified("https://invalid.example/not-used", cached, "0" * 64)


def test_modified_normalized_data_is_rejected_against_manifest(tmp_path):
    path = tmp_path / "examples.jsonl"
    path.write_text(json.dumps(toy_example()) + "\n")
    (tmp_path / "manifest.json").write_text(json.dumps({"normalized_sha256": {"examples.jsonl": "0" * 64}}))
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_examples(path)


def test_unreviewed_revision_rejected_before_network(tmp_path):
    with pytest.raises(ValueError, match="audited revision"):
        prepare_bfcl(tmp_path, revision="main")


def test_modified_checker_is_rejected_before_execution(tmp_path):
    path = tmp_path / "checker.py"
    path.write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="unrecognized"):
        load_official_checker(path)


def test_structured_parser_never_evaluates_model_text():
    row = toy_example()
    with pytest.raises(ValueError):
        canonical_calls(row, [{"name": "people_lookup", "arguments": "__import__('os').system('false')"}])
    with pytest.raises(ValueError, match="offered aliases"):
        canonical_calls(row, [{"name": "people.lookup", "arguments": {}}])
    with pytest.raises(ValueError, match="JSON object"):
        canonical_calls(row, [{"name": "people_lookup", "arguments": "[]"}])
    assert canonical_calls(row, [{"function": {"name": "people_lookup", "arguments": '{"age":7}'}}]) == [{"people.lookup": {"age": 7}}]


def test_official_checker_uses_aliases_optional_omission_and_string_normalization(checker_path):
    row = toy_example()
    result = score_example(row, [{"name": "people_lookup", "arguments": {"age": 7, "city": "san-diego"}}], checker_path)
    assert result["correct"] is True
    assert result["function_correct"] is True and result["arguments_correct"] is True
    assert result["metric"] == "bfcl_python_ast" and result["checker_revision"] == BFCL_REVISION


@pytest.mark.parametrize("arguments", [
    {"age": True, "city": "San Diego"},
    {"age": "7", "city": "San Diego"},
    {"age": 7, "city": "Seattle"},
    {"age": 7, "city": "San Diego", "extra": 1},
    {"city": "San Diego"},
])
def test_official_checker_rejects_wrong_values_types_missing_and_extra_arguments(checker_path, arguments):
    result = score_example(toy_example(), [{"name": "people_lookup", "arguments": arguments}], checker_path)
    assert result["correct"] is False and result["function_correct"] is True
    assert result["arguments_correct"] is False


def test_function_diagnostic_is_separate_from_argument_accuracy(checker_path):
    row = toy_example("multiple")
    row["name_map"]["wrong"] = "wrong"
    row["bfcl_functions"].append({"name": "wrong", "parameters": {"type": "dict", "properties": {}, "required": []}})
    result = score_example(row, [{"name": "wrong", "arguments": {}}], checker_path)
    assert result["correct"] is False and result["function_correct"] is False
    assert result["arguments_correct"] is None


def test_presence_tasks_have_no_fabricated_argument_labels():
    for category in ["irrelevance", "live_irrelevance"]:
        row = toy_example(category)
        assert row["ground_truth"] == []
        result = score_example(row, [])
        assert result["correct"] is True and result["arguments_correct"] is None
        assert result["metric"] == "bfcl_call_presence"
        assert score_example(row, [{"name": "people_lookup", "arguments": {}}])["correct"] is False
    row = toy_example("live_relevance")
    assert row["ground_truth"] is None
    result = score_example(row, [{"name": "people_lookup", "arguments": {}}])
    assert result["correct"] is True and result["function_correct"] is None and result["arguments_correct"] is None
    assert score_example(row, [])["correct"] is False
    # Official live_relevance checks call presence only, even for unknown names.
    assert score_example(row, [{"name": "unknown", "arguments": {}}])["correct"] is True


def test_malformed_calls_remain_in_function_selection_denominator():
    row = toy_example()
    result = score_example(row, [{"name": "people_lookup", "arguments": None}])
    assert result["correct"] is False and result["function_correct"] is False
    assert result["arguments_correct"] is None


def test_prepared_manifest_counts_and_split_if_available():
    directory = Path(__file__).resolve().parents[1] / "data/bfcl"
    if not (directory / "manifest.json").exists():
        pytest.skip("Prepared BFCL data not present; this test never downloads")
    manifest = json.loads((directory / "manifest.json").read_text())
    if set(manifest["counts"]) != {"simple_python", "multiple", "irrelevance", "live_simple", "live_multiple", "live_irrelevance", "live_relevance"}:
        pytest.skip("A category subset was prepared")
    rows = load_examples(directory)
    dev, test = load_examples(directory / "dev.jsonl"), load_examples(directory / "test.jsonl")
    assert len(rows) == 3051
    assert len(dev) == 303 and len(test) == 2748
    assert {x["id"] for x in dev}.isdisjoint(x["id"] for x in test)
    assert len(dev) + len(test) == len(rows)


def test_all_prepared_gold_candidates_have_only_prespecified_source_anomalies(checker_path):
    from jsonschema import Draft202012Validator

    directory = checker_path.parent.parent
    if not (directory / "examples.jsonl").exists():
        pytest.skip("Prepared BFCL data not present")
    rows = load_examples(directory)
    observed_failures = set()
    present_ids = {row["id"] for row in rows}

    def dictionary_candidate(alternatives):
        return {key: next((x for x in values if x != ""), []) for key, values in alternatives.items() if "" not in values}

    for row in rows:
        for tool in row["tools"]:
            Draft202012Validator.check_schema(tool["function"]["parameters"])
        if row["category"] in {"irrelevance", "live_irrelevance", "live_relevance"}:
            continue
        functions = {f["name"]: f for f in row["bfcl_functions"]}
        aliases = {original: alias for alias, original in row["name_map"].items()}
        calls = []
        for truth in row["ground_truth"]:
            name, parameters = next(iter(truth.items()))
            schema = functions[name]["parameters"]
            arguments = {}
            for parameter, choices in parameters.items():
                if parameter not in schema["required"] and "" in choices:
                    continue
                value = next((x for x in choices if x != ""), choices[0] if choices else [])
                parameter_schema = schema["properties"].get(parameter, {})
                kind = parameter_schema.get("type")
                if kind == "dict" and isinstance(value, dict):
                    value = dictionary_candidate(value)
                if kind in {"array", "tuple"} and parameter_schema.get("items", {}).get("type") == "dict" and isinstance(value, list):
                    value = [dictionary_candidate(item) for item in value]
                arguments[parameter] = value
            calls.append({"name": aliases[name], "arguments": arguments})
        if not score_example(row, calls, checker_path)["correct"]:
            observed_failures.add(row["id"])
    assert observed_failures == present_ids.intersection(KNOWN_ANNOTATION_ISSUES)
