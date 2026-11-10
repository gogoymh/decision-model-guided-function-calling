"""The schema check S(c) and the feedback representations Φ."""

import pytest

import dmcall
from dmcall.feedback import represent, represent_history, single_check
from conftest import WEATHER

TOOLS = [WEATHER]


def check(arguments, tools=TOOLS, name="get_weather"):
    return dmcall.check_schema(tools, [{"name": name, "arguments": arguments}])


# S(c) ---------------------------------------------------------------------------------------
def test_valid_call_and_no_call_pass():
    assert check({"city": "Seoul"})["accepted"] is True
    empty = dmcall.check_schema(TOOLS, [])
    assert empty["accepted"] is True and empty["well_formed"] is True and empty["errors"] == []


@pytest.mark.parametrize("arguments", [{}, {"city": "Seoul", "invented": 1}, {"city": 3}])
def test_argument_structure_errors(arguments):
    result = check(arguments)
    assert result["function_pass"] is True and result["arguments_pass"] is False and result["well_formed"] is True


def test_unknown_function_is_readable_but_not_available():
    result = check({}, name="invented")
    assert result["function_pass"] is False and result["arguments_pass"] is False and result["well_formed"] is True


@pytest.mark.parametrize("calls", [None, ["get_weather"], [{"name": "get_weather", "arguments": "{}"}],
                                   [{"name": "get_weather", "arguments": {"city": float("nan")}}],
                                   [{"name": "", "arguments": {}}]])
def test_unreadable_candidates_are_not_well_formed(calls):
    result = dmcall.check_schema(TOOLS, calls)
    assert result["accepted"] is False and result["well_formed"] is False


def test_types_and_required_names_only_not_value_constraints():
    tools = [{"type": "function", "function": {"name": "book", "description": "", "parameters": {
        "type": "object",
        "properties": {
            "genre": {"type": "string", "enum": ["Horror", "Comedy"]},
            "rooms": {"type": "integer", "minimum": 1},
            "metrics": {"type": "array", "enum": ["view", "trust"], "items": {"type": "string"}},
            "start": {"type": "string", "default": None},
            "end": {"type": "string"},
            "filters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
        "required": ["genre"],
    }}}]
    # Values outside an enum or range are the decision model's question; a declared null default is admissible.
    passed = check({"genre": "Any", "rooms": 0, "metrics": ["view"], "start": None, "filters": {"city": "Seoul"}},
                   tools, "book")
    assert passed["accepted"] is True and passed["errors"] == []
    for arguments in ({"genre": True}, {"genre": "Any", "rooms": "2"}, {"genre": "Any", "end": None},
                      {"genre": "Any", "metrics": "view"}, {"rooms": 1}, {"genre": "Any", "extra": 1},
                      {"genre": "Any", "filters": {}}):
        result = check(arguments, tools, "book")
        assert result["function_pass"] is True and result["arguments_pass"] is False, arguments


def test_alternatives_and_open_keywords():
    parameters = {"type": "object", "additionalProperties": False, "required": ["x"], "properties": {
        "x": {"oneOf": [{"type": "string", "enum": ["a"]}, {"type": "string", "maxLength": 1}]}}}
    tools = [{"type": "function", "function": {"name": "f", "description": "", "parameters": parameters}}]
    assert check({"x": "abc"}, tools, "f")["accepted"] is True
    for arguments in ({"x": 3}, {}, {"x": "a", "y": 1}):
        assert check(arguments, tools, "f")["accepted"] is False, arguments
    open_tools = [{"type": "function", "function": {**WEATHER["function"], "parameters": {
        **WEATHER["function"]["parameters"], "additionalProperties": True}}}]
    assert check({"city": "Seoul", "extra": 1}, open_tools)["accepted"] is True


def test_invalid_tool_schema_is_a_configuration_error():
    bad = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object",
                                                                           "properties": {"a": {"type": "nonsense"}}}}}]
    with pytest.raises(ValueError, match="invalid JSON Schema"):
        dmcall.check_schema(bad, [])
    with pytest.raises(ValueError, match="unique"):
        dmcall.check_schema([WEATHER, WEATHER], [])


def _tool(parameters):
    return [{"type": "function", "function": {"name": "f", "description": "", "parameters": parameters}}]


@pytest.mark.parametrize("parameters", [
    {"type": "object", "properties": {"a": {"$ref": "#/$defs/missing"}}},
    {"type": "object", "properties": {"a": {"$ref": "https://example.com/a.json"}}},
    {"type": "object", "properties": {"a": {"$ref": "#/properties/a"}}},
    {"type": "object", "properties": {"a": {"$ref": "#/type"}}},
    {"type": "object", "properties": {"a": {"$ref": "#/$defs/x"}}, "$defs": {"x": {"allOf": [{"$ref": "#/$defs/x"}]}}},
], ids=["missing", "remote", "itself", "scalar", "loop-through-allOf"])
def test_a_reference_that_cannot_be_followed_is_a_configuration_error(parameters):
    """It fails when the tools are checked, not later with a call in hand, and nothing is fetched."""
    with pytest.raises(ValueError, match=r"\$ref that cannot be resolved"):
        dmcall.check_schema(_tool(parameters), [])


def test_nested_resource_references_keep_their_own_scope():
    parameters = {"type": "object", "properties": {"nested": {
        "$id": "nested", "type": "object", "$defs": {"number": {"type": "integer"}},
        "properties": {"value": {"$ref": "#/$defs/number"}},
    }}}
    definition = dmcall.schema_of({"name": "f", "parameters": parameters})
    assert dmcall.check_schema([definition], [{"name": "f", "arguments": {"nested": {"value": 1}}}])["accepted"]
    assert not dmcall.check_schema([definition], [{"name": "f", "arguments": {"nested": {"value": "one"}}}])["accepted"]


def test_references_inside_the_parameters_and_recursive_schemas_work():
    tree = _tool({"type": "object", "properties": {"root": {"$ref": "#/$defs/node"}}, "$defs": {"node": {
        "type": "object", "properties": {"value": {"type": "integer"}, "child": {"$ref": "#/$defs/node"}},
        "required": ["value"]}}})
    good = [{"name": "f", "arguments": {"root": {"value": 1, "child": {"value": 2}}}}]
    bad = [{"name": "f", "arguments": {"root": {"value": 1, "child": {"value": "two"}}}}]
    assert dmcall.check_schema(tree, good)["accepted"]
    assert dmcall.check_schema(tree, bad)["errors"] == [{
        "check": "arguments", "path": 'calls[0].arguments["root"]["child"]["value"]',
        "message": "'two' is not of type 'integer'"}]
    # A value that only looks like a reference is data, and a property may be named "$ref".
    data = _tool({"type": "object", "properties": {"a": {"type": "object", "default": {"$ref": "#/nope"}},
                                                    "$ref": {"type": "string"}}})
    assert dmcall.check_schema(data, [{"name": "f", "arguments": {"a": {}, "$ref": "x"}}])["accepted"]


def test_a_tool_schema_is_compiled_once():
    from dmcall.schema import _compiled

    tools = _tool({"type": "object", "properties": {"only_here": {"type": "string"}}})
    dmcall.check_schema(tools, [])
    before = _compiled.cache_info()
    for _ in range(3):
        assert dmcall.check_schema(tools, [{"name": "f", "arguments": {"only_here": "x"}}])["accepted"]
    after = _compiled.cache_info()
    assert after.misses == before.misses and after.hits == before.hits + 3


# Φ ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("p, expected", [(0, "very likely incorrect"), (0.19, "very likely incorrect"),
                                         (0.2, "likely incorrect"), (0.48, "uncertain"), (0.52, "uncertain"),
                                         (0.6, "likely correct"), (0.8, "very likely correct"),
                                         (1, "very likely correct"), (None, None)])
def test_label_bands(p, expected):
    assert dmcall.label(p) == expected


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), True, "0.5"])
def test_label_rejects_non_probabilities(bad):
    with pytest.raises(ValueError):
        dmcall.label(bad)


FEEDBACK = {
    "source": "jev_noul_and_deterministic_schema_validation",
    "checks": {"function_selection": {"pass": True, "probability_yes": 0.95, "threshold": 0.5, "schema_pass": True},
               "arguments": {"skipped": False, "pass": False, "probability_yes": 0.12, "threshold": 0.5,
                             "schema_pass": True}},
    "failed_checks": ["arguments"],
    "schema_errors": [{"check": "arguments", "path": "calls[0].arguments", "message": "'probability_yes' 0.3"}],
    "instruction": "Reconsider ... The probabilities are verifier judgments, not ground truth or explanations.",
}


def test_hybrid_keeps_numbers_and_thresholds_and_adds_labels():
    out = represent(FEEDBACK, "hybrid")
    arguments = out["checks"]["arguments"]
    assert list(arguments) == ["skipped", "pass", "probability_yes", "feedback_label", "threshold", "schema_pass"]
    assert arguments["probability_yes"] == 0.12 and arguments["feedback_label"] == "very likely incorrect"
    assert out["checks"]["function_selection"]["feedback_label"] == "very likely correct"
    assert out["schema_errors"] == FEEDBACK["schema_errors"] and out["instruction"] == FEEDBACK["instruction"]
    assert "feedback_label" not in FEEDBACK["checks"]["arguments"]  # the input is not changed


def test_text_replaces_numbers_and_thresholds_with_labels():
    out = represent(FEEDBACK, "text")
    assert out["checks"]["arguments"] == {"skipped": False, "pass": False, "feedback_label": "very likely incorrect",
                                          "schema_pass": True}
    assert out["instruction"].endswith("The feedback labels are verifier judgments, not ground truth or explanations.")
    assert out["schema_errors"] == FEEDBACK["schema_errors"]  # task diagnostics stay verbatim


def test_numeric_is_unchanged_and_skipped_questions_stay_null():
    assert represent(FEEDBACK, "numeric") == FEEDBACK
    skipped = {"checks": {"arguments": {"skipped": True, "pass": None, "probability_yes": None, "threshold": 0.5}}}
    assert represent(skipped, "hybrid")["checks"]["arguments"]["feedback_label"] is None
    assert represent(skipped, "text")["checks"]["arguments"] == {"skipped": True, "pass": None, "feedback_label": None}
    with pytest.raises(ValueError, match="feedback"):
        represent(FEEDBACK, "words")
    history = [{"calls": [], "feedback": FEEDBACK}]
    assert represent_history(history, "text")[0]["feedback"] == represent(FEEDBACK, "text")
    with pytest.raises(ValueError):
        represent_history([{"calls": []}], "hybrid")


def test_single_check_removes_only_the_other_question():
    function_only = single_check(FEEDBACK, "function")
    assert list(function_only["checks"]) == ["function_selection"] and function_only["failed_checks"] == []
    arguments_only = single_check(FEEDBACK, "arguments")
    assert list(arguments_only["checks"]) == ["arguments"] and arguments_only["failed_checks"] == ["arguments"]
    assert arguments_only["schema_errors"] == FEEDBACK["schema_errors"]
    with pytest.raises(ValueError):
        single_check(FEEDBACK, "both")
