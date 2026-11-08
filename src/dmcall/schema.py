"""The local schema check S(c) of the paper.

S(c) is true when the candidate is readable, names an available function and supplies every
required argument with its declared JSON type. Value constraints (enum, ranges, lengths,
patterns, formats) are not checked here: whether a value fits the request and the parameter
description is the decision model's question. A value equal to a parameter's declared default
is admissible even when its JSON type differs (for example ``null``), and, as in a Python call,
keyword names outside a closed parameter list are rejected. An empty collection (no call) passes.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for

__all__ = ["check_schema", "function_definitions"]

# Only the structural typing part of a schema is checked.
_TYPE_KEYWORDS = frozenset({
    "$schema", "$id", "$ref", "$defs", "definitions", "type", "properties", "required",
    "items", "prefixItems", "additionalProperties", "patternProperties", "anyOf", "allOf",
    "oneOf", "default",
})


def function_definitions(tools: Any) -> list[dict[str, Any]]:
    """The observable part of each tool: name, description and parameters.

    Any other field of a tool definition stays out of what the decision model sees.
    """
    if not isinstance(tools, list):
        raise ValueError("tools must be a list of function definitions")
    definitions: list[dict[str, Any]] = []
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError("Every tool must be a function definition object")
        definition = tool.get("function", tool)
        if not isinstance(definition, dict):
            raise ValueError("Every tool.function must be an object")
        name = definition.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Available function names must be nonempty and unique")
        names.add(name)
        definitions.append({
            "name": name,
            "description": definition.get("description", ""),
            "parameters": copy.deepcopy(definition.get("parameters", {"type": "object", "properties": {}})),
        })
    return definitions


def _error(check: str, path: str, message: str) -> dict[str, str]:
    return {"check": check, "path": path, "message": message}


def _type_schema(schema: Any) -> Any:
    """A schema's types, required names and nesting without value constraints."""
    if isinstance(schema, list):
        return [_type_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return copy.deepcopy(schema)
    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _TYPE_KEYWORDS:
            continue
        if key in {"properties", "patternProperties", "$defs", "definitions"} and isinstance(value, dict):
            result[key] = {name: _type_schema(item) for name, item in value.items()}
        elif key in {"items", "prefixItems", "additionalProperties"}:
            result[key] = _type_schema(value)
        elif key in {"anyOf", "allOf", "oneOf"} and isinstance(value, list):
            branches = [_type_schema(item) for item in value]
            if key == "allOf":
                result.setdefault("allOf", []).extend(branches)
            else:
                # Without value constraints, oneOf branches may overlap; any branch suffices.
                result.setdefault("allOf", []).append({"anyOf": branches})
        else:
            result[key] = copy.deepcopy(value)
    return result


def _is_declared_default(issue: Any) -> bool:
    """A value equal to the parameter's own declared default is admissible."""
    if issue.validator != "type" or not isinstance(issue.schema, dict) or "default" not in issue.schema:
        return False
    try:
        return json.dumps(issue.instance, sort_keys=True) == json.dumps(issue.schema["default"], sort_keys=True)
    except (TypeError, ValueError):
        return False


def check_schema(tools: Any, calls: Any) -> dict[str, Any]:
    """Run S(c) on proposed ``calls`` (``[{"name", "arguments"}]``) against the available ``tools``.

    Returns ``accepted`` (S(c)), ``function_pass`` and ``arguments_pass`` (which part failed),
    ``well_formed`` (the candidate is readable, so the decision model can judge it), ``errors``
    and the ``feedback`` record of the check. Invalid tool definitions raise ``ValueError``:
    they are configuration errors, not a property of the candidate.
    """
    definitions = function_definitions(tools)
    validators: dict[str, Any] = {}
    schemas: dict[str, dict[str, Any]] = {}
    for definition in definitions:
        schema = definition["parameters"]
        if not isinstance(schema, dict):
            raise ValueError("Function parameters must be a JSON Schema object")
        validator_type = validator_for(schema, default=Draft202012Validator)
        try:
            validator_type.check_schema(schema)
        except SchemaError as exc:
            raise ValueError("An available function has an invalid JSON Schema") from exc
        validators[definition["name"]] = validator_type(_type_schema(schema))
        schemas[definition["name"]] = schema

    errors: list[dict[str, str]] = []
    well_formed = isinstance(calls, list)
    if not well_formed:
        errors.extend([_error("function", "calls", "Proposed calls must be a list."),
                       _error("arguments", "calls", "No valid call collection to inspect.")])
    else:
        for index, call in enumerate(calls):
            path = f"calls[{index}]"
            if not isinstance(call, dict):
                well_formed = False
                errors.extend([_error("function", path, "Each call must be an object."),
                               _error("arguments", path, "No argument object to inspect.")])
                continue
            name = call.get("name")
            if not isinstance(name, str) or not name:
                well_formed = False
                errors.append(_error("function", path + ".name", "Name must be a nonempty string."))
            elif name not in validators:
                errors.append(_error("function", path + ".name", "Selected function is not available."))
            arguments = call.get("arguments")
            if not isinstance(arguments, dict):
                well_formed = False
                errors.append(_error("arguments", path + ".arguments", "Arguments must be a JSON object."))
                continue
            try:
                json.dumps(arguments, allow_nan=False)
            except (TypeError, ValueError):
                well_formed = False
                errors.append(_error("arguments", path + ".arguments", "Arguments must be finite JSON values."))
                continue
            if not isinstance(name, str) or name not in validators:
                errors.append(_error("arguments", path + ".arguments",
                                     "Cannot validate arguments for an unavailable function."))
                continue
            for issue in sorted(validators[name].iter_errors(arguments), key=lambda e: str(list(e.path))):
                if _is_declared_default(issue):
                    continue
                suffix = "".join(f"[{json.dumps(component)}]" for component in issue.path)
                errors.append(_error("arguments", path + ".arguments" + suffix, issue.message))
            schema = schemas[name]
            # An ordinary function signature has closed top-level keyword names.
            # Respect an explicit additionalProperties or patternProperties rule.
            if "properties" in schema and "additionalProperties" not in schema and "patternProperties" not in schema:
                for unexpected in sorted(set(arguments) - set(schema["properties"])):
                    errors.append(_error("arguments", path + ".arguments", f"Unknown parameter: {unexpected}."))

    function_pass = not any(item["check"] == "function" for item in errors)
    arguments_pass = not any(item["check"] == "arguments" for item in errors)
    return {
        "function_pass": function_pass,
        "arguments_pass": arguments_pass,
        "accepted": function_pass and arguments_pass,
        "well_formed": well_formed,
        "errors": errors,
        "feedback": {
            "source": "deterministic_schema_validation",
            "function_pass": function_pass,
            "arguments_pass": arguments_pass,
            "schema_errors": errors,
        },
    }
