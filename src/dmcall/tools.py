"""Tools from plain Python functions, and calls from any SDK.

A tool is a function the model may call: a name, a description and a JSON Schema
for its keyword arguments. ``schema_of`` builds one from a Python function's
signature, type hints and docstring (Google, NumPy or Sphinx style), and also
accepts OpenAI, Anthropic and Gemini style definitions. The optional ``tool``
decorator overrides the inferred name, description or schema.
"""

from __future__ import annotations

import collections.abc
import enum
import functools
import inspect
import json
import re
import types
import typing
from copy import deepcopy
from typing import Any, Callable

from jsonschema.exceptions import SchemaError

from .schema import _ReferenceError, _validator

__all__ = ["invoke", "normalize_calls", "schema_of", "tool", "toolset"]

_ATTRIBUTE = "__dmcall_tool__"
_MISSING = object()
_SIMPLE = {str: "string", int: "integer", float: "number", bool: "boolean"}
_ARRAYS = (list, tuple, set, frozenset, collections.abc.Sequence, collections.abc.MutableSequence,
           collections.abc.Set, collections.abc.MutableSet, collections.abc.Collection, collections.abc.Iterable)
_MAPPINGS = (dict, collections.abc.Mapping, collections.abc.MutableMapping)
_PARAMETER_SECTIONS = {"args", "arguments", "parameters", "params", "keyword args", "keyword arguments",
                       "other parameters"}
_SECTIONS = _PARAMETER_SECTIONS | {"returns", "return", "yields", "yield", "raises", "examples", "example",
                                   "notes", "note", "attributes", "see also", "references", "warnings",
                                   "warning", "warns", "todo", "methods"}
_GOOGLE_ENTRY = re.compile(r"\*{0,2}(?P<name>\w+)\s*(?:\([^)]*\))?\s*:\s*(?P<text>.*)$")
_NUMPY_ENTRY = re.compile(r"\*{0,2}(?P<name>\w+)\s*(?::.*)?$")
_SPHINX_PARAM = re.compile(r":param\s+(?:[^:]*\s)?(?P<name>\w+)\s*:\s*(?P<text>.*)$")
_SPHINX_FIELD = re.compile(r":(?:type|rtype|returns?|raises?|var|ivar|cvar|yields?|meta)\b")


# Docstrings ------------------------------------------------------------------
def _parse_doc(doc: str) -> tuple[str, dict[str, str]]:
    """The description and per-parameter text of a Google, NumPy or Sphinx docstring."""
    lines = (doc or "").expandtabs().splitlines()
    summary: list[str] = []
    notes: dict[str, list[str]] = {}
    section = style = current = None
    entry_indent = None
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        title = stripped.rstrip(":").strip().lower()
        underlined = index + 1 < len(lines) and re.fullmatch(r"-{3,}", lines[index + 1].strip()) is not None
        if indent == 0 and stripped.endswith(":") and title in _SECTIONS:
            section, style, current, entry_indent = title, "google", None, None
            index += 1
            continue
        if indent == 0 and underlined and title in _SECTIONS:
            section, style, current, entry_indent = title, "numpy", None, None
            index += 2
            continue
        sphinx = _SPHINX_PARAM.match(stripped)
        if sphinx:
            section, style, current, entry_indent = "sphinx", "sphinx", sphinx["name"], indent
            notes[current] = [sphinx["text"]]
        elif _SPHINX_FIELD.match(stripped):
            section, style, current, entry_indent = "sphinx", "sphinx", None, indent
        elif section is None:
            summary.append(line)
        elif stripped and section in _PARAMETER_SECTIONS:
            if entry_indent is None:
                entry_indent = indent
            if indent <= entry_indent:
                match = (_GOOGLE_ENTRY if style == "google" else _NUMPY_ENTRY).match(stripped)
                current = match["name"] if match else None
                if current:
                    notes[current] = [match["text"]] if style == "google" else []
            elif current:
                notes[current].append(stripped)
        elif stripped and section == "sphinx" and current and indent > entry_indent:
            notes[current].append(stripped)
        index += 1
    description = re.sub(r"\n{3,}", "\n\n", "\n".join(summary)).strip()
    return description, {name: " ".join(part for part in parts if part).strip() for name, parts in notes.items()}


# Type hints ------------------------------------------------------------------
def _json_type(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    raise TypeError(f"Choice {value!r} is not a JSON string, number, boolean or null.")


def _choices(values):
    kinds = {_json_type(value) for value in values}
    schema: dict[str, Any] = {"enum": list(values)}
    if len(kinds) == 1:
        schema = {"type": kinds.pop(), **schema}
    return schema


def _is_union(origin):
    return origin is typing.Union or origin is types.UnionType


def _without_none(annotation):
    """X for Optional[X], so that a None default is a declared default rather than a type."""
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is typing.Annotated:
        return typing.Annotated[(_without_none(args[0]), *args[1:])]
    if _is_union(origin):
        rest = tuple(arg for arg in args if arg is not type(None))
        if rest and len(rest) < len(args):
            return rest[0] if len(rest) == 1 else typing.Union[rest]
    return annotation


def _schema(annotation, where: str) -> tuple[dict, str | None]:
    """JSON Schema for a type annotation, and a description from ``Annotated`` text if any."""
    if annotation is inspect.Parameter.empty or annotation is Any or annotation is object:
        return {}, None
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is typing.Annotated:
        schema, text = _schema(args[0], where)
        texts = [item for item in args[1:] if isinstance(item, str)]
        return schema, texts[0] if texts else text
    if annotation is None or annotation is type(None):
        return {"type": "null"}, None
    if annotation in _SIMPLE:
        return {"type": _SIMPLE[annotation]}, None
    if origin is typing.Literal:
        return _choices(args), None
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return _choices([member.value for member in annotation]), None
    if _is_union(origin):
        schemas = [_schema(arg, where)[0] for arg in args]
        if any(not schema for schema in schemas):
            return {}, None
        return ({"anyOf": schemas} if len(schemas) > 1 else schemas[0]), None
    if annotation in _ARRAYS or origin in _ARRAYS:
        schema: dict[str, Any] = {"type": "array"}
        if origin is tuple and args and not (len(args) == 2 and args[1] is Ellipsis):
            schema["prefixItems"] = [_schema(arg, where)[0] for arg in args]
            schema["minItems"] = schema["maxItems"] = len(args)
        elif args:
            items = _schema(args[0], where)[0]
            if items:
                schema["items"] = items
        return schema, None
    if annotation in _MAPPINGS or origin in _MAPPINGS:
        schema = {"type": "object"}
        if len(args) == 2:
            values = _schema(args[1], where)[0]
            if values:
                schema["additionalProperties"] = values
        return schema, None
    if typing.is_typeddict(annotation):
        hints = typing.get_type_hints(annotation, include_extras=True)
        properties = {}
        for name, hint in hints.items():
            schema, text = _schema(hint, f"{where}.{name}")
            properties[name] = {**schema, "description": text} if text else schema
        required = [name for name in hints if name in getattr(annotation, "__required_keys__", ())]
        schema = {"type": "object", "properties": properties, "additionalProperties": False}
        if required:
            schema["required"] = required
        return schema, None
    raise TypeError(f"{where}: cannot describe {annotation!r} as JSON Schema; "
                    "pass parameters= to dmcall.tool with a JSON Schema.")


def _annotations(function) -> dict:
    """The annotations as written. On Python 3.14+ an undefined name stays a ForwardRef instead of raising."""
    try:
        import annotationlib
    except ImportError:
        return dict(getattr(function, "__annotations__", None) or {})
    return annotationlib.get_annotations(function, format=annotationlib.Format.FORWARDREF)


def _hints(function) -> dict:
    """Type hints by parameter name. An annotation that cannot be resolved leaves only its own parameter untyped."""
    try:
        return typing.get_type_hints(function, include_extras=True)
    except Exception:
        pass
    try:
        annotations = _annotations(function)
    except Exception:
        return {}
    owner = function
    while hasattr(owner, "__wrapped__"):
        owner = owner.__wrapped__
    namespace = getattr(owner, "__globals__", None)
    hints = {}
    for name, annotation in annotations.items():
        def probe():
            pass
        probe.__annotations__ = {name: annotation}
        try:
            hints.update(typing.get_type_hints(probe, globalns=namespace, include_extras=True))
        except Exception:
            continue
    return hints


def _signature(function):
    try:
        return inspect.signature(function)
    except NameError:  # Python 3.14+ resolves annotations when they are read
        import annotationlib
        return inspect.signature(function, annotation_format=annotationlib.Format.FORWARDREF)


def _json_value(value):
    if isinstance(value, enum.Enum):
        value = value.value
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError):
        return _MISSING


def _infer(function: Callable, name: str | None = None) -> dict:
    """A function definition from a Python function's signature, type hints and docstring."""
    if not callable(function):
        raise TypeError("A tool is a Python function or a function definition dict.")
    name = name or getattr(function, "__name__", None)
    if not isinstance(name, str) or not name or name == "<lambda>":
        raise ValueError("Name this tool: dmcall.tool(function, name=...).")
    description, notes = _parse_doc(inspect.getdoc(function) or "")
    try:
        signature = _signature(function)
    except (TypeError, ValueError):
        raise TypeError(f"{name}: cannot read its signature; pass parameters= to dmcall.tool.") from None
    hints = _hints(function)
    properties, required, open_keywords = {}, [], False
    for parameter in signature.parameters.values():
        if parameter.kind is parameter.VAR_KEYWORD:
            open_keywords = True
            continue
        if parameter.kind is parameter.VAR_POSITIONAL:
            continue
        if parameter.kind is parameter.POSITIONAL_ONLY:
            raise TypeError(f"{name}: positional-only parameter {parameter.name!r} cannot be passed by name.")
        annotation = hints.get(parameter.name, parameter.annotation)
        if isinstance(annotation, (str, typing.ForwardRef)):  # unresolved: this parameter stays untyped
            annotation = inspect.Parameter.empty
        has_default = parameter.default is not parameter.empty
        if has_default and parameter.default is None:
            annotation = _without_none(annotation)
        schema, text = _schema(annotation, f"{name}.{parameter.name}")
        text = notes.get(parameter.name) or text
        if text:
            schema["description"] = text
        if has_default:
            default = _json_value(parameter.default)
            if default is not _MISSING:
                schema["default"] = default
        else:
            required.append(parameter.name)
        properties[parameter.name] = schema
    parameters: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    if open_keywords:
        parameters["additionalProperties"] = True
    return {"name": name, "description": description, "parameters": parameters}


def _checked(definition: dict) -> dict:
    name = definition.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Every tool needs a nonempty name.")
    if not isinstance(definition.get("description"), str):
        raise ValueError(f"Tool {name!r}: the description must be text.")
    parameters = definition.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("type", "object") != "object":
        raise ValueError(f"Tool {name!r}: parameters must be a JSON Schema object with type 'object'.")
    try:
        json.dumps(parameters, allow_nan=False)
        _validator(parameters)
    except _ReferenceError:
        raise ValueError(f"Tool {name!r}: a $ref in its parameters cannot be resolved inside them.") from None
    except (TypeError, ValueError, SchemaError) as exc:
        raise ValueError(f"Tool {name!r}: parameters are not a valid JSON Schema ({type(exc).__name__}).") from None
    return definition


_NAMED_SCHEMAS = frozenset({"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"})
_SCHEMA_CHILDREN = frozenset({
    "items", "prefixItems", "additionalProperties", "additionalItems", "allOf", "anyOf", "oneOf",
    "not", "if", "then", "else", "contains", "propertyNames", "unevaluatedProperties", "unevaluatedItems",
    "contentSchema",
})


def _lower_types(schema):
    """Gemini's OpenAPI-style schemas spell types in capitals (OBJECT, STRING); JSON Schema does not.

    Only schemas are rewritten. The values of ``enum``, ``const``, ``default``, ``example`` and
    ``examples`` are the caller's data and stay as written, and the keys of ``properties`` (and
    of the other maps of named schemas) are names, not keywords.
    """
    if isinstance(schema, list):
        return [_lower_types(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    result = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str) and value.isupper():
            result[key] = value.lower()
        elif key == "type" and isinstance(value, list):
            result[key] = [item.lower() if isinstance(item, str) and item.isupper() else item for item in value]
        elif key in _NAMED_SCHEMAS and isinstance(value, dict):
            result[key] = {name: _lower_types(item) for name, item in value.items()}
        elif key == "dependencies" and isinstance(value, dict):
            result[key] = {name: deepcopy(item) if isinstance(item, list) else _lower_types(item)
                           for name, item in value.items()}
        elif key in _SCHEMA_CHILDREN:
            result[key] = _lower_types(value)
        else:
            result[key] = deepcopy(value)
    return result


# Public helpers ----------------------------------------------------------------
def tool(function: Callable | None = None, *, name: str | None = None, description: str | None = None,
         parameters: dict | None = None):
    """Mark a function as a tool, optionally overriding its name, description or parameter schema.

    Plain functions work as tools without it; use it to customize the schema::

        @dmcall.tool(description="Current weather for a city.")
        def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> dict: ...
    """
    def mark(target):
        if parameters is not None:
            target_name = name or getattr(target, "__name__", None)
            if not isinstance(target_name, str) or target_name == "<lambda>":
                raise ValueError("Name this tool: dmcall.tool(function, name=...).")
            definition = {"name": target_name, "description": "", "parameters": deepcopy(parameters)}
            definition["description"] = _parse_doc(inspect.getdoc(target) or "")[0]
        else:
            definition = _infer(target, name)
        if description is not None:
            definition["description"] = description
        try:
            setattr(target, _ATTRIBUTE, _checked(definition))
        except (AttributeError, TypeError):
            raise TypeError("dmcall.tool decorates a function; pass other callables to tools= directly.") from None
        return target

    return mark(function) if function is not None else mark


def schema_of(tool: Any) -> dict:
    """An OpenAI-style tool definition ``{"type": "function", "function": {...}}`` for a function or dict.

    Dicts may be OpenAI (Chat Completions or Responses), Anthropic (``input_schema``) or
    Gemini (``parameters`` or ``parametersJsonSchema``) function definitions.
    """
    if isinstance(tool, dict):
        definition = tool["function"] if isinstance(tool.get("function"), dict) else tool
        if "functionDeclarations" in definition or "function_declarations" in definition:
            raise ValueError("Pass each Gemini function declaration as its own tool.")
        parameters = next((definition[key] for key in ("parameters", "input_schema", "parametersJsonSchema",
                                                         "parameters_json_schema") if definition.get(key) is not None),
                          {"type": "object", "properties": {}})
        result = {"name": definition.get("name"), "description": definition.get("description") or "",
                  "parameters": _lower_types(deepcopy(parameters))}
    elif callable(tool):
        result = deepcopy(getattr(tool, _ATTRIBUTE, None) or _infer(tool))
    else:
        raise TypeError("A tool is a Python function or a function definition dict.")
    return {"type": "function", "function": _checked(result)}


def toolset(tools: Any = None, functions: dict[str, Callable] | None = None) -> tuple[list[dict], dict]:
    """Tool definitions for the model, and the Python functions that implement them by name."""
    if tools is None:
        tools = []
    elif isinstance(tools, dict) or callable(tools):
        tools = [tools]
    if not isinstance(tools, (list, tuple)):
        raise TypeError("tools must be a list of Python functions or function definition dicts.")
    schemas, registry, names = [], {}, set()
    for item in tools:
        schema = schema_of(item)
        name = schema["function"]["name"]
        if name in names:
            raise ValueError(f"Two tools are named {name!r}.")
        names.add(name)
        schemas.append(schema)
        if callable(item) and not isinstance(item, dict):
            registry[name] = item
    for name, function in (functions or {}).items():
        if name not in names:
            raise ValueError(f"functions= has {name!r}, which is not one of the tools.")
        if not callable(function):
            raise TypeError(f"functions[{name!r}] is not callable.")
        registry[name] = function
    return schemas, registry


def _plain(item):
    for method in ("model_dump", "to_dict"):
        if callable(getattr(item, method, None)):
            return item.model_dump(mode="json") if method == "model_dump" else item.to_dict()
    return item


def normalize_calls(calls: Any) -> list[dict]:
    """Calls as ``[{"name", "arguments"}]`` from DM-Call, OpenAI, Anthropic or Gemini shapes or SDK objects.

    ``None`` means no call (OpenAI SDKs report ``tool_calls=None``). Arguments that are not
    valid JSON become ``None``, which the schema check rejects.
    """
    if calls is None:
        return []
    calls = _plain(calls)
    if isinstance(calls, dict):
        calls = [calls]
    if not isinstance(calls, (list, tuple)):
        raise TypeError("calls must be a call or a list of calls.")
    result = []
    for item in calls:
        item = _plain(item)
        if not isinstance(item, dict):
            raise TypeError("Each call must be a dict or an SDK tool-call object.")
        for key in ("functionCall", "function_call", "function"):
            if isinstance(item.get(key), dict):
                item = item[key]
                break
        arguments: Any = {}
        for key in ("arguments", "input", "args"):
            if key in item:
                arguments = item[key]
                break
        if arguments is None:
            arguments = {}
        elif isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except ValueError:
                arguments = None
        result.append({"name": item.get("name"), "arguments": deepcopy(arguments)})
    return result


def _enum_type(annotation):
    while True:
        origin, args = typing.get_origin(annotation), typing.get_args(annotation)
        if origin is typing.Annotated:
            annotation = args[0]
        elif _is_union(origin):
            members = [arg for arg in args if arg is not type(None)]
            if len(members) != 1:
                return None
            annotation = members[0]
        else:
            break
    return annotation if isinstance(annotation, type) and issubclass(annotation, enum.Enum) else None


def _is_async(function: Callable) -> bool:
    while isinstance(function, functools.partial):
        function = function.func
    return any(inspect.iscoroutinefunction(target) or inspect.isasyncgenfunction(target)
               for target in (function, getattr(function, "__call__", None)))


def invoke(function: Callable, arguments: dict) -> Any:
    """Call a tool's Python function with keyword arguments; Enum parameters receive Enum members."""
    if _is_async(function):
        raise TypeError("Async tools are not supported yet; pass a synchronous function.")
    hints = _hints(function)
    values = dict(arguments)
    for name, value in values.items():
        target = _enum_type(hints.get(name))
        if target is not None and not isinstance(value, target):
            try:
                values[name] = target(value)
            except ValueError:
                pass
    return function(**values)
