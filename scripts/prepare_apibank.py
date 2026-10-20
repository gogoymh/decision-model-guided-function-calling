"""Prepare an offline API-Bank first-call extraction subset, without executing upstream code.

This adapted task predicts the first API call from the preceding dialogue. Every
example receives the same complete API catalog, independently of its target.
It is not the official interactive/API-execution score and contains no no-call
negatives. The source pin and audited exclusions precede any model experiment.

Usage: python scripts/prepare_apibank.py --upstream /path/to/AlibabaResearch/DAMO-ConvAI
       [--output data/apibank-firstcall]
"""

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

REVISION = "faf754671f5b850e1b4d1ff870db37a6a2480df6"
UPSTREAM_URL = "https://github.com/AlibabaResearch/DAMO-ConvAI"
DIALOGUE_PATH = Path("api-bank/lv1-lv2-samples/level-1-given-desc")
API_PATH = Path("api-bank/apis")
SPLIT_SALT = "apibank-firstcall-v1|"
RAW_SHA256 = "10383f74487192a2fdc7594593fdf3ead8c99d1092c154818ff474fd80ca1477"
API_SHA256 = "cb443449b8fbc9d037dabf25145d9837e0615e4b76df021088571290b04563ea"
LICENSE_HASHES = {
    "api-bank/LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "LICENSE": "16f9444e99d9f667b3e376186831d54250039a1d67e820f0955fa651f092454d",
}
NO_API_FILES = {
    "RegisterUser-ForgotPassword-DeleteAccount-GetUserToken-level-2-1.jsonl",
    "SearchEngine-level-1-1.jsonl",
}
SCHEMA_EXCLUSIONS = {
    *(f"CancelTimedSwitch-level-1-{i}.jsonl" for i in range(1, 6)),
    *(f"TimedSwitch-level-1-{i}.jsonl" for i in range(1, 4)),
}
SCHEMA_TYPES = {
    "str": {"type": "string"},
    "int": {"type": "integer"},
    "bool": {"type": "boolean"},
    "list": {"type": "array"},
    "list(str)": {"type": "array", "items": {"type": "string"}},
}


def _bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _jsonl(rows):
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                   for row in rows).encode()


def _file_hashes(root, paths):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def _tree_hash(hashes):
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_sources(root):
    root = Path(root).resolve()
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True)
    if result.stdout.strip() != REVISION:
        raise ValueError(f"API-Bank must use the audited revision {REVISION}.")
    dialogues = _file_hashes(root, (root / DIALOGUE_PATH).glob("*.jsonl"))
    apis = _file_hashes(root, (root / API_PATH).glob("*.py"))
    if len(dialogues) != 214 or _tree_hash(dialogues) != RAW_SHA256:
        raise ValueError("API-Bank dialogue checksum/count mismatch; source labels must not be edited.")
    if len(apis) != 55 or _tree_hash(apis) != API_SHA256:
        raise ValueError("API-Bank API-source checksum/count mismatch.")
    for name, expected in LICENSE_HASHES.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Upstream license checksum mismatch: {name}")
    return {"dialogues": dialogues, "api_sources": apis, "licenses": dict(LICENSE_HASHES)}


def _class_literals(node):
    values = {}
    for statement in node.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "description", "input_parameters", "output_parameters"
                }:
                    # literal_eval parses literals only; upstream imports and methods are never executed.
                    values[target.id] = ast.literal_eval(statement.value)
    if set(values) != {"description", "input_parameters", "output_parameters"}:
        raise ValueError(f"Missing literal API definition for {node.name}")
    return values


def _signature(node):
    call = next((statement for statement in node.body
                 if isinstance(statement, ast.FunctionDef) and statement.name == "call"), None)
    if call is None:
        raise ValueError(f"Missing call signature for {node.name}")
    arguments = call.args.posonlyargs + call.args.args
    required = [argument.arg for argument in arguments[:len(arguments) - len(call.args.defaults)]
                if argument.arg != "self"]
    required += [argument.arg for argument, default in zip(call.args.kwonlyargs, call.args.kw_defaults)
                 if default is None]
    return required


def read_catalog(root):
    definitions = {}
    for path in sorted((Path(root) / API_PATH).glob("*.py")):
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in module.body:
            if not isinstance(node, ast.ClassDef) or not any(
                    isinstance(base, ast.Name) and base.id == "API" for base in node.bases):
                continue
            if node.name in definitions:
                raise ValueError(f"Duplicate API class: {node.name}")
            literal = _class_literals(node)
            required = _signature(node)
            inputs = literal["input_parameters"]
            if not isinstance(literal["description"], str) or not isinstance(inputs, dict):
                raise ValueError(f"Invalid literal API definition: {node.name}")
            if not set(required).issubset(inputs):
                raise ValueError(f"Call signature and schema disagree: {node.name}")
            properties = {}
            for name, parameter in inputs.items():
                declared = parameter.get("type")
                if declared not in SCHEMA_TYPES:
                    raise ValueError(f"Unsupported API-Bank parameter type: {node.name}.{name}: {declared}")
                properties[name] = {**deepcopy(SCHEMA_TYPES[declared]),
                                    "description": parameter.get("description", "")}
            definitions[node.name] = {
                "definition": literal,
                "required": required,
                "tool": {"type": "function", "function": {
                    "name": node.name, "description": literal["description"],
                    "parameters": {"type": "object", "properties": properties,
                                   "required": required, "additionalProperties": False},
                }},
            }
    if not definitions:
        raise ValueError("No literal API subclasses found.")
    return dict(sorted(definitions.items()))


def coerce_argument(value, declared):
    """Prespecified declared-type normalization, never model-text evaluation or target-based schema repair."""
    if declared == "str" and isinstance(value, str):
        return value
    if declared == "int":
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value):
            return int(value)
    if declared == "bool":
        if isinstance(value, bool):
            return value
        if value in ("True", "False"):
            return value == "True"
    if declared in {"list", "list(str)"}:
        parsed = ast.literal_eval(value) if isinstance(value, str) else deepcopy(value)
        if isinstance(parsed, list) and (declared == "list" or all(isinstance(item, str) for item in parsed)):
            # A JSON round trip rejects sets, NaN, non-string dictionary keys, and other non-JSON literals.
            def check(item):
                if item is None or isinstance(item, (str, bool, int)):
                    return
                if isinstance(item, float) and math.isfinite(item):
                    return
                if isinstance(item, list):
                    for member in item:
                        check(member)
                    return
                if isinstance(item, dict) and all(isinstance(key, str) for key in item):
                    for member in item.values():
                        check(member)
                    return
                raise ValueError("Array argument contains a non-JSON literal.")
            check(parsed)
            return parsed
    raise ValueError(f"Argument does not match declared type {declared}.")


def normalize_target(target, catalog):
    name, parameters = target["api_name"], target["param_dict"]
    if name not in catalog or not isinstance(parameters, dict):
        raise ValueError("Unknown API or invalid target parameter object.")
    definition = catalog[name]
    inputs = definition["definition"]["input_parameters"]
    issues = []
    if set(parameters) - set(inputs):
        issues.append({"kind": "unknown_arguments", "names": sorted(set(parameters) - set(inputs))})
    if set(definition["required"]) - set(parameters):
        issues.append({"kind": "missing_required", "names": sorted(set(definition["required"]) - set(parameters))})
    arguments, coerced = {}, []
    for key, value in parameters.items():
        if key not in inputs:
            continue
        try:
            arguments[key] = coerce_argument(value, inputs[key]["type"])
        except (ValueError, TypeError, SyntaxError):
            issues.append({"kind": "argument_type", "name": key, "declared_type": inputs[key]["type"]})
        else:
            if type(arguments[key]) is not type(value):
                coerced.append(key)
    return None if issues else [{"name": name, "arguments": arguments}], issues, sorted(coerced)


def family_split(filename):
    family = re.sub(r"-\d+$", "", Path(filename).stem)
    bucket = int(hashlib.sha256((SPLIT_SALT + family).encode()).hexdigest(), 16) % 5
    return family, "dev" if bucket == 0 else "test"


def model_input(example):
    """Only public dialogue and the fixed tool catalog; labels/source metadata never enter a model prompt."""
    return deepcopy({key: example[key] for key in ("messages", "tools")})


def score_example(example, calls):
    """Exact first-call extraction score, calculated after generation without executing an API.

    Both sides use the declared-type normalization fixed before model runs. A
    different function or no call is incorrect; argument accuracy is conditional
    on selecting the target function. This is not API-Bank's executable metric.
    """
    if not example.get("scorable") or not isinstance(example.get("ground_truth"), list):
        raise ValueError("Schema-incompatible API-Bank targets must not enter scored runs.")
    truth = example["ground_truth"]
    if len(truth) != 1:
        raise ValueError("First-call extraction requires exactly one reference call.")
    result = {"correct": False, "function_correct": False, "arguments_correct": None,
              "metric": "apibank_firstcall_exact", "errors": []}
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
        result["errors"] = ["Expected one complete API-call candidate."]
        return result
    call = calls[0]
    if call.get("name") != truth[0]["name"]:
        result["errors"] = ["Function selection mismatch."]
        return result
    result["function_correct"] = True
    result["arguments_correct"] = False
    parameters = next((tool["function"]["parameters"] for tool in example["tools"]
                       if tool["function"]["name"] == call["name"]), None)
    arguments = call.get("arguments")
    if parameters is None or not isinstance(arguments, dict):
        result["errors"] = ["Candidate arguments are not an object for a documented function."]
        return result
    if set(arguments) - set(parameters["properties"]) or set(parameters["required"]) - set(arguments):
        result["errors"] = ["Candidate arguments violate documented names or requiredness."]
        return result
    normalized = {}
    for name, value in arguments.items():
        schema = parameters["properties"][name]
        kind = schema["type"]
        declared = {"string": "str", "integer": "int", "boolean": "bool", "array": "list"}[kind]
        if kind == "array" and schema.get("items", {}).get("type") == "string":
            declared = "list(str)"
        try:
            normalized[name] = coerce_argument(value, declared)
        except (ValueError, TypeError, SyntaxError):
            result["errors"] = ["Candidate argument has an invalid declared type."]
            return result
    result["arguments_correct"] = normalized == truth[0]["arguments"]
    result["correct"] = result["arguments_correct"]
    if not result["correct"]:
        # No gold values enter errors or repair feedback. Scoring is strictly post-generation.
        result["errors"] = ["Argument value mismatch."]
    return result


def build_dataset(root, catalog):
    tools = [definition["tool"] for definition in catalog.values()]
    firstcalls, exclusions = [], []
    for path in sorted((Path(root) / DIALOGUE_PATH).glob("*.jsonl")):
        dialogue = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        position = next((index for index, turn in enumerate(dialogue) if turn.get("role") == "API"), None)
        if position is None:
            exclusions.append({"source_file": path.name, "reason": "no_api_target",
                               "note": "Not a no-call negative; this file has no first API target."})
            continue
        messages = []
        for turn in dialogue[:position]:
            role = {"User": "user", "AI": "assistant"}.get(turn.get("role"))
            if role is None or not isinstance(turn.get("text"), str):
                raise ValueError(f"Unexpected first-call prefix in {path.name}")
            messages.append({"role": role, "content": turn["text"]})
        target = dialogue[position]
        truth, issues, coerced = normalize_target(target, catalog)
        family, split = family_split(path.name)
        example = {
            "id": f"apibank-firstcall/{path.stem}", "category": "apibank_firstcall", "family": family,
            "split": split, "dataset_revision": REVISION, "source_file": path.name,
            "target_turn": position, "messages": messages, "tools": deepcopy(tools),
            "ground_truth": truth,
            "raw_ground_truth": {"api_name": target["api_name"], "param_dict": deepcopy(target["param_dict"])},
            "scorable": not issues, "issues": issues, "coerced_arguments": coerced,
        }
        firstcalls.append(example)
        if issues:
            exclusions.append({"id": example["id"], "source_file": path.name,
                               "reason": "schema_incompatible_target", "issues": issues,
                               "note": "Preserve the raw target; do not rename device_id to name."})
    return firstcalls, exclusions


def prepare(upstream, output):
    root, output = Path(upstream).resolve(), Path(output)
    hashes = verify_sources(root)
    catalog = read_catalog(root)
    firstcalls, exclusions = build_dataset(root, catalog)
    eligible = [row for row in firstcalls if row["scorable"]]
    dev = [row for row in eligible if row["split"] == "dev"]
    test = [row for row in eligible if row["split"] == "test"]
    actual_no_api = {row["source_file"] for row in exclusions if row["reason"] == "no_api_target"}
    actual_schema = {row["source_file"] for row in exclusions if row["reason"] == "schema_incompatible_target"}
    if actual_no_api != NO_API_FILES or actual_schema != SCHEMA_EXCLUSIONS:
        raise ValueError("First-call exclusions differ from the prespecified source audit.")
    if (len(catalog), len(firstcalls), len(eligible), len(dev), len(test)) != (53, 212, 204, 48, 156):
        raise ValueError("API-Bank first-call counts differ from the audited preparation.")
    files = {
        "tools.json": _bytes([entry["tool"] for entry in catalog.values()]),
        "catalog_audit.json": _bytes(catalog), "all_firstcalls.jsonl": _jsonl(firstcalls),
        "examples.jsonl": _jsonl(eligible), "dev.jsonl": _jsonl(dev), "test.jsonl": _jsonl(test),
        "exclusions.jsonl": _jsonl(exclusions),
    }
    for relative in hashes["dialogues"]:
        files["raw/" + Path(relative).name] = (root / relative).read_bytes()
    for relative in hashes["api_sources"]:
        files["upstream/apis/" + Path(relative).name] = (root / relative).read_bytes()
    files["upstream/API-Bank-LICENSE"] = (root / "api-bank/LICENSE").read_bytes()
    files["upstream/REPOSITORY-LICENSE"] = (root / "LICENSE").read_bytes()
    manifest = {
        "format_version": "apibank-firstcall-v1", "benchmark": "API-Bank Level-1 first-call extraction (adapted)",
        "upstream_url": UPSTREAM_URL, "revision": REVISION, "source_files_sha256": hashes,
        "raw_dialogue_tree_sha256": RAW_SHA256, "api_source_tree_sha256": API_SHA256,
        "counts": {"dialogue_files": 214, "first_api_targets": 212, "no_api_files": 2,
                   "schema_excluded": 8, "scorable": 204, "dev": 48, "test": 156, "catalog_tools": 53},
        "task": {"target": "First API turn only", "tool_policy": "Fixed independent catalog of all 53 API classes",
                 "observations": "No external-tool observations; prefix ends before first API turn",
                 "score": "Exact API name and declared-type-normalized arguments, not official execution score",
                 "no_call_negatives": False, "labels": "ground_truth/raw_ground_truth only; never model input"},
        "normalization": {"required": "Non-default call signature arguments; conditional kwargs remain in descriptions",
                          "types": "str unchanged; int digit string; bool exact True/False; list safe literal parsing",
                          "excluded": "8 name/device_id conflicts retained in all_firstcalls/exclusions and raw files"},
        "split": {"unit": "Dialogue family with final -N removed", "salt": SPLIT_SALT,
                  "rule": "SHA-256 integer modulo 5: 0=dev; otherwise=test", "chosen_before_model_runs": True},
        "licenses": {"api_bank_path": "Apache-2.0", "repository_root": "MIT",
                     "files": ["upstream/API-Bank-LICENSE", "upstream/REPOSITORY-LICENSE"]},
        "preparer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "files_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())},
    }
    files["manifest.json"] = _bytes(manifest)
    # Repeated preparation is idempotent; incompatible existing data is never silently overwritten.
    for name, data in files.items():
        path = output / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Existing preparation differs: {path}; choose a new output directory.")
    for name, data in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/apibank-firstcall"))
    args = parser.parse_args(argv)
    manifest = prepare(args.upstream, args.output)
    print(json.dumps({"output": str(args.output), "counts": manifest["counts"], "model_requests": 0}, indent=2))


if __name__ == "__main__":
    main()
