"""Offline BFCL scoring using the pinned upstream Python AST checker.

No benchmark function, generated source code, or external API is executed here.
The downloaded checker is trusted package code, SHA-256 verified before loading.
Its function bodies are unchanged; only imports are replaced to avoid importing
BFCL's many unrelated API/GPU/model dependencies for Python-only AST scoring.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from collections import Counter
from enum import Enum
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from .benchmarks import BFCL_REVISION, DEFAULT_CATEGORIES, OFFICIAL_CHECKER_SHA256


class _Language(Enum):
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"


@lru_cache(maxsize=4)
def load_official_checker(checker_path: str | Path = "data/bfcl/upstream/ast_checker.py") -> Callable[..., dict[str, Any]]:
    """Load byte-for-byte pinned upstream code with Python-only dependencies.

    After restoring aliases to original names, the BFCL model configuration's
    ``underscore_to_dot`` switch is False. Java/JavaScript are excluded from this
    benchmark and their converter imports are unreachable. Every AST checker
    function and constant remains exactly as upstream.
    """
    path = Path(checker_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing official BFCL checker: {path}. Run `python -m jev_fc.benchmarks` first.")
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != OFFICIAL_CHECKER_SHA256:
        raise ValueError(f"Refusing unrecognized BFCL checker (SHA-256 {digest})")
    tree = ast.parse(source, filename=str(path))
    allowed_imports = {
        "bfcl_eval.constants.enums", "bfcl_eval.constants.model_config",
        "bfcl_eval.constants.type_mappings",
        "bfcl_eval.eval_checker.ast_eval.type_convertor.java_type_converter",
        "bfcl_eval.eval_checker.ast_eval.type_convertor.js_type_converter",
    }
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module not in allowed_imports:
            raise ValueError(f"Unreviewed checker dependency: {node.module}")
    tree.body = [node for node in tree.body if not isinstance(node, ast.ImportFrom)]
    namespace: dict[str, Any] = {
        "Language": _Language,
        "MODEL_CONFIG_MAPPING": {"jev": SimpleNamespace(underscore_to_dot=False)},
    }
    exec(compile(tree, filename=str(path), mode="exec"), namespace)
    return namespace["ast_checker"]


def canonical_calls(example: dict[str, Any], calls: list[dict[str, Any]], *, allow_unknown_names: bool = False) -> list[dict[str, Any]]:
    """Parse structured API calls and restore aliases; never eval model text."""
    if not isinstance(calls, list):
        raise ValueError("Calls must be a list")
    name_map = example.get("name_map", {})
    parsed = []
    for item in calls:
        if not isinstance(item, dict):
            raise ValueError("Each call must be an object")
        call = item.get("function", item)
        name = call.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Each call must have a nonempty function name")
        if name_map and name not in name_map and not allow_unknown_names:
            raise ValueError(f"Function name is not among the offered aliases: {name}")
        arguments = call.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError("Function arguments must decode to a JSON object")
        # JSON is also a cheap guarantee that no non-JSON objects enter checker.
        json.dumps(arguments, allow_nan=False)
        parsed.append({name_map.get(name, name): copy.deepcopy(arguments)})
    return parsed


def score_example(
    example: dict[str, Any],
    calls: list[dict[str, Any]],
    checker_path: str | Path = "data/bfcl/upstream/ast_checker.py",
) -> dict[str, Any]:
    """Score a final proposal offline, keeping metric availability explicit.

    ``correct`` is official AST validity for labelled call tasks, no-call
    correctness for irrelevance, and call presence for live_relevance.
    ``function_correct`` is a separate exact function-multiset diagnostic.
    ``arguments_correct`` is evaluated only when the correct function is chosen,
    and is None when function identity is wrong or argument labels do not exist.
    Never send this result (which can reveal gold labels) to generation/verifiers.
    """
    category = example["category"]
    if category not in DEFAULT_CATEGORIES:
        raise ValueError(f"Unsupported BFCL category: {category}")
    if example.get("dataset_revision", BFCL_REVISION) != BFCL_REVISION:
        raise ValueError("Dataset/checker revision mismatch")
    result: dict[str, Any] = {
        "correct": False,
        "function_correct": None,
        "arguments_correct": None,
        "presence_correct": None,
        "metric": "bfcl_python_ast" if category not in {"irrelevance", "live_irrelevance", "live_relevance"} else "bfcl_call_presence",
        "checker_revision": BFCL_REVISION,
        "error": [],
    }
    try:
        decoded = canonical_calls(example, calls, allow_unknown_names=category in {"irrelevance", "live_irrelevance", "live_relevance"})
    except (TypeError, ValueError, AttributeError) as exc:
        result["function_correct"] = None if category == "live_relevance" else False
        if result["metric"] == "bfcl_call_presence":
            result["presence_correct"] = False
        result["error"] = [str(exc)]
        result["error_type"] = "invalid_structured_call"
        return result

    if category in {"irrelevance", "live_irrelevance"}:
        result.update(correct=not decoded, presence_correct=not decoded, function_correct=not decoded)
        return result
    if category == "live_relevance":
        result.update(correct=bool(decoded), presence_correct=bool(decoded))
        return result

    ground_truth = example["ground_truth"]
    expected_names = Counter(name for item in ground_truth for name in item)
    actual_names = Counter(name for item in decoded for name in item)
    function_correct = expected_names == actual_names
    checker = load_official_checker(str(Path(checker_path).resolve()))
    verdict = checker(
        copy.deepcopy(example["bfcl_functions"]), decoded, copy.deepcopy(ground_truth),
        _Language.PYTHON, category, "jev",
    )
    result.update(
        correct=bool(verdict["valid"]),
        function_correct=function_correct,
        arguments_correct=bool(verdict["valid"]) if function_correct else None,
        error=verdict.get("error", []),
    )
    if not verdict["valid"]:
        result["error_type"] = verdict.get("error_type", "unknown")
    return result
