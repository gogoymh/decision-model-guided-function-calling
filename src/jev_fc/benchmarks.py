"""Pinned BFCL v4 single-turn data, API schemas, and leakage-safe examples.

This module uses only the Python standard library. Ground truth is for offline
evaluation; pass ``public_example(example)`` to model/verifier clients.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import urllib.request
from pathlib import Path
from typing import Any, Iterable

BFCL_REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BFCL_REPOSITORY = "https://github.com/ShishirPatil/gorilla"
BFCL_PREFIX = "berkeley-function-call-leaderboard/bfcl_eval"
BFCL_RAW = f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{BFCL_REVISION}"
OFFICIAL_CHECKER_SHA256 = "2aae7a68461a8f76c0be3894c8901b66b56967a1989d3ab066051e3fb97f1538"
DEFAULT_CATEGORIES = (
    "simple_python", "multiple", "irrelevance", "live_simple",
    "live_multiple", "live_irrelevance", "live_relevance",
)
PRESENCE_CATEGORIES = frozenset({"irrelevance", "live_irrelevance", "live_relevance"})
EXPECTED_COUNTS = dict(zip(DEFAULT_CATEGORIES, (400, 200, 240, 258, 1053, 884, 16)))
KNOWN_ANNOTATION_ISSUES = {
    "live_simple_106-63-0": "Required array arguments have empty alternative sets [] instead of an accepted empty-list value [[]]",
    "live_simple_112-68-0": "Required array arguments have empty alternative sets [] instead of an accepted empty-list value [[]]",
    "live_multiple_862-181-3": "Ground-truth journey_start_time is absent from the selected function's parameter schema",
    "live_multiple_964-207-0": "Ground-truth question is absent from the selected function's parameter schema",
}
# Reference values whose JSON type contradicts the declared parameter type. The
# pinned checker accepts only the reference type, so a call that satisfies the
# declared types cannot be scored correct, and the reference call fails the local
# type check. They stay in every denominator.
SCHEMA_TYPE_CONFLICTS = {
    "live_multiple_552-153-1": "Reference is_unisex is boolean true; the parameter is a string ('True', 'False', 'dontcare')",
    "live_multiple_595-158-1": "Reference number_of_rooms is integer 2; the parameter is a string ('1', '2', '3', 'dontcare')",
    "live_multiple_596-158-2": "Reference number_of_rooms is integer 1; the parameter is a string ('1', '2', '3', 'dontcare')",
    "live_multiple_731-167-2": "Reference in_unit_laundry is boolean true; the parameter is a string ('True', 'False', 'dontcare')",
    "live_multiple_733-167-4": "Reference in_unit_laundry is boolean true; the parameter is a string ('True', 'False', 'dontcare')",
    "live_multiple_735-167-6": "Reference in_unit_laundry is boolean true; the parameter is a string ('True', 'False', 'dontcare')",
}

# Hashes were measured from the official repository at BFCL_REVISION. Updates
# deliberately require review of schemas/checker behavior and a new pin.
QUESTION_HASHES = {
    "simple_python": "82dd63ba502eb2520c6b5d1d9a5c4b590e03ff261565175561f6228a367d1991",
    "multiple": "aef168155ebd74b7ac2401198b201343bc7d16d7a3d7e0d4e6d8ee82c6969b2a",
    "irrelevance": "2b6ed4c2e992cdcf5f1678a701851f944bef7550ee026ed1ddb89efed5be01a6",
    "live_simple": "1af2ac87dca47556db7b7e37e51e28b459a38b594e3c7b3c792b4903598ca0c4",
    "live_multiple": "fd8ccfad4d911420d0e3341dbe2fff77d1d341da934248b9bb2bda24ab3a10c8",
    "live_irrelevance": "6559fda2beaceb609a2cd2e504c65b4a56cb448e1ef88fddfd199e163d163349",
    "live_relevance": "e03f9e241657a137cba48a89ee12f47bf3fcb7e4f6274263e9c699a0c974203a",
}
ANSWER_HASHES = {
    "simple_python": "90cd5bc653690ee8e459b5b3f3fc9458606f7f3fcbf795bb51b7dc581f8c86dc",
    "multiple": "244e00ce9395df948bcafc7bee64e8f9c87ef70887587d83cae45b13699f3047",
    "live_simple": "fec9cfa9744a936f9126981e85a2023da1e63e273eafebc81923a1162fad70ce",
    "live_multiple": "97e90d59c5bd76c55a2920ce93e5566e9046307d3f558578f085f9d3a56c3084",
}
LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
SCHEMA_TYPES = {
    "dict": "object", "float": "number", "tuple": "array", "any": "string",
    "object": "object", "number": "number", "integer": "integer",
    "string": "string", "boolean": "boolean", "array": "array", "null": "null",
}


def json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert the BFCL Python schema dialect to JSON Schema recursively.

    ``any -> string`` matches BFCL's official GORILLA_TO_OPENAPI mapping.
    Defaults and prose are preserved; the BFCL-only ``optional`` annotation is
    removed because requiredness is already specified by the parent's array.
    A property *named* ``type`` is not confused with a schema's ``type`` keyword.
    """
    result = copy.deepcopy(schema)
    if "type" in result:
        value = result["type"]
        if not isinstance(value, str) or value not in SCHEMA_TYPES:
            raise ValueError(f"Unsupported BFCL Python schema type: {value!r}")
        result["type"] = SCHEMA_TYPES[value]
    if "properties" in result:
        result["properties"] = {key: json_schema(value) for key, value in result["properties"].items()}
    if isinstance(result.get("items"), dict):
        result["items"] = json_schema(result["items"])
    if isinstance(result.get("additionalProperties"), dict):
        result["additionalProperties"] = json_schema(result["additionalProperties"])
    for key in ("oneOf", "anyOf", "allOf"):
        if key in result:
            result[key] = [json_schema(item) for item in result[key]]
    result.pop("optional", None)
    return result


def _function_alias(name: str, used: set[str]) -> str:
    alias = re.sub(r"[^A-Za-z0-9_-]", "_", name) or "function"
    if len(alias) > 64 or alias in used:
        alias = f"{alias[:51]}_{hashlib.sha256(name.encode()).hexdigest()[:12]}"
    if alias in used:
        raise ValueError(f"Duplicate or colliding BFCL function name: {name!r}")
    return alias


def normalize_example(
    question: dict[str, Any],
    category: str,
    answer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if category not in DEFAULT_CATEGORIES:
        raise ValueError(f"Unsupported category {category!r}; only the audited single-turn subset is supported")
    turns = question["question"]
    if len(turns) != 1 or not isinstance(turns[0], list):
        raise ValueError(f"Expected one user turn for {question['id']}")
    if category not in PRESENCE_CATEGORIES:
        if answer is None or answer["id"] != question["id"]:
            raise ValueError(f"Missing or mismatched ground-truth ID for {question['id']}")
        ground_truth = copy.deepcopy(answer["ground_truth"])
    elif category == "live_relevance":
        ground_truth = None  # No argument/function-identity labels exist here.
    else:
        ground_truth = []  # Official label is absence of a call, not an answer file.

    tools, aliases = [], {}
    original_names: set[str] = set()
    for function in question["function"]:
        name = function["name"]
        if name in original_names:
            raise ValueError(f"Duplicate function name in {question['id']}: {name}")
        original_names.add(name)
        alias = _function_alias(name, set(aliases))
        aliases[alias] = name
        description = function.get("description", "")
        if "response" in function:
            description += " Return schema: " + json.dumps(function["response"], ensure_ascii=False)
        tools.append({
            "type": "function",
            "function": {
                "name": alias,
                "description": description,
                "parameters": json_schema(function["parameters"]),
            },
        })
    return {
        "id": question["id"],
        "category": category,
        "messages": copy.deepcopy(turns[0]),
        "tools": tools,
        "name_map": aliases,
        "bfcl_functions": copy.deepcopy(question["function"]),
        "ground_truth": ground_truth,
        "dataset_revision": BFCL_REVISION,
    }


def public_example(example: dict[str, Any]) -> dict[str, Any]:
    """Return the only fields allowed in generator/verifier inputs.

    Metadata such as IDs/categories may be retained for routing/logging; prompt
    constructors should serialize only messages and tools, since category names
    (e.g. irrelevance) themselves reveal the expected decision.
    """
    return copy.deepcopy({key: example[key] for key in ("id", "category", "messages", "tools") if key in example})


def stratified_split(
    examples: Iterable[dict[str, Any]], *, dev_fraction: float = 0.1, seed: int = 1729,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stable hash-ranked split, with floor counts and a minimum of one.

    Seed/ID determine membership independently of input order. All rows
    are copied, so no split metadata can mutate a caller's examples.
    """
    if not 0 <= dev_fraction < 1:
        raise ValueError("dev_fraction must be in [0, 1)")
    rows = list(examples)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate example IDs")
    categories = sorted({row["category"] for row in rows})
    dev_ids: set[str] = set()
    for category in categories:
        pool = [row for row in rows if row["category"] == category]
        pool.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest())
        count = max(1, math.floor(len(pool) * dev_fraction)) if dev_fraction else 0
        dev_ids.update(row["id"] for row in pool[:count])
    dev, test = [], []
    for original in rows:
        row = copy.deepcopy(original)
        row["split"] = "dev" if row["id"] in dev_ids else "test"
        (dev if row["split"] == "dev" else test).append(row)
    return dev, test


def _jsonl(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def load_examples(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        path = path / "examples.jsonl"
    data = path.read_bytes()
    manifest_path = path.parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_digest = manifest.get("normalized_sha256", {}).get(path.name)
        if expected_digest and hashlib.sha256(data).hexdigest() != expected_digest:
            raise ValueError(f"Prepared benchmark checksum mismatch for {path}; regenerate data instead of silently scoring edited labels")
    rows = _jsonl(data)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate IDs in {path}")
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _fetch_verified(url: str, destination: Path, expected_sha256: str) -> dict[str, Any]:
    if destination.exists():
        data = destination.read_bytes()
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "jev-backed-function-calling/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {destination.name}; expected {expected_sha256}, got {digest}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return {"url": url, "sha256": digest, "bytes": len(data)}


def prepare_bfcl(
    output_dir: str | Path = "data/bfcl",
    revision: str = BFCL_REVISION,
    categories: Iterable[str] = DEFAULT_CATEGORIES,
    *,
    dev_fraction: float = 0.1,
    seed: int = 1729,
) -> dict[str, Any]:
    """Download, verify, normalize, and partition the audited BFCL v4 subset.

    Cached files are rehashed, so a repeated run works offline. Unknown revisions
    are rejected rather than silently changing the experimental benchmark.
    """
    if revision != BFCL_REVISION:
        raise ValueError(f"Only audited revision {BFCL_REVISION} is supported; review and update pins before changing versions")
    selected = tuple(categories)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(DEFAULT_CATEGORIES):
        raise ValueError(f"Choose distinct categories from {DEFAULT_CATEGORIES}")
    output = Path(output_dir)
    sources, examples, counts = {}, [], {}
    for category in selected:
        relative = f"raw/BFCL_v4_{category}.json"
        sources[relative] = _fetch_verified(
            f"{BFCL_RAW}/{BFCL_PREFIX}/data/BFCL_v4_{category}.json",
            output / relative, QUESTION_HASHES[category],
        )
        questions = _jsonl((output / relative).read_bytes())
        if len(questions) != EXPECTED_COUNTS[category]:
            raise ValueError(f"Unexpected number of examples in {category}")
        answers = None
        if category not in PRESENCE_CATEGORIES:
            relative_answer = f"raw/possible_answer/BFCL_v4_{category}.json"
            sources[relative_answer] = _fetch_verified(
                f"{BFCL_RAW}/{BFCL_PREFIX}/data/possible_answer/BFCL_v4_{category}.json",
                output / relative_answer, ANSWER_HASHES[category],
            )
            answers = _jsonl((output / relative_answer).read_bytes())
            if len(answers) != len(questions):
                raise ValueError(f"Mismatched question/answer count for {category}")
        examples.extend(normalize_example(row, category, answers[i] if answers else None) for i, row in enumerate(questions))
        counts[category] = len(questions)

    sources["upstream/ast_checker.py"] = _fetch_verified(
        f"{BFCL_RAW}/{BFCL_PREFIX}/eval_checker/ast_eval/ast_checker.py",
        output / "upstream/ast_checker.py", OFFICIAL_CHECKER_SHA256,
    )
    sources["upstream/LICENSE"] = _fetch_verified(f"{BFCL_RAW}/LICENSE", output / "upstream/LICENSE", LICENSE_SHA256)
    dev, test = stratified_split(examples, dev_fraction=dev_fraction, seed=seed)
    files = {
        "examples.jsonl": _write_jsonl(output / "examples.jsonl", examples),
        "dev.jsonl": _write_jsonl(output / "dev.jsonl", dev),
        "test.jsonl": _write_jsonl(output / "test.jsonl", test),
    }
    manifest = {
        "benchmark": "BFCL v4 audited single-turn Python subset",
        "repository": BFCL_REPOSITORY,
        "revision": revision,
        "license": "Apache-2.0",
        "counts": counts,
        "total": len(examples),
        "split": {
            "method": "sha256(seed:id) rank within category; floor 10% with minimum one development case",
            "seed": seed, "dev_fraction": dev_fraction,
            "dev_count": len(dev), "test_count": len(test),
            "dev_by_category": {cat: sum(row["category"] == cat for row in dev) for cat in selected},
            "test_by_category": {cat: sum(row["category"] == cat for row in test) for cat in selected},
        },
        "sources": sources,
        "normalized_sha256": files,
        "score_scope": "Official BFCL Python AST checker and separate call-presence metrics; not the complete BFCL leaderboard aggregate",
        "normalization": "dict->object; float->number; tuple->array; any->string; API-safe function aliases; original functions and answers retained only for scoring",
        "known_annotation_issues": KNOWN_ANNOTATION_ISSUES,
        "schema_type_conflicts": SCHEMA_TYPE_CONFLICTS,
        "annotation_policy": "Retain all original data and denominators; any exclusion sensitivity must be declared separately",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/bfcl")
    parser.add_argument("--revision", default=BFCL_REVISION)
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES), choices=DEFAULT_CATEGORIES)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    manifest = prepare_bfcl(args.output_dir, args.revision, args.categories, dev_fraction=args.dev_fraction, seed=args.seed)
    print(json.dumps({key: manifest[key] for key in ("revision", "total", "counts", "split")}, indent=2))


if __name__ == "__main__":
    main()
