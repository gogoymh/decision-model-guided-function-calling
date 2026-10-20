"""Offline, common native-call-candidate input adapter for the audited API-Bank data.

The original prepared data and protocol-pilot runs remain untouched. Every model
receives the same appended instruction; labels, schemas, split, and original
conversation order are preserved. The output must be a new directory. This
compatibility amendment followed inspection of format-pilot test outcomes;
all prior inputs and exploratory results must be retained and disclosed.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_apibank  # noqa: E402

INSTRUCTION = (
    "Produce the next function-call candidate for the preceding conversation using the native structured "
    "tool-call interface. A tool-call candidate is only a proposal: the evaluator will not execute functions. "
    "Use only documented parameters and values supported by the conversation."
)
EXAMPLE_FILES = ("examples.jsonl", "dev.jsonl", "test.jsonl", "all_firstcalls.jsonl")
PROTOCOL_REASON = (
    "The original API-Bank input caused Claude to reject assistant prefill (HTTP 400). The first "
    "common user-turn format pilot was then inspected on the test split: GPT mini returned textual "
    "descriptions of the next API call with empty native call lists in all 468 Direct records. This "
    "second compatibility amendment explicitly requests a native structured call candidate. Test "
    "outcomes had already been inspected; earlier inputs and all three-model format-pilot results "
    "are retained as exploratory, and this amendment is not a fresh untouched-test evaluation."
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _rows(data):
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def _adapt(data):
    rows = _rows(data)
    for row in rows:
        messages = row.get("messages")
        if (not isinstance(messages, list) or not messages
                or any(not isinstance(message, dict) or message.get("role") not in {"user", "assistant", "system"}
                       or not isinstance(message.get("content"), str) for message in messages)):
            raise ValueError("Prepared input must contain nonempty text conversation messages.")
        messages.append({"role": "user", "content": INSTRUCTION})
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                   for row in rows).encode()


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Input adapter output must be a new directory; existing output is never overwritten.")
    if source == output.resolve() or source in output.resolve().parents:
        raise ValueError("Input adapter output must be outside the original prepared directory.")
    original_manifest = (source / "manifest.json").read_bytes()
    manifest = json.loads(original_manifest)
    if (manifest.get("format_version") != "apibank-firstcall-v1"
            or manifest.get("revision") != prepare_apibank.REVISION
            or manifest.get("preparer_sha256") != sha256(Path(prepare_apibank.__file__).read_bytes())
            or "input_adapter" in manifest):
        raise ValueError("Source must be the original audited v1 preparation using the current preparer.")
    hashes = manifest.get("files_sha256")
    if not isinstance(hashes, dict) or not set(EXAMPLE_FILES + ("tools.json", "catalog_audit.json")).issubset(hashes):
        raise ValueError("Source preparation manifest is missing required data files.")
    files = {}
    for name, expected in hashes.items():
        relative = PurePosixPath(name)
        if (not isinstance(name, str) or relative.is_absolute() or ".." in relative.parts
                or name == "manifest.json" or not relative.parts):
            raise ValueError("Source manifest contains an unsafe file path.")
        path = source / name
        if not path.is_file() or path.is_symlink() or source not in path.resolve().parents:
            raise ValueError("Source preparation file is missing or escapes its directory.")
        data = path.read_bytes()
        if sha256(data) != expected:
            raise ValueError("Source preparation checksum mismatch: " + name)
        files[name] = data
    counts = manifest.get("counts", {})
    catalog = json.loads(files["tools.json"])
    examples, dev, test, firstcalls = (_rows(files[name]) for name in EXAMPLE_FILES)
    if (len(catalog) != counts.get("catalog_tools") or len(examples) != counts.get("scorable")
            or len(dev) != counts.get("dev") or len(test) != counts.get("test")
            or len(firstcalls) != counts.get("first_api_targets")):
        raise ValueError("Source preparation counts differ from its audited manifest.")
    if (dev != [row for row in examples if row["split"] == "dev"]
            or test != [row for row in examples if row["split"] == "test"]
            or examples != [row for row in firstcalls if row["scorable"]]
            or any(row["tools"] != catalog for row in firstcalls)):
        raise ValueError("Source examples, split, or independent tool catalog are inconsistent.")
    for name in EXAMPLE_FILES:
        files[name] = _adapt(files[name])
    files["provenance/source_manifest.json"] = original_manifest
    adapted_manifest = deepcopy(manifest)
    adapted_manifest["input_adapter"] = {
        "format_version": "apibank-native-candidate-v1",
        "script_sha256": sha256(Path(__file__).read_bytes()),
        "instruction": INSTRUCTION,
        "applies_to": "Every prepared example, every model, Direct and DMFC",
        "original_messages_preserved_in_order": True,
        "source_data_path": str(source),
        "source_manifest_sha256": sha256(original_manifest),
        "source_manifest_copy": "provenance/source_manifest.json",
        "protocol_reason": PROTOCOL_REASON,
        "amended_after_test_outcomes_inspected": True,
        "prior_runs_status": "Original and common-user-turn runs retained as exploratory protocol/format pilots",
        "validation_plan": "Fixed common development cases first, then one fixed full test protocol for all three generators",
    }
    adapted_manifest["files_sha256"] = {name: sha256(data) for name, data in sorted(files.items())}
    files["manifest.json"] = _json_bytes(adapted_manifest)
    # All source checks finish before writing. Creation is exclusive even if another process races.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    for name, data in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return adapted_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = prepare(args.source, args.output)
    print(json.dumps({"output": str(args.output), "counts": manifest["counts"],
                      "input_adapter": manifest["input_adapter"], "model_requests": 0}, indent=2))
    return manifest


if __name__ == "__main__":
    main()
