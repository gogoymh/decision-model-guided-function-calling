"""Prepare an offline continuation with a changed worker count and preserved checkpoints.

Stop and checkpoint the source runner first, then pass --checkpointed. No API
requests are sent and no source files are written. Only the worker setting changes;
the original observations keep their recorded concurrency in audit metadata.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_additional_evaluation as runner  # noqa: E402

CHECKPOINT_FILES = ("manifest.json", "records.jsonl", "initial.jsonl", "errors.jsonl")


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def _rows(data):
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def _jsonl_bytes(rows):
    return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows).encode()


def _snapshot(source):
    result = {}
    for name in CHECKPOINT_FILES:
        path = source / name
        if path.is_symlink():
            raise ValueError("Checkpoint files must be local regular files, not symlinks.")
        if path.exists() and not path.is_file():
            raise ValueError("Checkpoint entry is not a regular file.")
        result[name] = path.read_bytes() if path.exists() else None
    if result["manifest.json"] is None:
        raise ValueError("Source has no additional-evaluation manifest.")
    return result


def _active_runner_pids(source):
    """Inspect local process arguments internally; never print commands or credentials."""
    output = subprocess.run(["ps", "-ww", "-axo", "pid=,command="], capture_output=True, text=True, check=True).stdout
    active = []
    for line in output.splitlines():
        pieces = line.strip().split(maxsplit=1)
        if len(pieces) != 2 or not any(name in pieces[1] for name in
                                      ("run_additional_evaluation.py", "run_claude_additional_evaluation.py")):
            continue
        try:
            command = shlex.split(pieces[1])
        except ValueError:
            # A command referring to this checkpoint but not safely parseable is also unsafe to copy.
            if str(source) in pieces[1]:
                active.append(int(pieces[0]))
            continue
        candidates = [command[index + 1] for index, part in enumerate(command[:-1]) if part == "--output"]
        candidates += [part.removeprefix("--output=") for part in command if part.startswith("--output=")]
        for candidate in candidates:
            path = Path(candidate)
            possible = [path.resolve()] if path.is_absolute() else [(ROOT / path).resolve(), path.resolve()]
            if source in possible:
                active.append(int(pieces[0]))
                break
    return sorted(set(active))


def _validate_original(source, snapshot):
    manifest = json.loads(snapshot["manifest.json"])
    settings = {key: value for key, value in manifest.items() if key not in {"config_id", "created_at"}}
    if (settings.get("format_version") != runner.FORMAT
            or settings.get("runner_sha256") != runner.sha256(Path(runner.__file__))
            or settings.get("source_sha256") != runner._source_hash()
            or settings.get("methods") != list(runner.METHODS)
            or settings.get("shared_initial_methods") != list(runner.METHODS)
            or manifest.get("config_id") != runner._fingerprint(settings)):
        raise ValueError("Source is not an unchanged, fingerprinted additional-evaluation checkpoint.")
    ids, replicates, workers = settings.get("ids"), settings.get("replicates"), settings.get("workers")
    if (not isinstance(ids, list) or not ids or len(ids) != len(set(ids))
            or any(not isinstance(identity, str) or not identity for identity in ids)
            or isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1
            or isinstance(workers, bool) or not isinstance(workers, int) or workers < 1
            or settings.get("expected_records") != len(ids) * replicates * len(runner.METHODS)):
        raise ValueError("Source ID, repetition, worker, or expected-record grid is inconsistent.")
    runner.validate_resume(source, settings, True)
    records = _rows(snapshot["records.jsonl"] or b"")
    initials = _rows(snapshot["initial.jsonl"] or b"")
    if any("original_config_id" in row or "recorded_workers" in row for row in records + initials):
        raise ValueError("Source already carries concurrency-continuation metadata; use its audited history explicitly.")
    for row in records:
        trace, final = row.get("trace"), row.get("final_attempt")
        if (not isinstance(trace, list) or not trace or isinstance(final, bool)
                or not isinstance(final, int) or not 0 <= final < len(trace)
                or row.get("calls") != trace[final].get("proposal", {}).get("calls")
                or row.get("score") != trace[final].get("score")
                or not isinstance(row.get("accounting"), dict)):
            raise ValueError("Source result has an inconsistent trace, final proposal, score, or accounting.")
        for name in ("logical_latency_s", "wall_time_s"):
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Source result contains invalid recorded timing.")
    return manifest, settings, records, initials


def _write_fsynced(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _provider_adapter(path):
    if path is None:
        return {}
    path = Path(path).resolve()
    expected = ROOT / "scripts/run_claude_additional_evaluation.py"
    if path != expected.resolve() or not path.is_file():
        raise ValueError("Only the local Claude additional-evaluation provider adapter is supported.")
    spec = importlib.util.spec_from_file_location("continuation_provider_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = module.adapter_settings()
    if (set(settings) != {"provider_adapter_sha256", "provider_input_rule"}
            or settings["provider_adapter_sha256"] != runner.sha256(path)
            or not isinstance(settings["provider_input_rule"], str)):
        raise ValueError("Provider adapter fingerprint or rule is invalid.")
    return settings


def prepare(source, output, workers=32, *, checkpointed=False, provider_input_adapter=None):
    source, output = Path(source).resolve(), Path(output).absolute()
    if not checkpointed:
        raise ValueError("Stop and checkpoint the source job first; explicit --checkpointed is required.")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("Continuation workers must be a positive integer.")
    if output.exists() or output.is_symlink():
        raise ValueError("Continuation output must be a new directory; existing output is never overwritten.")
    if source == output.resolve() or source in output.resolve().parents or output.resolve() in source.parents:
        raise ValueError("Continuation output must be separate from the source checkpoint.")
    if _active_runner_pids(source):
        raise ValueError("The source runner is still active; stop and checkpoint it before continuation preparation.")
    original = _snapshot(source)
    manifest, settings, records, initials = _validate_original(source, original)
    provider_settings = _provider_adapter(provider_input_adapter)
    if provider_settings and (settings.get("benchmark") != "bfcl"
                              or settings.get("llm_provider") != "anthropic"
                              or settings.get("llm_api_style") != "anthropic_messages"
                              or not settings.get("llm_model", "").startswith("claude")):
        raise ValueError("Provider adapter continuation requires a Claude BFCL source run.")
    if workers == settings["workers"]:
        raise ValueError("Continuation must specify a changed worker count.")
    amended_settings = {**settings, "workers": workers, **provider_settings}
    original_id, new_id = manifest["config_id"], runner._fingerprint(amended_settings)
    created_at = datetime.now(timezone.utc).isoformat()
    child_manifest = {**amended_settings, "config_id": new_id, "created_at": created_at}
    child_manifest_bytes = _json_bytes(child_manifest)

    def amend(row):
        return {**deepcopy(row), "config_id": new_id, "original_config_id": original_id,
                "recorded_workers": settings["workers"]}

    files = {"manifest.json": child_manifest_bytes,
             "records.jsonl": _jsonl_bytes(amend(row) for row in records),
             "initial.jsonl": _jsonl_bytes(amend(row) for row in initials),
             "provenance/original_manifest.json": original["manifest.json"]}
    if original["errors.jsonl"] is not None:
        files["errors.jsonl"] = original["errors.jsonl"]
    provenance = {
        "format_version": "additional-concurrency-continuation-v1", "created_at": created_at,
        "script_sha256": runner.sha256(Path(__file__)), "source_directory": str(source),
        "original_config_id": original_id, "child_config_id": new_id,
        "source_files_sha256": {name: _digest(data) if data is not None else None for name, data in original.items()},
        "child_manifest_sha256": _digest(child_manifest_bytes),
        "settings_changes": {name: {"before": settings.get(name), "after": amended_settings[name]}
                             for name in ("workers", *provider_settings)},
        "record_amendments": ["config_id", "original_config_id", "recorded_workers"],
        "counts": {"completed_method_records": len(records), "cached_initials": len(initials),
                   "expected_method_records": settings["expected_records"],
                   "remaining_method_records": settings["expected_records"] - len(records)},
        "runtime_history": [
            {"phase": "preserved_checkpoint", "workers": settings["workers"],
             "records": len(records), "cached_initials": len(initials)},
            {"phase": "continuation", "workers": workers,
             "record_rule": "Rows without recorded_workers use child manifest workers; cached initials keep their original concurrency."},
        ],
        "timing_note": "Physical serving concurrency changes; cached initial and completed response times are preserved. Latency comparisons must disclose the mixed concurrency.",
        "checkpointed_assertion": True, "source_process_check": "No matching local source runner active",
    }
    files["provenance/concurrency_continuation.json"] = _json_bytes(provenance)
    if _active_runner_pids(source) or _snapshot(source) != original:
        raise ValueError("Source checkpoint changed during validation; no continuation was written.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    try:
        for name, data in files.items():
            _write_fsynced(output / name, data)
        runner.validate_resume(output, amended_settings, True)
        if _active_runner_pids(source) or _snapshot(source) != original:
            raise ValueError("Source checkpoint changed during copying; continuation rejected.")
        descriptor = os.open(output, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        shutil.rmtree(output)
        raise
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--checkpointed", action="store_true")
    parser.add_argument("--provider-input-adapter", type=Path,
                        help="Optional local run_claude_additional_evaluation.py adapter; adds its audited rule/hash")
    args = parser.parse_args(argv)
    provenance = prepare(args.source, args.output, args.workers, checkpointed=args.checkpointed,
                         provider_input_adapter=args.provider_input_adapter)
    print(json.dumps({"output": str(args.output), "child_config_id": provenance["child_config_id"],
                      "counts": provenance["counts"], "settings_changes": provenance["settings_changes"],
                      "network_requests": 0}, indent=2))
    return provenance


if __name__ == "__main__":
    main()
