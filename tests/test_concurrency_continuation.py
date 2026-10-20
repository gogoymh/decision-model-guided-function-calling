"""Concurrency continuation preserves observations and rejects unsafe checkpoints."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


continuation = load_module("concurrency_continuation", ROOT / "scripts/prepare_concurrency_continuation.py")
fixtures = load_module("additional_fixture", ROOT / "tests/test_additional_evaluation.py")
runner = continuation.runner


def checkpoint(tmp_path):
    events = []
    args = fixtures.args(tmp_path, output=tmp_path / "source", workers=8)
    generator, verifier = fixtures.Generator(events), fixtures.Verifier(events)
    cases = [fixtures.example()]
    settings = fixtures.config(args, cases, generator, verifier)
    runner.execute(args, cases, runner.prepare_apibank.score_example, settings, generator, verifier)
    # Stage a valid partial checkpoint with Direct complete and its shared initial cached.
    direct = runner.read_jsonl(args.output / "records.jsonl")[0]
    (args.output / "records.jsonl").write_text(json.dumps(direct) + "\n")
    (args.output / "errors.jsonl").write_text('{"error_type":"offline_fixture"}\n')
    return args, cases, settings


def test_only_worker_setting_and_explicit_audit_metadata_change(tmp_path, monkeypatch):
    args, cases, settings = checkpoint(tmp_path)
    monkeypatch.setattr(continuation, "_active_runner_pids", lambda source: [])
    original = continuation._snapshot(args.output)
    child = tmp_path / "workers32"
    provenance = continuation.prepare(args.output, child, 32, checkpointed=True)
    assert continuation._snapshot(args.output) == original
    old_manifest, new_manifest = (json.loads(path.read_text()) for path in
                                  (args.output / "manifest.json", child / "manifest.json"))
    new_settings = {key: value for key, value in new_manifest.items() if key not in {"created_at", "config_id"}}
    assert new_settings == {**settings, "workers": 32}
    assert new_manifest["config_id"] == runner._fingerprint(new_settings)
    assert new_manifest["config_id"] != old_manifest["config_id"]
    for name in ("records.jsonl", "initial.jsonl"):
        old_rows, new_rows = runner.read_jsonl(args.output / name), runner.read_jsonl(child / name)
        assert len(old_rows) == len(new_rows)
        for old, new in zip(old_rows, new_rows):
            assert new == {**old, "config_id": new_manifest["config_id"],
                           "original_config_id": old_manifest["config_id"], "recorded_workers": 8}
    assert (child / "errors.jsonl").read_bytes() == original["errors.jsonl"]
    assert (child / "provenance/original_manifest.json").read_bytes() == original["manifest.json"]
    assert provenance["child_manifest_sha256"] == continuation._digest((child / "manifest.json").read_bytes())
    assert provenance["source_files_sha256"] == {name: continuation._digest(data) if data is not None else None
                                                for name, data in original.items()}
    assert provenance["settings_changes"] == {"workers": {"before": 8, "after": 32}}
    config_id, completed, initials = runner.validate_resume(child, new_settings, True)
    assert config_id == new_manifest["config_id"] and completed == {("a", 0, "direct")} and len(initials) == 1
    # An offline resumed run reuses the original initial and generates only the missing repair.
    args.output, args.workers, args.resume = child, 32, True
    events = []
    generator, verifier = fixtures.Generator(events), fixtures.Verifier(events)
    summary = runner.execute(args, cases, runner.prepare_apibank.score_example, new_settings, generator, verifier)
    assert len(generator.calls) == 1 and generator.calls[0][1]
    assert summary["completed_records"] == 2
    child_rows = runner.read_jsonl(child / "records.jsonl")
    assert child_rows[0]["recorded_workers"] == 8 and "recorded_workers" not in child_rows[1]


def test_checkpoint_assertion_active_job_and_existing_output_rejected(tmp_path, monkeypatch):
    args, _, _ = checkpoint(tmp_path)
    child = tmp_path / "new"
    with pytest.raises(ValueError, match="checkpointed"):
        continuation.prepare(args.output, child)
    monkeypatch.setattr(continuation, "_active_runner_pids", lambda source: [123])
    with pytest.raises(ValueError, match="still active"):
        continuation.prepare(args.output, child, checkpointed=True)
    assert not child.exists()
    monkeypatch.setattr(continuation, "_active_runner_pids", lambda source: [])
    with pytest.raises(ValueError, match="separate"):
        continuation.prepare(args.output, args.output / "nested", checkpointed=True)
    child.mkdir()
    (child / "sentinel").write_text("unchanged")
    with pytest.raises(ValueError, match="new directory"):
        continuation.prepare(args.output, child, checkpointed=True)
    assert (child / "sentinel").read_text() == "unchanged"


@pytest.mark.parametrize("corrupt", ("duplicate_record", "changed_cache", "changed_fingerprint"))
def test_invalid_resume_grid_cache_and_config_rejected(tmp_path, monkeypatch, corrupt):
    args, _, _ = checkpoint(tmp_path)
    monkeypatch.setattr(continuation, "_active_runner_pids", lambda source: [])
    if corrupt == "duplicate_record":
        path = args.output / "records.jsonl"
        path.write_bytes(path.read_bytes() * 2)
    elif corrupt == "changed_cache":
        path = args.output / "initial.jsonl"
        row = runner.read_jsonl(path)[0]
        row["proposal"]["calls"][0]["arguments"]["name"] = "different"
        path.write_text(json.dumps(row) + "\n")
    else:
        path = args.output / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["temperature"] = .8
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        continuation.prepare(args.output, tmp_path / "bad", checkpointed=True)
    assert not (tmp_path / "bad").exists()


def test_checkpoint_change_during_validation_writes_nothing(tmp_path, monkeypatch):
    args, _, _ = checkpoint(tmp_path)
    monkeypatch.setattr(continuation, "_active_runner_pids", lambda source: [])
    snapshot = continuation._snapshot
    calls = 0

    def changing(source):
        nonlocal calls
        calls += 1
        result = snapshot(source)
        if calls == 2:
            result["initial.jsonl"] += b"\n"
        return result

    monkeypatch.setattr(continuation, "_snapshot", changing)
    with pytest.raises(ValueError, match="changed during validation"):
        continuation.prepare(args.output, tmp_path / "changed", checkpointed=True)
    assert not (tmp_path / "changed").exists()


def test_unfinished_shared_initial_is_preserved_without_a_completed_row(tmp_path, monkeypatch):
    args, _, settings = checkpoint(tmp_path)
    monkeypatch.setattr(continuation, "_active_runner_pids", lambda source: [])
    settings = {**settings, "ids": ["a", "pending"], "expected_records": 4}
    config_id = runner._fingerprint(settings)
    manifest = json.loads((args.output / "manifest.json").read_text())
    manifest.update(settings, config_id=config_id)
    (args.output / "manifest.json").write_text(json.dumps(manifest))
    for name in ("records.jsonl", "initial.jsonl"):
        rows = runner.read_jsonl(args.output / name)
        for row in rows:
            row["config_id"] = config_id
        if name == "initial.jsonl":
            rows.append({**rows[0], "id": "pending"})
        (args.output / name).write_bytes(continuation._jsonl_bytes(rows))
    child = tmp_path / "pending-continuation"
    continuation.prepare(args.output, child, checkpointed=True)
    _, completed, initials = runner.validate_resume(child, {**settings, "workers": 32}, True)
    assert completed == {("a", 0, "direct")}
    assert set(initials) == {("a", 0), ("pending", 0)}
    assert initials[("pending", 0)]["recorded_workers"] == 8


def test_local_active_process_guard_matches_absolute_relative_and_equal_output(tmp_path, monkeypatch):
    source = (ROOT / "runs/checkpoint-fixture").resolve()
    listing = (f"101 python scripts/run_additional_evaluation.py --run --output '{source}'\n"
               "102 python scripts/run_additional_evaluation.py --output runs/checkpoint-fixture --run\n"
               f"103 python scripts/run_additional_evaluation.py --output={source}\n"
               "104 python scripts/run_additional_evaluation.py --output elsewhere\n"
               f"105 python unrelated.py --output '{source}'\n")
    monkeypatch.setattr(continuation.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=listing))
    assert continuation._active_runner_pids(source) == [101, 102, 103]


def test_source_change_after_copy_rejects_and_removes_new_output(tmp_path, monkeypatch):
    args, _, _ = checkpoint(tmp_path)
    monkeypatch.setattr(continuation, "_active_runner_pids", lambda source: [])
    snapshot = continuation._snapshot
    calls = 0

    def changing(source):
        nonlocal calls
        calls += 1
        result = snapshot(source)
        if calls == 3:
            result["records.jsonl"] += b"\n"
        return result

    monkeypatch.setattr(continuation, "_snapshot", changing)
    child = tmp_path / "changed-after-copy"
    with pytest.raises(ValueError, match="changed during copying"):
        continuation.prepare(args.output, child, checkpointed=True)
    assert not child.exists()
