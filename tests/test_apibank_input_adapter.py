"""The common API-Bank adapter only appends a neutral user turn in a new directory."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("input_adapter", ROOT / "scripts/prepare_apibank_input_adapter.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def fixture_source(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    tools = [{"type": "function", "function": {"name": "Book", "description": "Book a room.",
              "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}}]
    cases = [{"id": str(index), "category": "apibank_firstcall", "split": split,
              "scorable": True, "dataset_revision": adapter.prepare_apibank.REVISION,
              "messages": [{"role": "system", "content": "Conversation data."},
                           {"role": "user", "content": "Book a room."},
                           {"role": role, "content": "Original final turn."}],
              "ground_truth": [{"name": "Book", "arguments": {}}],
              "raw_ground_truth": {"api_name": "Book", "param_dict": {}}, "tools": deepcopy(tools)}
             for index, (split, role) in enumerate((('dev', 'assistant'), ('test', 'user')))]
    files = {"tools.json": adapter._json_bytes(tools), "catalog_audit.json": b"{}\n",
             "upstream/API-Bank-LICENSE": b"license stays identical\n", "raw/original.jsonl": b"raw unchanged\n"}
    for name, selected in (("examples.jsonl", cases), ("all_firstcalls.jsonl", cases),
                           ("dev.jsonl", cases[:1]), ("test.jsonl", cases[1:])):
        files[name] = "".join(json.dumps(case) + "\n" for case in selected).encode()
    manifest = {"format_version": "apibank-firstcall-v1", "revision": adapter.prepare_apibank.REVISION,
                "preparer_sha256": adapter.sha256(Path(adapter.prepare_apibank.__file__).read_bytes()),
                "counts": {"catalog_tools": 1, "scorable": 2, "dev": 1, "test": 1, "first_api_targets": 2},
                "split": {"unit": "fixed family"}, "source_files_sha256": {"fixed": "upstream audit"},
                "files_sha256": {name: adapter.sha256(data) for name, data in files.items()}}
    files["manifest.json"] = adapter._json_bytes(manifest)
    for name, data in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return source, files


def test_common_instruction_preserves_all_original_inputs_and_audit(tmp_path):
    source, original_files = fixture_source(tmp_path)
    output = tmp_path / "adapted"
    manifest = adapter.prepare(source, output)
    for name, original in original_files.items():
        assert (source / name).read_bytes() == original
        if name in adapter.EXAMPLE_FILES:
            before, after = adapter._rows(original), adapter._rows((output / name).read_bytes())
            assert len(before) == len(after)
            for old, new in zip(before, after):
                assert new["messages"][:-1] == old["messages"]
                assert new["messages"][-1] == {"role": "user", "content": adapter.INSTRUCTION}
                assert {key: value for key, value in new.items() if key != "messages"} == {
                    key: value for key, value in old.items() if key != "messages"}
        elif name != "manifest.json":
            assert (output / name).read_bytes() == original
    assert (output / "provenance/source_manifest.json").read_bytes() == original_files["manifest.json"]
    for name, expected in manifest["files_sha256"].items():
        assert adapter.sha256((output / name).read_bytes()) == expected
    assert manifest["counts"] == json.loads(original_files["manifest.json"])["counts"]
    assert manifest["input_adapter"]["source_manifest_sha256"] == adapter.sha256(original_files["manifest.json"])
    assert manifest["input_adapter"]["script_sha256"] == adapter.sha256(Path(adapter.__file__).read_bytes())
    assert "Book" not in adapter.INSTRUCTION and "ORACLE" not in adapter.INSTRUCTION
    # The frozen runner must accept this distinct manifest and dataset without any changes.
    runner_spec = importlib.util.spec_from_file_location("frozen_runner", ROOT / "scripts/run_additional_evaluation.py")
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    selected, _, provenance = runner.load_data(SimpleNamespace(data=output, benchmark="apibank", split="test", limit=None))
    assert len(selected) == 1 and selected[0]["messages"][-1]["role"] == "user"
    assert provenance["dataset_sha256"] == adapter.sha256((output / "examples.jsonl").read_bytes())


def test_new_directory_only_and_checksum_failure_writes_nothing(tmp_path):
    source, original_files = fixture_source(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "sentinel").write_text("preserve")
    with pytest.raises(ValueError, match="new directory"):
        adapter.prepare(source, existing)
    assert (existing / "sentinel").read_text() == "preserve"
    with pytest.raises(ValueError, match="outside"):
        adapter.prepare(source, source / "nested")
    assert not (source / "nested").exists()
    (source / "examples.jsonl").write_text("tampered")
    output = tmp_path / "bad"
    with pytest.raises(ValueError, match="checksum"):
        adapter.prepare(source, output)
    assert not output.exists()
    assert (source / "test.jsonl").read_bytes() == original_files["test.jsonl"]


def test_rejects_changed_preparer_or_already_adapted_source(tmp_path):
    source, original_files = fixture_source(tmp_path)
    original = json.loads(original_files["manifest.json"])
    for change in ({"preparer_sha256": "changed"}, {"input_adapter": {"already": True}}):
        (source / "manifest.json").write_bytes(adapter._json_bytes({**original, **change}))
        with pytest.raises(ValueError, match="original audited"):
            adapter.prepare(source, tmp_path / "output")
        assert not (tmp_path / "output").exists()
