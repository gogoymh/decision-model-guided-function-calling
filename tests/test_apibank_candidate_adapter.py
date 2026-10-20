"""The second compatibility amendment preserves data and discloses test-outcome inspection."""

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


adapter = load_module("candidate_adapter", ROOT / "scripts/prepare_apibank_candidate_adapter.py")
fixtures = load_module("input_fixture", ROOT / "tests/test_apibank_input_adapter.py")
runner = load_module("candidate_frozen_runner", ROOT / "scripts/run_additional_evaluation.py")


def test_candidate_instruction_preserves_all_data_and_has_own_provenance(tmp_path):
    source, original = fixtures.fixture_source(tmp_path)
    first_pilot = tmp_path / "first-format-pilot"
    fixtures.adapter.prepare(source, first_pilot)
    pilot_files = {path.relative_to(first_pilot): path.read_bytes() for path in first_pilot.rglob('*') if path.is_file()}
    output = tmp_path / "candidate"
    manifest = adapter.prepare(source, output)
    for name in adapter.EXAMPLE_FILES:
        before, after = adapter._rows(original[name]), adapter._rows((output / name).read_bytes())
        assert len(before) == len(after)
        for old, new in zip(before, after):
            assert new == {**old, 'messages': old['messages'] + [{'role': 'user', 'content': adapter.INSTRUCTION}]}
            assert sum(message['content'] == adapter.INSTRUCTION for message in new['messages']) == 1
            assert all(message['content'] != fixtures.adapter.INSTRUCTION for message in new['messages'])
    for name, data in original.items():
        assert (source / name).read_bytes() == data
        if name not in adapter.EXAMPLE_FILES and name != 'manifest.json':
            assert (output / name).read_bytes() == data
    assert {path.relative_to(first_pilot): path.read_bytes() for path in first_pilot.rglob('*') if path.is_file()} == pilot_files
    provenance = manifest['input_adapter']
    assert provenance['format_version'] == 'apibank-native-candidate-v1'
    assert provenance['script_sha256'] == adapter.sha256(Path(adapter.__file__).read_bytes())
    assert provenance['script_sha256'] != fixtures.adapter.sha256(Path(fixtures.adapter.__file__).read_bytes())
    assert provenance['amended_after_test_outcomes_inspected'] is True
    assert 'exploratory' in provenance['prior_runs_status']
    assert 'not a fresh untouched-test evaluation' in provenance['protocol_reason']
    assert provenance['source_manifest_sha256'] == adapter.sha256(original['manifest.json'])
    assert manifest['counts'] == json.loads(original['manifest.json'])['counts']
    for name, digest in manifest['files_sha256'].items():
        assert adapter.sha256((output / name).read_bytes()) == digest
    selected, _, _ = runner.load_data(SimpleNamespace(data=output, benchmark='apibank', split='test', limit=None))
    assert len(selected) == 1 and selected[0]['messages'][-1]['role'] == 'user'
    # Native tool selection remains automatic; the adapter changes only messages, not forcing a call.
    for style, expected in (('responses', 'auto'), ('anthropic_messages', {'type': 'auto', 'disable_parallel_tool_use': True})):
        client = SimpleNamespace(model='offline-model', api_style=style, reasoning_effort=None,
                                 temperature=None, max_completion_tokens=4096)
        payload = runner.FunctionCallingLLM._payload(client, runner.public_input(selected[0]), [])
        assert payload['tool_choice'] == expected


def test_original_only_new_directory_and_bad_hash_rejected(tmp_path):
    source, _ = fixtures.fixture_source(tmp_path)
    adapted = tmp_path / 'already-adapted'
    fixtures.adapter.prepare(source, adapted)
    with pytest.raises(ValueError, match='original audited'):
        adapter.prepare(adapted, tmp_path / 'rejected')
    assert not (tmp_path / 'rejected').exists()
    output = tmp_path / 'existing'
    output.mkdir()
    (output / 'sentinel').write_text('unchanged')
    with pytest.raises(ValueError, match='new directory'):
        adapter.prepare(source, output)
    assert (output / 'sentinel').read_text() == 'unchanged'
    (source / 'tools.json').write_text('tampered')
    with pytest.raises(ValueError, match='checksum'):
        adapter.prepare(source, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()
