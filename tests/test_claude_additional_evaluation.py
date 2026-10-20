"""Claude schema incompatibility is an unreadable prediction, never a removed case."""

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


adapter = load_module("claude_adapter", ROOT / "scripts/run_claude_additional_evaluation.py")
continuation = load_module("claude_continuation", ROOT / "scripts/prepare_concurrency_continuation.py")
fixtures = load_module("claude_fixtures", ROOT / "tests/test_additional_evaluation.py")
runner = adapter.runner


def invalid_example():
    example = fixtures.example()
    example['tools'][0]['function']['parameters']['properties'] = {
        'vehicle': {'type': 'object', 'properties': {'año_vehiculo': {'type': 'integer'}}}}
    return example


def failing_generator(status=400):
    calls = []

    def generate(example, history):
        calls.append((example, history))
        raise adapter.LLMAPIError('SECRET_PROVIDER_MESSAGE', status_code=status, attempts=1)

    generator = SimpleNamespace(provider='anthropic', api_style='anthropic_messages', model='claude-test',
                                gateway='apim', generate=generate)
    return generator, calls


def test_actual_known_schema_400_preserves_input_and_unknown_usage(monkeypatch):
    generator, calls = failing_generator()
    example = invalid_example()
    original = json.dumps(example, sort_keys=True)
    clock = iter([10., 10.75])
    monkeypatch.setattr(adapter.time, 'perf_counter', lambda: next(clock))
    proposal = adapter.wrap_generator(generator).generate(example, [])
    assert len(calls) == 1 and json.dumps(example, sort_keys=True) == original
    assert proposal['calls'] is None and proposal['incomplete'] == 'unsupported_schema'
    assert proposal['usage'] == {} and proposal['raw_usage'] is None
    assert proposal['latency_s'] == .75 and proposal['retry_wait_s'] == 0 and proposal['api_calls'] == 1
    assert proposal['provider_input_error']['request_sent'] is True
    assert 'SECRET' not in json.dumps(proposal)
    cost = runner.token_cost({'trace': [{'proposal': proposal, 'verification': {'api_calls': 0}}]}, {})
    assert cost['complete'] is False and cost['estimated_usd'] is None


@pytest.mark.parametrize('status,invalid', [(400, False), (429, True), (500, True)])
def test_unrelated_schema_or_transport_errors_still_stop(status, invalid):
    generator, calls = failing_generator(status)
    with pytest.raises(adapter.LLMAPIError):
        adapter.wrap_generator(generator).generate(invalid_example() if invalid else fixtures.example(), [])
    assert len(calls) == 1


def test_valid_generation_unchanged_and_full_bfcl_retains_one_unsupported_id():
    expected = fixtures.proposal()
    generator = SimpleNamespace(provider='anthropic', api_style='anthropic_messages', model='claude-test',
                                gateway='apim', generate=lambda example, history: expected)
    assert adapter.wrap_generator(generator).generate(fixtures.example(), []) is expected
    data = ROOT / 'data/bfcl/examples.jsonl'
    if not data.exists():
        pytest.skip('Canonical data unavailable; tests do not download it.')
    rows = runner.read_jsonl(data)
    unsupported = [row for row in rows if adapter.unsupported_property_names(row)]
    assert len(rows) == 3051 and [row['id'] for row in unsupported] == ['live_simple_67-31-0']
    assert adapter.unsupported_property_names(unsupported[0]) == ['año_vehiculo']


def test_wrapper_settings_and_provider_continuation_resume_are_fingerprinted(tmp_path, monkeypatch):
    events = []
    args = fixtures.args(tmp_path, output=tmp_path/'source', workers=8,
                         benchmark='bfcl', model='claude-test')
    generator, verifier = fixtures.Generator(events), fixtures.Verifier(events)
    generator.model = 'claude-test'
    cases = [fixtures.example()]
    old_settings = fixtures.config(args, cases, generator, verifier)
    runner.execute(args, cases, runner.prepare_apibank.score_example, old_settings, generator, verifier)
    monkeypatch.setattr(continuation, '_active_runner_pids', lambda source: [])
    child = tmp_path/'compatible32'
    provenance = continuation.prepare(args.output, child, 32, checkpointed=True,
                                      provider_input_adapter=Path(adapter.__file__))
    manifest = json.loads((child/'manifest.json').read_text())
    settings = {key: value for key,value in manifest.items() if key not in {'created_at','config_id'}}
    assert settings == {**old_settings, 'workers': 32, **adapter.adapter_settings()}
    assert runner.validate_resume(child, settings, True)[1] == {('a', 0, 'direct'), ('a', 0, 'jev_joint')}
    assert set(provenance['settings_changes']) == {'workers','provider_adapter_sha256','provider_input_rule'}
    new_args = fixtures.args(tmp_path, workers=32, benchmark='bfcl', model='claude-test')
    with adapter.patched_runner():
        recomputed = runner.settings_for(new_args, cases,
                                        {'dataset_sha256':'fixed-data','dataset_preparer_sha256':'prepared'},
                                        generator,verifier)
    assert recomputed == settings
    assert runner.make_clients is adapter.ORIGINAL_MAKE_CLIENTS
    assert runner.settings_for is adapter.ORIGINAL_SETTINGS_FOR


def test_wrapper_rejects_other_benchmarks_before_client_or_network_creation():
    with pytest.raises(ValueError, match='only for BFCL'):
        adapter.main(['--benchmark','apibank','--model','claude-test','--api-style','anthropic_messages',
                      '--data','unused','--output','unused'])


def test_unsupported_case_stays_wrong_in_all_three_repetitions(tmp_path):
    class RejectingGenerator(fixtures.Generator):
        def generate(self, example, history):
            assert set(example) == {'messages', 'tools'}
            assert all('score' not in record for record in history)
            self.calls.append((example, history))
            raise adapter.LLMAPIError('SECRET', status_code=400, attempts=1)

    events = []
    args = fixtures.args(tmp_path, replicates=3, benchmark='bfcl', model='claude-test')
    generator = RejectingGenerator(events)
    generator.model = 'claude-test'
    adapter.wrap_generator(generator)
    verifier = fixtures.Verifier(events, accepts=[False] * 12)
    cases = [invalid_example()]
    settings = adapter.settings_for(args, cases, {'dataset_sha256':'fixed-data','dataset_preparer_sha256':'prepared'},
                                    generator, verifier)
    summary = runner.execute(args, cases, runner.prepare_apibank.score_example, settings, generator, verifier)
    rows = runner.read_jsonl(args.output/'records.jsonl')
    assert len(rows) == 6 and len(generator.calls) == 12
    assert {row['replicate'] for row in rows} == {0, 1, 2}
    assert all(row['calls'] is None and row['score']['correct'] is False for row in rows)
    assert all(row['accounting']['complete'] is False and row['accounting']['estimated_usd'] is None for row in rows)
    assert all(step['proposal']['incomplete'] == 'unsupported_schema' for row in rows for step in row['trace'])
    assert summary['repetition_report']['direct']['accuracy_mean'] == 0
    assert summary['repetition_report']['jev_joint']['accuracy_mean'] == 0
    assert summary['methods']['jev_joint']['incomplete_final_n'] == 3
