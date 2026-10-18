"""Table 3 tooling: extending JBFC trajectories to a larger repair limit and the table."""

import hashlib
import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

from jev_fc.benchmarks import load_examples
from jev_fc.pipeline import run_example
from jev_fc.reporting import PAPER_PRICES, PAPER_SETTINGS

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


extend_run = load_script("extend_run")
table3 = load_script("table3")

EXAMPLE = {"id": "one", "category": "simple_python", "messages": [{"role": "user", "content": "Weather"}],
           "tools": [{"type": "function", "function": {"name": "weather", "description": "Weather", "parameters": {
               "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}


class Generator:
    """Deterministic: each proposal depends only on how many records the history holds."""

    def __init__(self):
        self.histories = []

    def generate(self, example, history=None):
        self.histories.append(deepcopy(history or []))
        return {"calls": [{"name": "weather", "arguments": {"city": f"City{len(history or [])}"}}],
                "latency_s": 1.0, "usage": {"prompt_tokens": 10, "completion_tokens": 2}}


class Verifier:
    def verify(self, example, calls):
        ok = calls[0]["arguments"]["city"] == "City4"
        return {"function_pass": True, "arguments_pass": ok, "arguments_skipped": False,
                "function_probability": .9, "arguments_probability": .9 if ok else .1,
                "feedback": {"arguments_pass": ok, "city": calls[0]["arguments"]["city"]},
                "latency_s": .5, "api_calls": 1, "usage": {"input_tokens": 5, "output_tokens": 0}}


def test_extension_reproduces_an_uninterrupted_run_with_the_larger_limit():
    initial = Generator().generate(EXAMPLE, [])
    full_generator = Generator()
    full = run_example(EXAMPLE, full_generator, Verifier(), "jev_joint", 7, initial_proposal=initial)
    short = run_example(EXAMPLE, Generator(), Verifier(), "jev_joint", 2, initial_proposal=initial)
    assert short["exhausted"] and len(short["trace"]) == 3
    generator = Generator()
    extended = extend_run.continue_row(EXAMPLE, {"id": "one", "trace": short["trace"]}, generator, Verifier(), 7)
    keys = ("attempt", "proposal", "verification", "schema", "feedback", "gate_pass")
    assert [{k: s[k] for k in keys} for s in extended["trace"]] == [{k: s[k] for k in keys} for s in full["trace"]]
    assert extended["accepted"] and extended["repairs"] == full["repairs"] == 4
    # Only the new attempts reach the real generator, with the same cumulative history.
    assert generator.histories == full_generator.histories[2:]
    assert extended["logical_latency_s"] == full["logical_latency_s"]


def test_extension_refuses_a_trajectory_that_the_pipeline_does_not_reproduce():
    initial = Generator().generate(EXAMPLE, [])
    short = run_example(EXAMPLE, Generator(), Verifier(), "jev_joint", 2, initial_proposal=initial)
    tampered = deepcopy(short["trace"])
    tampered[1]["schema"]["accepted"] = False
    tampered[1]["schema"]["errors"] = [{"check": "arguments", "path": "x", "message": "invented"}]
    with pytest.raises(ValueError, match="differs from the saved trajectory"):
        extend_run.continue_row(EXAMPLE, {"id": "one", "trace": tampered}, Generator(), Verifier(), 7)


def summary(accuracy, model="gpt-5-nano"):
    return {"manifest": {"llm_model": model}, "methods": {"jev_joint": {
        "accuracy": accuracy, "function_accuracy": .8, "mean_repairs": .7, "exhaustion_rate": .3,
        "mean_latency_s": 3.0, "mean_cost_usd": 2e-4}}}


def test_unrun_table3_lists_the_five_limits():
    table = table3.latex_table3()
    for r in (1, 2, 3, 5, 7):
        assert f"${r}$ & -- & -- & -- & -- & -- & -- \\\\" in table
    assert "Bold" not in table and "-- means unrun." in table and r"\label{tab:repairs}" in table
    assert "extended to $r=7$, so $r=2$ matches it." in table


def test_filled_table3_bolds_the_best_accuracy_and_drops_the_unrun_note():
    table = table3.latex_table3({1: summary(.70), 2: summary(.73), 3: summary(.74), 5: summary(.74), 7: summary(.72)})
    assert r"$3$ & \textbf{74.0} & \textbf{80.0} & 0.70 & 30.0 & 3.00 & 0.200 \\" in table
    assert r"$5$ & \textbf{74.0} &" in table and r"$7$ & 72.0 &" in table
    assert r"Generator: \texttt{gpt-5-nano}." in table and "Bold marks the highest accuracy" in table
    assert "--" not in table


def test_table3_paper_check_follows_the_paper_protocol_except_the_limit():
    path = Path("data/bfcl/examples.jsonl")
    if not path.exists():
        pytest.skip("Requires prepared BFCL files")
    manifest = {**PAPER_SETTINGS, "ids": [row["id"] for row in load_examples(path)], "methods": ["jev_joint"],
                "max_repairs": 7, "prices_usd_per_million": dict(PAPER_PRICES),
                "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "extends": {"max_repairs": 2}}
    table3.check_paper_protocol({"manifest": manifest})
    for change in ({"reasoning_effort": "low"}, {"extends": {"max_repairs": 3}}, {"ids": ["simple_python_0"]},
                   {"llm_model": "gpt-5.4-mini-2026-03-17"}):
        with pytest.raises(ValueError):
            table3.check_paper_protocol({"manifest": {**manifest, **change}})


def test_table3_template_fills_only_its_block(tmp_path):
    paper = tmp_path / "paper.tex"
    paper.write_text("% BEGIN AUTO RESULTS\nold\n% END AUTO RESULTS\n% BEGIN AUTO REPAIRS\nold\n% END AUTO REPAIRS\n")
    table3.main(["--template", "--paper", str(paper)])
    source = paper.read_text()
    assert "% BEGIN AUTO RESULTS\nold\n% END AUTO RESULTS" in source and r"\label{tab:repairs}" in source
