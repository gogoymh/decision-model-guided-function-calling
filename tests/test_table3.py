"""Repair-limit tooling: extending JBFC trajectories to a larger limit and the paper figure (formerly Table 3)."""

import hashlib
import importlib.util
import json
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


def test_unrun_figure_draws_empty_axes_for_the_six_limits():
    figure = table3.latex_repairs_figure()
    assert table3.LIMITS == (1, 2, 3, 5, 7, 10) and "xtick={1,2,3,5,7,10}" in figure
    assert "not yet run" in figure and r"\addplot" not in figure and "Limits without points are unrun." in figure
    assert r"\label{fig:repairs}" in figure and r"\begin{figure}[t]" in figure
    assert "continued to $r=10$ and cut at each limit ($r=3$ is Table~\\ref{tab:main})." in figure
    assert "legend columns=2" in figure and "gray" not in figure


def test_filled_figure_plots_and_labels_every_limit():
    figure = table3.latex_repairs_figure({1: summary(.70), 2: summary(.73), 3: summary(.74), 5: summary(.74),
                                          7: summary(.72), 10: summary(.71)})
    call, function = [line for line in figure.splitlines() if line.startswith(r"\addplot")]
    assert "(1,70.0000) (2,73.0000) (3,74.0000) (5,74.0000) (7,72.0000) (10,71.0000)" in call
    assert "(10,80.0000)" in function and "mark=square*" in function
    assert r"{2\\\color{repaircost}0.20}" in figure and r"{10\\\color{repaircost}0.20}" in figure
    assert r"at (axis cs:10,71.0000) {71.0}" in figure and "gray: USD per 1,000 requests" in figure
    assert "Generator" not in figure  # the generator is named in Table 2's caption
    assert "not yet run" not in figure and "unrun" not in figure
    assert r"\textbf{DMFC}" in figure and "JBFC" not in figure


def test_table3_paper_check_follows_the_paper_protocol_except_the_limit():
    path = Path("data/bfcl/examples.jsonl")
    if not path.exists():
        pytest.skip("Requires prepared BFCL files")
    manifest = {**PAPER_SETTINGS, "ids": [row["id"] for row in load_examples(path)], "methods": ["jev_joint"],
                "max_repairs": 10, "prices_usd_per_million": dict(PAPER_PRICES),
                "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "extends": {"max_repairs": 7}}
    root = {**PAPER_SETTINGS, "methods": ["direct", "jev_joint"], "max_repairs": 2,
            "prices_usd_per_million": dict(PAPER_PRICES)}
    table3.check_paper_protocol({"manifest": manifest}, root)
    for change in ({"reasoning_effort": "low"}, {"ids": ["simple_python_0"]}, {"llm_model": "gpt-5.4-mini-2026-03-17"}):
        with pytest.raises(ValueError):
            table3.check_paper_protocol({"manifest": {**manifest, **change}}, root)
    for change in ({"max_repairs": 3}, {"extends": {"max_repairs": 2}}, {"reasoning_effort": "low"}):
        with pytest.raises(ValueError, match="r=2 run"):
            table3.check_paper_protocol({"manifest": manifest}, {**root, **change})


def make_run(directory, records, extends=None):
    directory.mkdir()
    (directory / "records.jsonl").write_text(records)
    manifest = {"config_id": f"config-{directory.name}", "max_repairs": int(directory.name[1:]),
                "methods": ["jev_joint"] if extends else ["direct", "jev_joint"]}
    if extends:
        manifest["extends"] = {"run_dir": str(extends), "config_id": f"config-{extends.name}",
                               "max_repairs": int(extends.name[1:]),
                               "records_sha256": hashlib.sha256((extends / "records.jsonl").read_bytes()).hexdigest()}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def test_table3_follows_a_chain_of_extensions_back_to_the_original_run(tmp_path):
    root = make_run(tmp_path / "r2", "root\n")
    middle = make_run(tmp_path / "r7", "middle\n", extends=root)
    top = make_run(tmp_path / "r10", "top\n", extends=middle)
    assert table3.check_extension(top) == (top, [middle, root])
    with pytest.raises(ValueError, match="extend_run"):
        table3.check_extension(root)
    (root / "records.jsonl").write_text("changed\n")
    with pytest.raises(ValueError, match="records changed"):
        table3.check_extension(top)


def test_table3_template_fills_only_its_block(tmp_path):
    paper = tmp_path / "paper.tex"
    paper.write_text("% BEGIN AUTO RESULTS\nold\n% END AUTO RESULTS\n% BEGIN AUTO REPAIRS\nold\n% END AUTO REPAIRS\n")
    table3.main(["--template", "--paper", str(paper)])
    source = paper.read_text()
    assert "% BEGIN AUTO RESULTS\nold\n% END AUTO RESULTS" in source and r"\label{fig:repairs}" in source


APIM = {"jev_gateway": "apim", "jev_base_url": "https://gw.example/llm/typesafe",
        "jev_endpoint": "https://gw.example/llm/typesafe/v1/systemone"}
DIRECT = {"jev_gateway": "direct", "jev_base_url": "https://api.typesafe.ai",
          "jev_endpoint": "https://api.typesafe.ai/v1/systemone"}


def settings(**changes):
    values = {key: f"same-{key}" for key in extend_run.SAME_AS_SOURCE}
    values.update(APIM, jev_model="jev-1.13.0")
    return {**values, **changes}


def test_jev_route_may_change_only_on_request_and_is_recorded():
    source = {**settings(), "max_repairs": 7}
    assert extend_run.compare_settings(source, settings()) == ([], {})
    changed, record = extend_run.compare_settings(source, settings(**DIRECT))
    assert set(changed) == set(extend_run.JEV_ROUTE) and record == {}
    changed, record = extend_run.compare_settings(source, settings(**DIRECT), jev_gateway="direct")
    assert changed == [] and record == {**DIRECT, "jev_routes": [{"first_attempt": 0, **APIM},
                                                                  {"first_attempt": 8, **DIRECT}]}
    changed, _ = extend_run.compare_settings(source, settings(**DIRECT, jev_model="jev-2"), jev_gateway="direct")
    assert changed == ["jev_model"]
    # A later extension on the same route keeps the recorded routes.
    later = {**source, **record, "max_repairs": 10}
    assert extend_run.compare_settings(later, settings(**DIRECT), jev_gateway="direct") == (
        [], {**DIRECT, "jev_routes": record["jev_routes"]})


def test_table3_accepts_a_recorded_jev_route_and_states_it_in_the_caption():
    path = Path("data/bfcl/examples.jsonl")
    if not path.exists():
        pytest.skip("Requires prepared BFCL files")
    routes = [{"first_attempt": 0, **APIM}, {"first_attempt": 8, **DIRECT}]
    manifest = {**PAPER_SETTINGS, "ids": [row["id"] for row in load_examples(path)], "methods": ["jev_joint"],
                "max_repairs": 10, "prices_usd_per_million": dict(PAPER_PRICES),
                "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **DIRECT, "jev_routes": routes}
    root = {**PAPER_SETTINGS, **APIM, "methods": ["direct", "jev_joint"], "max_repairs": 2,
            "prices_usd_per_million": dict(PAPER_PRICES)}
    table3.check_paper_protocol({"manifest": manifest}, root)
    for bad in ([{"first_attempt": 0, **DIRECT}, {"first_attempt": 8, **DIRECT}], [{"first_attempt": 8, **DIRECT}],
                [{"first_attempt": 0, **APIM}, {"first_attempt": 8, **APIM}]):
        with pytest.raises(ValueError, match="Jev routes"):
            table3.check_paper_protocol({"manifest": {**manifest, "jev_routes": bad}}, root)
    with pytest.raises(ValueError):  # a different Jev model is never a route change
        table3.check_paper_protocol({"manifest": {**manifest, "jev_model": "jev-2"}}, root)
    final = {**summary(.77), "manifest": {"llm_model": "gpt-5-nano", "jev_routes": routes}}
    table = table3.latex_repairs_figure({r: summary(.7) for r in (1, 2, 3, 5, 7)} | {10: final})
    assert r"Repairs 8--10 call \textit{Jev} directly." in table
    assert "call \\textit{Jev}" not in table3.latex_repairs_figure({r: summary(.7) for r in table3.LIMITS})
