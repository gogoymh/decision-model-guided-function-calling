"""The paper's final JBFC settings: threshold 0.5 and the first proposal when none passes (make final)."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

from jev_fc.reporting import PAPER_PRICES, build_summary, read_jsonl, token_cost

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("final_tables", ROOT / "scripts/final_tables.py")
final_tables = importlib.util.module_from_spec(spec)
spec.loader.exec_module(final_tables)

MINI_PRICES = {"llm_input": .75, "llm_cached_input": .075, "llm_output": 4.5, "jev_input": .042}
BASE = {"version": "0.1.0", "source_sha256": "code", "dataset_sha256": "data", "split": "all", "split_seed": 1729,
        "limit": None, "replicates": 1, "max_repairs": 3, "exhaustion_policy": "execute_last", "synthetic": False,
        "llm_provider": "openai", "llm_api_style": "azure_chat_completions", "llm_gateway": "apim",
        "max_completion_tokens": 4096, "temperature": None, "jev_model": "jev-1.13.0", "jev_gateway": "direct",
        "reasoning_effort": "none"}
IDS = ["a", "b", "c", "d", "e"]
# JBFC's attempts per example as (correct, passes): a and e start right but never pass and end wrong; b is
# fixed by a passing repair; c starts wrong, never passes and ends right; d passes at once.
JOINT = {"a": [(1, 0), (0, 0), (0, 0), (0, 0)], "b": [(0, 0), (1, 1)], "c": [(0, 0), (0, 0), (0, 0), (1, 0)],
         "d": [(1, 1)], "e": [(1, 0), (0, 0), (1, 0), (0, 0)]}
# Figure 2's run continues the trajectories that reached r=3 up to r=10.
EXTENDED = {"a": [(0, 0)] * 7, "c": [(0, 0), (1, 1)], "e": [(0, 0)] * 6 + [(1, 1)]}


def step(attempt, correct, passes, run, usage=True):
    return {"attempt": attempt,
            "proposal": {"calls": [{"name": f"f{attempt}", "arguments": {}}], "content": f"run {run}",
                         "latency_s": 1.0, "usage": {"prompt_tokens": 100, "completion_tokens": 10} if usage else {}},
            "verification": {"function_probability": .9, "arguments_probability": .9, "latency_s": .5,
                             "api_calls": 1, "usage": {"input_tokens": 50}},
            "schema": {"accepted": True, "errors": []}, "feedback": {}, "gate_pass": bool(passes),
            "score": {"correct": bool(correct), "function_correct": bool(correct),
                      "arguments_correct": bool(correct)}}


def record(example, method, attempts, run, prices, cap=3, unknown=False):
    trace = [step(i, correct, passes, run, usage=not (unknown and i == 1)) for i, (correct, passes) in
             enumerate(attempts)]
    passed = trace[-1]["gate_pass"]
    row = {"id": example, "replicate": 0, "method": method, "category": "simple_python", "max_repairs": cap,
           "exhaustion_policy": "execute_last", "calls": trace[-1]["proposal"]["calls"], "score": trace[-1]["score"],
           "accepted": None if method in ("direct", "self_refine") else passed,
           "exhausted": method not in ("direct", "self_refine") and not passed, "dispatched": True,
           "abstained": False, "repairs": len(trace) - 1, "trace": trace, "synthetic": False,
           "initial_proposal_shared": True, "logical_latency_s": 1.5 * len(trace)}
    row["accounting"] = token_cost(row, prices)
    return row


def rows_for(method, run, prices, cap=3, unknown=()):
    rows = []
    for example in IDS:
        first = JOINT[example][0]
        if method == "direct":
            attempts = [(first[0], 1)]
        elif method == "self_refine":
            attempts = [(first[0], 0), (0, 0), (0, 0), (first[0], 1)]
        elif method == "schema_only":
            attempts = [(first[0], 1)]
        elif cap == 3:
            attempts = JOINT[example]
        else:
            attempts = JOINT[example] + EXTENDED.get(example, [])
        rows.append(record(example, method, attempts, run, prices, cap, unknown=example in unknown))
    return rows


def write_run(root, name, methods, threshold, model="gpt-5-nano", run=1, cap=3, extra=None, unknown=()):
    prices = PAPER_PRICES if model == "gpt-5-nano" else MINI_PRICES
    path = root / "results" / name
    path.mkdir(parents=True)
    manifest = {**BASE, "ids": IDS, "methods": methods, "shared_initial_methods": methods, "threshold": threshold,
                "llm_model": model, "prices_usd_per_million": prices, "max_repairs": cap,
                "expected_records": len(IDS) * len(methods), "config_id": f"config-{name}", **(extra or {})}
    rows = [{**row, "config_id": manifest["config_id"]}
            for method in methods for row in rows_for(method, run, prices, cap, unknown if method != "direct" else ())]
    (path / "manifest.json").write_text(json.dumps(manifest))
    (path / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    build_summary(path)
    return path


@pytest.fixture
def root(tmp_path, monkeypatch):
    """Every run the final tables use, with five examples."""
    monkeypatch.setattr(final_tables, "_check_all_examples", lambda manifest: None)
    write_run(tmp_path, "table2_gpt-5-nano_r3", list(final_tables.METHODS), .8)
    for model in final_tables.MODELS:
        for run in (1, 2, 3):
            write_run(tmp_path, f"jbfc_t0.5_{model}_run{run}", ["direct", "jev_joint"], .5, model, run,
                      unknown=("b",) if (model, run) == ("gpt-5-nano", 2) else ())
    for method in ("jev_function", "jev_arguments", "jev_joint_latest"):
        write_run(tmp_path, f"{method}_t0.5_gpt-5-nano_run1", ["direct", method], .5)
    source = tmp_path / "results/jbfc_t0.5_gpt-5-nano_run1"
    write_run(tmp_path, "table3_gpt-5-nano_r10_t0.5", ["jev_joint"], .5, cap=10, extra={"extends": {
        "run_dir": "results/jbfc_t0.5_gpt-5-nano_run1", "config_id": "config-jbfc_t0.5_gpt-5-nano_run1",
        "max_repairs": 3, "records_sha256": final_tables._sha256(source / "records.jsonl")}})
    return tmp_path


def test_keep_first_returns_the_first_proposal_only_when_nothing_passed():
    prices = PAPER_PRICES
    rejected = record("a", "jev_joint", JOINT["a"], 1, prices)
    kept = final_tables.keep_first(rejected)
    assert kept["calls"] == rejected["trace"][0]["proposal"]["calls"] and kept["score"]["correct"] is True
    assert kept["final_attempt"] == 0 and kept["final_rule"] == "keep_first"
    # The loop ran as recorded: trajectory, acceptance, repairs, time and cost are unchanged.
    for key in ("trace", "accepted", "exhausted", "repairs", "logical_latency_s", "accounting"):
        assert kept[key] == rejected[key]
    assert rejected["calls"] == rejected["trace"][-1]["proposal"]["calls"]  # the input is not changed
    passed = record("b", "jev_joint", JOINT["b"], 1, prices)
    assert final_tables.keep_first(passed)["calls"] == passed["calls"]
    assert final_tables.keep_first(passed)["final_attempt"] == 1
    # Direct and self-refine have no first passing proposal to stop at: they keep their last proposal.
    refine = record("a", "self_refine", [(1, 0), (0, 0), (0, 0), (0, 1)], 1, prices)
    assert final_tables.keep_first(refine)["calls"] == refine["calls"]


def test_derive_combines_runs_and_applies_the_rule(root):
    parts = {"direct": "results/jbfc_t0.5_gpt-5-nano_run1", "jev_joint": "results/jbfc_t0.5_gpt-5-nano_run1"}
    first = final_tables.derive("results/final/x", parts, root)
    last = final_tables.derive("results/final/y", parts, root, rule=False)
    assert first["methods"]["direct"]["accuracy"] == pytest.approx(.6)
    assert last["methods"]["jev_joint"]["accuracy"] == pytest.approx(.6)  # b and c right, a and e broken
    assert first["methods"]["jev_joint"]["accuracy"] == pytest.approx(.8)  # a and e kept, c lost
    assert first["manifest"]["final_rule"] == "keep_first" and "final_rule" not in last["manifest"]
    assert first["manifest"]["derived"]["parts"][0]["run_dir"] == "results/jbfc_t0.5_gpt-5-nano_run1"
    assert first["manifest"]["config_id"] != last["manifest"]["config_id"]
    # Same requests, so the same time and cost; the derived run is a normal run.
    for key in ("mean_cost_usd", "mean_latency_s", "mean_repairs", "exhaustion_rate", "coverage"):
        assert first["methods"]["jev_joint"][key] == last["methods"]["jev_joint"][key]
    assert final_tables.load_run(root / "results/final/x")[0]["methods"] == ["direct", "jev_joint"]
    # Metrics that read the final proposal follow the rule: only b ends on another function than it started.
    assert first["methods"]["jev_joint"]["function_change_rate"] == pytest.approx(.2)
    assert last["methods"]["jev_joint"]["function_change_rate"] == pytest.approx(.8)
    assert first["by_category"]["simple_python/jev_joint"]["function_change_rate"] == pytest.approx(.2)
    saved = json.loads((root / "results/final/x/summary.json").read_text())
    assert saved["methods"]["jev_joint"]["function_change_rate"] == pytest.approx(.2)
    assert "function_change_rate" in (root / "results/final/x/summary.csv").read_text().splitlines()[0]
    # Writing again replaces the derived run instead of appending.
    again = final_tables.derive("results/final/x", parts, root)
    assert again["completed_records"] == 10


def test_derive_refuses_parts_that_do_not_belong_together(root):
    nano = "results/jbfc_t0.5_gpt-5-nano_run1"
    with pytest.raises(ValueError, match="threshold"):
        final_tables.derive("results/final/x", {"direct": nano, "jev_joint": "results/table2_gpt-5-nano_r3"}, root)
    with pytest.raises(ValueError, match="first proposal"):
        final_tables.derive("results/final/x", {"direct": "results/jbfc_t0.5_gpt-5-nano_run2", "jev_joint": nano},
                            root)
    with pytest.raises(ValueError, match="llm_model"):
        final_tables.derive("results/final/x", {"direct": "results/jbfc_t0.5_gpt-5.4-mini_run1",
                                                "jev_joint": nano}, root)
    with pytest.raises(ValueError, match="no jev_function rows"):
        final_tables.derive("results/final/x", {"jev_function": nano}, root)


def test_build_derives_tables_1_and_2_with_both_rules(root):
    built = final_tables.build(root)
    nano = built["table1"]["gpt-5-nano"]
    assert nano[0]["manifest"]["methods"] == list(final_tables.METHODS)
    assert [s["manifest"]["methods"] for s in nano[1:]] == [["direct", "jev_joint"]] * 2
    assert {part["run_dir"] for part in nano[0]["manifest"]["derived"]["parts"]} == {
        "results/table2_gpt-5-nano_r3", "results/jbfc_t0.5_gpt-5-nano_run1",
        "results/jev_function_t0.5_gpt-5-nano_run1", "results/jev_arguments_t0.5_gpt-5-nano_run1",
        "results/jev_joint_latest_t0.5_gpt-5-nano_run1"}
    # The rule applies to every method that stops at its first passing proposal.
    assert nano[0]["methods"]["jev_joint_latest"]["accuracy"] == pytest.approx(.8)
    assert nano[0]["methods"]["self_refine"]["accuracy"] == pytest.approx(.6)
    assert [s["methods"]["jev_joint"]["accuracy"] for s in built["last"]["gpt-5.4-mini"]] == [.6, .6, .6]
    rules = final_tables.rule_report(built["table1"], built["last"])
    item = rules["gpt-5-nano"]
    assert item["first"]["mean"] == pytest.approx(.8) and item["last"]["mean"] == pytest.approx(.6)
    # Per example: a +1, b 0, c -1, d 0, e +1.
    assert item["first - last"]["delta"] == pytest.approx(.2)
    assert item["first - direct"]["delta"] == pytest.approx(.2) and item["last - direct"]["delta"] == pytest.approx(0)
    assert item["transitions_first"][0] == {"fixes": 1, "breaks": 0}
    assert item["transitions_last"][0] == {"fixes": 2, "breaks": 2}
    costs = final_tables.table1_costs(built["table1"])
    assert [column["unknown"] for column in costs] == [0, 1, 0, 0]  # run 2's JBFC row b has no usage
    assert all(column["mean"] is not None for column in costs)
    report = final_tables.table1_report(built["table1"])
    assert report["columns"]["gpt-5-nano/jbfc"]["accuracy"]["mean"] == pytest.approx(.8)


def test_build_refuses_a_rerun_with_another_threshold(root):
    manifest_path = root / "results/jbfc_t0.5_gpt-5.4-mini_run2/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_path.write_text(json.dumps({**manifest, "threshold": .7}))
    with pytest.raises(ValueError, match="threshold"):
        final_tables.build(root)


def test_figure_cuts_the_continued_run_and_matches_table_2(root):
    figure = final_tables.figure_summaries(root)
    assert sorted(figure) == [1, 2, 3, 5, 7, 10]
    built = final_tables.build(root)
    assert figure[3]["methods"]["jev_joint"]["accuracy"] == built["table1"]["gpt-5-nano"][0]["methods"]["jev_joint"][
        "accuracy"]
    # At r=10, c and e pass (c right, e right) and a still fails: it returns its right first proposal.
    assert figure[10]["methods"]["jev_joint"]["accuracy"] == pytest.approx(1.0)
    rows = {row["id"]: row for row in read_jsonl(root / "results/final/figure2/r10/records.jsonl")}
    assert rows["a"]["final_attempt"] == 0 and rows["c"]["final_attempt"] == 5
    # From r=3 to r=10 only c changes (wrong first proposal kept at r=3, a passing right call at r=10).
    changes = final_tables.limit_changes(figure)
    assert (changes["turned_right"], changes["turned_wrong"], changes["turned_right_passed"]) == (1, 0, 1)
    assert changes["turned_right_no_call"] == 0 and changes["time_ratio"] > 1


def test_figure_needs_the_continuation_of_the_threshold_rerun(root):
    path = root / "results/table3_gpt-5-nano_r10_t0.5/manifest.json"
    manifest = json.loads(path.read_text())
    path.write_text(json.dumps({**manifest, "extends": {**manifest["extends"], "records_sha256": "other"}}))
    with pytest.raises(ValueError, match="changed"):
        final_tables.figure_summaries(root)


def test_rules_table():
    item = {"direct": {"mean": .811}, "last": {"mean": .807}, "first": {"mean": .821},
            "first - last": {"delta": .014, "ci95": [.009, .018]}}
    nano = {"direct": {"mean": .641}, "last": {"mean": .751}, "first": {"mean": .748},
            "first - last": {"delta": -.0035, "ci95": [-.0072, .0001]}}
    table = final_tables.latex_rules({"gpt-5-nano": nano, "gpt-5.4-mini": item})
    assert r"\texttt{gpt-5-nano} & 64.1 & \textbf{75.1} & 74.8 & $-$0.35 [$-$0.72, 0.01]\\" in table
    assert r"Generator & Direct & Last & First & First $-$ last\\" in table
    assert r"\texttt{gpt-5.4-mini} & 81.1 & 80.7 & \textbf{82.1} & +1.40 [0.90, 1.80]\\" in table
    assert final_tables._signed(-.00001) == "0.00" and final_tables._signed(-.0003) == "$-$0.03"
    assert r"\label{tab:rules}" in table and "mean of three runs" in table


@pytest.mark.parametrize("split_source", [False, True])
def test_main_fills_the_paper(root, monkeypatch, split_source):
    monkeypatch.setattr(final_tables, "ROOT", root)
    paper = root / "main.tex"
    blocks = "".join(f"% BEGIN AUTO {name}\nUNFILLED\n% END AUTO {name}\n" for name in
                     ("CATEGORIES", "RESULTS", "REPAIRS", "RULES"))
    target = paper
    if split_source:
        (root / "sec").mkdir()
        target = root / "sec/experiment_setting_and_results.tex"
        paper.write_text(r"\title{Decision-Model-Guided Function Calling}" + "\n"
                         + r"\input{sec/experiment_setting_and_results}" + "\n")
        original_main = paper.read_text()
    target.write_text(blocks)
    report = final_tables.main(["--paper", str(paper), "--json", str(root / "final.json")])
    text = target.read_text()
    if split_source:
        assert paper.read_text() == original_main
        assert r"\textbf{DMFC}" in text and "JBFC" not in text
    assert "UNFILLED" not in text and r"\label{tab:rules}" in text and r"\label{tab:categories}" in text
    assert r"\label{tab:main}" in text and r"\label{fig:repairs}" in text
    assert json.loads((root / "final.json").read_text())["settings"]["threshold"] == .5
    assert report["table2"]["history_ablation"]["delta"] == pytest.approx(0)
    labels = report["history_labels"]
    assert labels["jev_s_per_check"] == pytest.approx(.5) and labels["generator_s_per_proposal"] == pytest.approx(1)
    assert labels["jev_usd_per_1k_checks"] == pytest.approx(50 * .042 / 1000)


def test_main_preserves_acm_short_layout_and_exports_tail_latency(root, monkeypatch):
    monkeypatch.setattr(final_tables, "ROOT", root)
    paper = root / "paper" / "main.tex"
    section = paper.parent / "sec" / "experiment_setting_and_results.tex"
    section.parent.mkdir(parents=True)
    paper.write_text(r"\title{Decision-Model-Guided Function Calling}" + "\n"
                     + r"\input{sec/experiment_setting_and_results}" + "\n")
    original_main = paper.read_text()
    source = final_tables.ACM_SHORT + "\nAUTHOR PROSE\n"
    for name, header in (("CATEGORIES", "Generator & Method & Accuracy & API s & USD/1k"),
                         ("RESULTS", "Method & Call & Func. & Args. & API s & USD/1k")):
        source += (f"% BEGIN AUTO {name}\n" + r"\begin{table}[t]" + "\n"
                   + r"\centering\footnotesize\setlength{\tabcolsep}{3.5pt}" + "\n"
                   + r"\begin{tabular}{llrrr}" + "\n" + header + "\\\\\n"
                   + r"\midrule" + "\nSTALE DATA\n" + r"\bottomrule" + "\n"
                   + r"\end{tabular}" + "\n" + rf"\caption{{Custom {name} caption.}}" + "\n"
                   + r"\end{table}" + f"\n% END AUTO {name}\n")
    source += "% BEGIN AUTO REPAIRS\n" + r"\input{sec/repair_tradeoff}" + "\n% END AUTO REPAIRS\nFALLBACK PROSE\n"
    section.write_text(source)
    plot = section.parent / "repair_tradeoff.tex"
    plot_source = "AUTHOR PLOT LAYOUT\n"
    for color in ("tradeoffaccuracy", "tradeoffmean", "tradeofftail"):
        plot_source += rf"\addplot[color={color}, mark=*, thick] coordinates {{(1,0) (10,0)}};" + "\n"
    plot_source += (r"\caption{Custom plot caption.}" + "\n"
                    + r"\Description{Call accuracy rises from 0.0 to 0.0 percent. "
                    r"Mean API time rises from 0.00 to 0.00 seconds, and the 95th percentile rises from "
                    r"0.00 to 0.00 seconds.}" + "\n")
    plot.write_text(plot_source)
    raw_before = {path: final_tables._sha256(path) for path in (root / "results").glob("*/records.jsonl")}

    report = final_tables.main(["--paper", str(paper), "--json", str(root / "short.json")])

    output = section.read_text()
    assert paper.read_text() == original_main
    # Only measured rows change: prose, font, spacing, column headers and captions remain exact.
    def strip_rows(text):
        return re.sub(r"(\\midrule\n).*?(\n\\bottomrule)", r"\1DATA\2", text, flags=re.DOTALL)

    assert strip_rows(output) == strip_rows(source)
    assert "STALE DATA" not in output and "% BEGIN AUTO RULES" not in output
    assert r"\textbf{DMFC}" in output and "JBFC" not in output
    assert len(re.findall(r" & Direct & ", output)) == 2
    assert r"Direct & 60.0$\pm$0.0 & 1.50" in output
    assert r"\textbf{DMFC} & \textbf{80.0}$\pm$0.0 & 4.50" in output
    updated_plot = plot.read_text()
    assert "AUTHOR PLOT LAYOUT" in updated_plot and r"\caption{Custom plot caption.}" in updated_plot
    assert report["figure2"][10]["p95_latency_s"] == pytest.approx(16.5)
    assert report["table2"]["methods"]["jev_joint"]["p95_latency_s"] is not None
    assert "(10,16.500000)" in updated_plot and "to 16.50 seconds" in updated_plot
    assert raw_before == {path: final_tables._sha256(path) for path in raw_before}


def test_measured_tradeoff_rejects_missing_p95():
    figure = {limit: {"methods": {"jev_joint": {"accuracy": .75, "mean_latency_s": 2,
                                               "p95_latency_s": 5}}} for limit in final_tables.table3.LIMITS}
    figure[3]["methods"]["jev_joint"]["p95_latency_s"] = None
    source = "\n".join(rf"\addplot[color={color}, thick] coordinates {{(1,0)}};"
                       for color in ("tradeoffaccuracy", "tradeoffmean", "tradeofftail"))
    with pytest.raises(ValueError, match="p95_latency_s"):
        final_tables.measured_tradeoff(source, figure)


def test_main_combined_marker_keeps_six_rows_and_updates_ablations_plot(root, monkeypatch):
    import additional_paper_tables

    monkeypatch.setattr(final_tables, "ROOT", root)
    paper = root / "paper/main.tex"
    section = paper.parent / "sec/experiment_setting_and_results.tex"
    section.parent.mkdir(parents=True)
    paper.write_text(r"\title{Decision-Model-Guided Function Calling}" + "\n"
                     + r"\input{sec/experiment_setting_and_results}" + "\n")
    main_before = paper.read_text()
    section.write_text(final_tables.ACM_SHORT + "\nAUTHOR TEXT\n% BEGIN AUTO CATEGORIES\n"
                       + final_tables.COMBINED_MAIN + "\nOLD COMBINED TABLE\n% END AUTO CATEGORIES\n"
                       + "% BEGIN AUTO RESULTS\n" + r"\begin{table}\midrule" + "\nOLD ABLATION\n"
                       + r"\bottomrule\caption{Ablation caption.}\end{table}" + "\n% END AUTO RESULTS\n"
                       + "% BEGIN AUTO REPAIRS\n" + r"\input{sec/repair_tradeoff}" + "\n% END AUTO REPAIRS\nLIMIT TEXT\n")
    plot = section.parent / "repair_tradeoff.tex"
    plot.write_text("\n".join(rf"\addplot[color={color}, thick] coordinates {{(1,0)}};"
                             for color in ("tradeoffaccuracy", "tradeoffmean", "tradeofftail")))
    # Completed-result validation belongs to the exporter, independently tested against actual records.
    # This regression isolates make final's routing and preserves all six sentinel outcomes.
    seen = []
    report = {"status": "complete", "rows": [{"outcome": index + .1} for index in range(6)]}
    def completed_report(actual_root):
        seen.append(actual_root)
        return report
    def six_row_table(actual_report):
        assert actual_report == report
        return final_tables.COMBINED_MAIN + "\n" + r"\begin{table*}" + "\n" + "\n".join(
            f"benchmark {index} & generator & Direct {row['outcome']} & DMFC {row['outcome']}\\\\"
            for index, row in enumerate(actual_report["rows"])) + "\n" + r"\end{table*}" + "\n"
    monkeypatch.setattr(additional_paper_tables, "build_report", completed_report)
    monkeypatch.setattr(additional_paper_tables, "latex_table", six_row_table)
    def no_legacy_rows(*args, **kwargs):
        raise AssertionError("Combined results must never be replaced by the legacy four rows")
    monkeypatch.setattr(final_tables, "compact_main_rows", no_legacy_rows)

    final_tables.main(["--paper", str(paper), "--json", str(root / "combined-final.json")])

    text = section.read_text()
    assert seen == [root] and paper.read_text() == main_before
    assert text.count(" & generator & Direct ") == 6
    assert "OLD COMBINED TABLE" not in text and "OLD ABLATION" not in text
    assert "Direct 5.1 & DMFC 5.1" in text and r"\caption{Ablation caption.}" in text
    assert "AUTHOR TEXT" in text and "LIMIT TEXT" in text
    assert "(10,16.500000)" in plot.read_text()


def test_combined_marker_refuses_unfinished_additions_before_paper_write(root, monkeypatch):
    import additional_paper_tables

    monkeypatch.setattr(final_tables, "ROOT", root)
    paper = root / "main.tex"
    source = final_tables.ACM_SHORT + "\n% BEGIN AUTO CATEGORIES\n" + final_tables.COMBINED_MAIN + "\nSIX ROWS\n% END AUTO CATEGORIES\n"
    paper.write_text(source)
    def incomplete(actual_root):
        raise ValueError("Additional result grid is incomplete")
    monkeypatch.setattr(additional_paper_tables, "build_report", incomplete)
    with pytest.raises(ValueError, match="incomplete"):
        final_tables.update_paper(paper, {}, {}, {}, [])
    assert paper.read_text() == source
