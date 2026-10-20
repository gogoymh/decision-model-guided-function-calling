"""Table of the Jev threshold experiments (make threshold-table): docs/threshold.md.

Per generator, Direct and DMFC at the runs' original threshold (0.8) come from Table 1's three
runs; every other threshold with reruns (scripts/rethreshold.py, results/jbfc_t<T>_<model>_run<k>)
adds a DMFC row. Reruns start from their run's first proposals, so all rows of a generator
are paired example by example. Columns: mean accuracy +- s.d. across runs and each run;
paired differences from Direct and from DMFC at 0.8 (95% intervals, each example averaged
over the runs first, then 2,000 resamples with seed 1729); first proposals rejected by the
first check (right and wrong ones); fixes and breaks against the first proposal, and the
breaks where nothing passed and the last proposal was returned; the accuracy if a
trajectory that never passes returned its first proposal instead (same trajectories, so
exact); repairs, limit reached, acceptance and its precision, cost and time; accuracy by
category. The paper and README use threshold 0.5 with the first proposal kept when none passes (user
decision 2026-10-18; make final, scripts/final_tables.py); this table records how that was chosen.

Usage: python scripts/threshold_table.py [--output docs/threshold.md] [--json OUT]
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import paper_tables  # noqa: E402
import table1_repeat  # noqa: E402
import threshold_compare  # noqa: E402

from jev_fc.reporting import TABLE1_CATEGORIES  # noqa: E402

MODELS = (("gpt-5-nano", table1_repeat.SMALL_RUNS), ("gpt-5.4-mini", table1_repeat.LARGE_RUNS))
CATEGORIES = [(category, label) for category, label, _ in TABLE1_CATEGORIES if category != "live_relevance"]


def thresholds_found(model, results=None):
    """Thresholds with at least one finished rerun for this generator, highest first."""
    found = set()
    for path in Path(results or ROOT / "results").glob(f"jbfc_t*_{model}_run*"):
        match = re.fullmatch(rf"jbfc_t([0-9.]+)_{re.escape(model)}_run\d+", path.name)
        if match and table1_repeat._complete(path):
            found.add(float(match.group(1)))
    return sorted(found, reverse=True)


def trajectory_stats(run_dir):
    """DMFC trajectories of one run (live relevance excluded): first-step gate, fixes, breaks, fallback."""
    stats = dict(n=0, right_first=0, right_rejected=0, wrong_first=0, wrong_rejected=0, fixes=0, breaks=0,
                 breaks_at_limit=0, keep_first_correct=0)
    first_checks = {}
    with (Path(run_dir) / "records.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["method"] != "jev_joint":
                continue
            step = row["trace"][0]
            verification = step.get("verification") or {}
            first_checks[row["id"]] = (verification.get("function_probability"), verification.get("arguments_probability"),
                                       bool((step.get("schema") or {}).get("accepted")), verification.get("arguments_skipped"))
            if row["category"] == "live_relevance":
                continue
            first, final, accepted = bool(step["score"]["correct"]), bool(row["score"]["correct"]), bool(row["accepted"])
            stats["n"] += 1
            stats["right_first" if first else "wrong_first"] += 1
            if not step.get("gate_pass"):
                stats["right_rejected" if first else "wrong_rejected"] += 1
            stats["fixes"] += not first and final
            stats["breaks"] += first and not final
            stats["breaks_at_limit"] += first and not final and not accepted
            stats["keep_first_correct"] += final if accepted else first
    return stats, first_checks


def first_gate(check, threshold):
    """The first check's verdict for recorded probabilities at a threshold (schema and both Jev checks)."""
    function, arguments, schema, skipped = check
    return bool(schema and function is not None and function >= threshold
                and (skipped or (arguments is not None and arguments >= threshold)))


def jev_repeat(pairs, threshold):
    """Jev asked twice about the same first proposals (run and rerun): agreement and changed verdicts."""
    same = flips = total = 0
    gaps = []
    for before, after in pairs:
        for example, check in after.items():
            old = before.get(example)
            if old is None:
                continue
            total += 1
            same += check[:2] == old[:2]
            flips += first_gate(check, threshold) != first_gate(old, threshold)
            if check[0] is not None and old[0] is not None:
                gaps.append(abs(check[0] - old[0]))
    if not total:
        return None
    return {"first_proposals": total, "same_probabilities": same / total, "verdict_changed": flips / total,
            "function_gap_p95": statistics.quantiles(gaps, n=20)[18] if len(gaps) > 1 else None}


def mean_stats(stats):
    keys = stats[0].keys()
    return {key: statistics.fmean(item[key] for item in stats) for key in keys}


def method_row(summaries, method):
    def values(key):
        return [(summary["methods"].get(method) or {}).get(key) for summary in summaries]
    row = {key: table1_repeat.spread(values(key)) for key in threshold_compare.RUN_KEYS}
    row["categories"] = {category: table1_repeat.spread([paper_tables._value(summary, method, "accuracy", category)
                                                         for summary in summaries]) for category, _ in CATEGORIES}
    return row


def model_report(model, base_dirs, root=ROOT):
    """Rows for one generator; paths are relative to root (the repository)."""
    root = Path(root)
    base = [paper_tables.load_summary(root / run_dir) for run_dir in base_dirs]
    rows_by_run = [table1_repeat.load_rows(root / run_dir) for run_dir in base_dirs]
    base_stats = [trajectory_stats(root / run_dir) for run_dir in base_dirs]
    report = {"runs": list(base_dirs), "direct": method_row(base, "direct"), "jbfc": {}}
    report["jbfc"][str(base[0]["manifest"]["threshold"])] = {
        **method_row(base, "jev_joint"), "run_numbers": list(range(1, len(base_dirs) + 1)),
        "vs_direct": table1_repeat.paired(rows_by_run, "jev_joint", "direct"), "vs_base": None,
        "trajectories": mean_stats([stats for stats, _ in base_stats]), "jev_repeat": None}
    for threshold in thresholds_found(model, root / "results"):
        rerun_dirs = [str(root / threshold_compare.rerun_dir(threshold, model, run))
                      for run in range(1, len(base_dirs) + 1)]
        compared = threshold_compare.compare_model(model, [str(root / run_dir) for run_dir in base_dirs], threshold,
                                                   rerun_dirs)
        if compared is None:
            continue
        reruns = [rerun_dirs[run - 1] for run in compared["runs"]]
        summaries = [paper_tables.load_summary(run_dir) for run_dir in reruns]
        stats = [trajectory_stats(run_dir) for run_dir in reruns]
        repeat = jev_repeat([(base_stats[run - 1][1], checks) for run, (_, checks) in zip(compared["runs"], stats)],
                            threshold)
        name = f"jbfc_t{threshold:g}"
        report["jbfc"][f"{threshold:g}"] = {
            **method_row(summaries, "jev_joint"), "run_numbers": compared["runs"],
            "vs_direct": compared["paired"][f"{name} - direct"], "vs_base": compared["paired"][f"{name} - jbfc"],
            "trajectories": mean_stats([item for item, _ in stats]), "jev_repeat": repeat}
    return report


def _pct(value, digits=1):
    return "–" if value is None else f"{100 * value:.{digits}f}"


def _signed(value):
    """Points with a typographic minus; values that round to zero carry no sign."""
    text = f"{100 * value:+.1f}"
    return "0.0" if text in ("+0.0", "-0.0") else text.replace("-", "−")


def _delta(paired):
    if not paired or paired.get("delta") is None:
        return "–"
    low, high = paired["ci95"]
    return f"{_signed(paired['delta'])} ({_signed(low).lstrip('+')}, {_signed(high).lstrip('+')})"


def _mean_sd(item):
    if item["mean"] is None:
        return "–"
    return f"{100 * item['mean']:.1f} ± {100 * item['sd']:.1f}" if item["sd"] is not None else f"{100 * item['mean']:.1f}"


def _known_mean(item, scale=1, digits=3):
    """Mean over runs whose value is known; '*' marks a mean that leaves out runs with unknown cost or time
    (a request blocked by the provider's content filter has no usage, so its run's mean cost is unknown)."""
    known = [value for value in item["runs"] if value is not None]
    if not known:
        return "–"
    text = f"{scale * statistics.fmean(known):.{digits}f}"
    return text + "*" if len(known) < len(item["runs"]) else text


def _runs(item):
    return " / ".join(f"{100 * value:.1f}" for value in item["runs"]) if item["runs"] else "–"


def _label(item, threshold, row):
    """The threshold, with the runs it covers when some reruns are not finished."""
    if len(row["run_numbers"]) == len(item["runs"]):
        return threshold
    return f"{threshold} ({', '.join(map(str, row['run_numbers']))}회차)"


def markdown(report):
    lines = [
        "# Jev 임계값 실험", "",
        "> 자동 생성 파일입니다: `make threshold-table` (`scripts/threshold_table.py`). 직접 고치지 마세요.", "",
        "아래 표는 기존 BFCL GPT 실험에서 임계값과 최종 제안 규칙을 선택한 기록입니다. 현재 원고의 "
        "세 생성기·두 벤치마크 Table 1과 구분하며, 경로와 JSON 키는 당시 기록을 유지합니다.", "",
        "기존 BFCL Table 1의 세 회차(생성기마다)마다 그 회차의 첫 제안을 그대로 두고 **DMFC**만 다른 임계값(함수·인자 검사 모두)으로 "
        "다시 실행했습니다(`make threshold THRESHOLD=…`, `scripts/rethreshold.py`). Direct와 임계값 0.8의 **DMFC**는 "
        "기존 실행 그대로이고, 수정 한도는 r=3, 종료 규칙은 한도까지 통과하지 못하면 마지막 제안을 최종 예측으로 내는 "
        "`execute_last`입니다. 0.8 외의 임계값은 이 예제들의 결과를 본 뒤 고른 값입니다. 이 표를 보고 사용자가 "
        "임계값 0.5와, 통과한 제안이 없으면 첫 제안을 내는 규칙을 논문 설정으로 정했습니다(2026-10-18; 논문 표는 "
        "`make final`, `scripts/final_tables.py`). 아래 \"첫 제안 유지 시 정확도\" 열이 그 규칙의 값입니다.", "",
        "정확도는 live relevance를 뺀 3,035개 기준입니다. 차이는 같은 예제끼리 짝지어, 예제마다 회차 평균을 낸 뒤 "
        "예제를 2,000번 재표집한 95% 구간입니다(seed 1729).", "",
        "## 정확도", "",
        "| 생성기 | 방법 | 임계값 | 정확도 (%, 평균 ± 표준편차) | 회차별 | Direct 대비 (점, 95% 구간) | 0.8 대비 (점, 95% 구간) |",
        "|---|---|---|---|---|---|---|"]
    for model, item in report.items():
        lines.append(f"| `{model}` | Direct | – | {_mean_sd(item['direct']['accuracy'])} | {_runs(item['direct']['accuracy'])} | – | – |")
        for threshold, row in item["jbfc"].items():
            lines.append(f"| | **DMFC** | {_label(item, threshold, row)} | {_mean_sd(row['accuracy'])} | {_runs(row['accuracy'])} | "
                         f"{_delta(row['vs_direct'])} | {_delta(row['vs_base'])} |")
    lines += ["", "## 동작과 비용 (회차 평균)", "",
              "첫 제안 거부는 첫 검사에서 통과하지 못한 첫 제안 수입니다(맞는 것 / 틀린 것). 망가뜨린 것 중 괄호 안은 "
              "한도까지 통과하지 못해 마지막 제안을 최종 예측으로 낸 경우입니다. \"첫 제안 유지\"는 같은 궤적에서 통과한 제안이 "
              "없을 때 마지막 제안 대신 첫 제안을 냈다면의 정확도입니다(새 실행 없이 정확히 계산).", "",
              "| 생성기 | 임계값 | 첫 제안 거부: 맞는 것 | 틀린 것 | 고친 것 | 망가뜨린 것 (한도 도달) | 첫 제안 유지 시 정확도 (%) "
              "| 수정 횟수 | 한도 도달 (%) | 통과 (%) | 통과한 것의 정답률 (%) | USD/1k | API 초 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for model, item in report.items():
        direct = item["direct"]
        lines.append(f"| `{model}` | Direct | – | – | – | – | – | – | – | – | – | "
                     f"{_known_mean(direct['mean_cost_usd'], 1000)} | {_known_mean(direct['mean_latency_s'], 1, 2)} |")
        for threshold, row in item["jbfc"].items():
            t = row["trajectories"]
            lines.append(
                f"| | {_label(item, threshold, row)} | {t['right_rejected']:.0f} / {t['right_first']:.0f} | {t['wrong_rejected']:.0f} / "
                f"{t['wrong_first']:.0f} | {t['fixes']:.0f} | {t['breaks']:.0f} ({t['breaks_at_limit']:.0f}) | "
                f"{_pct(t['keep_first_correct'] / t['n'])} | {row['mean_repairs']['mean']:.2f} | "
                f"{_pct(row['exhaustion_rate']['mean'])} | {_pct(row['coverage']['mean'])} | "
                f"{_pct(row['accepted_precision']['mean'])} | {_known_mean(row['mean_cost_usd'], 1000)} | "
                f"{_known_mean(row['mean_latency_s'], 1, 2)} |")
    lines += ["", "## 범주별 정확도 (%, 회차 평균)", "",
              "| 생성기 | 방법 | " + " | ".join(label for _, label in CATEGORIES) + " |",
              "|---|---|" + "---|" * len(CATEGORIES)]
    for model, item in report.items():
        lines.append(f"| `{model}` | Direct | " + " | ".join(_pct(item["direct"]["categories"][c]["mean"]) for c, _ in CATEGORIES) + " |")
        for threshold, row in item["jbfc"].items():
            lines.append(f"| | **DMFC** {_label(item, threshold, row)} | " + " | ".join(_pct(row["categories"][c]["mean"]) for c, _ in CATEGORIES) + " |")
    notes = []
    for model, item in report.items():
        for threshold, row in item["jbfc"].items():
            repeat = row["jev_repeat"]
            if repeat:
                gap = "–" if repeat["function_gap_p95"] is None else f"{repeat['function_gap_p95']:.2f}"
                notes.append(f"`{model}` {threshold}: 확률이 같은 첫 제안 {_pct(repeat['same_probabilities'])}%, "
                             f"함수 확률 차이의 95번째 백분위수 {gap}, 첫 검사 판정이 바뀐 첫 제안 {_pct(repeat['verdict_changed'])}%")
    lines += ["", "## 참고", "",
              "- 다시 실행하면 Jev에 같은 첫 제안을 다시 묻습니다. Jev의 답은 호출마다 조금씩 다릅니다: "
              + ("; ".join(notes) or "–") + ".",
              "- `*`: 공급자의 콘텐츠 필터에 막혀 사용량이 기록되지 않은 요청이 있는 회차는 그 회차의 평균 비용·시간을 알 수 "
              "없어, 나머지 회차로 평균했습니다(정확도에는 영향 없음; 막힌 제안은 틀린 것으로 셉니다).",
              "- 실행: " + "; ".join(f"`{model}` " + ", ".join(f"`{run}`" for run in item["runs"]) for model, item in report.items())
              + "; 재실행은 `results/jbfc_t<임계값>_<생성기>_run<회차>`.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=str(ROOT / "docs/threshold.md"))
    parser.add_argument("--json", help="also write the numbers to this file")
    args = parser.parse_args(argv)
    report = {model: model_report(model, base_dirs) for model, base_dirs in MODELS}
    Path(args.output).write_text(markdown(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}", file=sys.stderr)
    return report


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
