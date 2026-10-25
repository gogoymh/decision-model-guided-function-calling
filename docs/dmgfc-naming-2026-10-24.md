# Current manuscript naming

The current paper title is **Decision-Model-Guided Function Calling with Cumulative
Verification Feedback**, retaining the subtitle as requested in the user's PR
comment. The method is abbreviated **DMGFC**. Its ablations are **DMGFC-F** (function judgments), **DMGFC-A**
(argument judgments), and **DMGFC-L** (latest feedback only).

The repository is now
[decision-model-guided-function-calling](https://github.com/gogoymh/decision-model-guided-function-calling).
Use `git@github.com-personal:gogoymh/decision-model-guided-function-calling.git`
with the author's personal GitHub SSH configuration.

The naming revision updates the current manuscript, its PDF metadata, generated
table and figure labels, README artwork, and current explanatory documentation.
It preserves other restored Abstract and Introduction wording, measured values,
experiment method IDs, input fingerprints, and raw traces. `src/jev_fc` and
historical papers under `paper/versions/` retain their original contents. Dated
migration and review notes retain the names used at the time they were written.
The public distribution and import are `dmgfc`, the agent class is
`dmgfc.DMGFC`, and the evaluation command is `dmgfc-eval`. Public method options
are `dmgfc`, `dmgfc-f`, `dmgfc-a`, and `dmgfc-latest`. Existing `dmfc.DMFC`,
`jbfc.JBFC`, and old method/command names remain compatibility aliases. The
[migration guide](dmgfc-public-api-migration-2026-10-24.md) lists the exact changes.
This API rename does not rewrite historical result IDs or source hashes.

## Naming audit (October 25, 2026)

Current README examples, package metadata, command help, manuscript labels and
README artwork use **DMGFC**. The paper build README now also points to
`dmgfc.DMGFC` and `dmgfc-eval`; its earlier wording incorrectly presented
`dmfc.DMFC` as the current API. Current-paper verification examples use DMGFC.
The repository's citation lists Minhyeong Yu, Wonduk Seo and Seunghyun Lee.

Remaining old names belong to the following preserved interfaces or records:

| Location | Why the old name remains |
|---|---|
| `dmfc.DMFC`, `dmfc-eval`, older method options and tool metadata | Backward compatibility for existing applications |
| `paired_dmfc_minus_direct` and prepared experiment metadata | Recorded result schemas and input fingerprints |
| Research scripts recorded in experiment manifests | Even a comment change would alter the source hashes used to verify completed experiments |
| Dated revision notes, archived manuscripts and PDFs | Historical names at the time of that revision; earlier linked notes are explicitly marked historical |
| Archived-paper and compatibility tests | Verification that earlier papers and import paths remain supported |

Use the current names for new code and documentation. Do not mass-replace
historical identifiers or alter completed experiment sources to remove a search
match.

## PDF and upload source

Build the current multi-file manuscript with `make paper`. Check its short-paper
structure with:

```bash
uv run --extra dev python scripts/verify_paper.py paper/main.pdf --format acm-short
```

The current upload archive is
[`paper/DMGFC-Overleaf-source-2026-10-24.zip`](../paper/DMGFC-Overleaf-source-2026-10-24.zip).
It contains the following eleven files, with `main.tex` at the ZIP root:

- `main.tex`
- `acmart.cls`
- `ACM-Reference-Format.bst`
- `references.bib`
- `sec/abstract.tex` (required empty input)
- `sec/framework.tex`
- `sec/introduction.tex`
- `sec/methodology.tex`
- `sec/experiment_setting_and_results.tex`
- `sec/conclusion.tex`
- `sec/repair_tradeoff.tex`

On October 24, 2026, the PDF was compiled locally with TeX Live 2025, and all four
pages were rendered and visually checked. The archive was unpacked into a
separate directory and compiled using only those eleven files. Its PDF text and
page rendering match the current manuscript. All manuscript changes are title or
method-label substitutions; experimental values and other body text are unchanged.

This remains the restored draft: red passages, `xxx`, and the existing unrun
multi-verifier claim in the Introduction are preserved. The `acm-short` check
verifies structure and layout; it does not certify submission readiness. The
stronger `acm-review` check still requires the Web/track relevance text removed
during the October 21 restoration.

## README artwork

The hero and mechanism diagram use `dmgfc-hero.png` and `dmgfc-framework.png`.
Chart measurements are unchanged; only their displayed method labels change.
Generate local chart previews without launching a browser with:

```bash
python3 docs/assets/src/charts.py docs/assets/src/data.json work/artwork --html-only
```

The updated PNGs were captured and visually checked using the Codex in-app
browser. The framework's example probabilities remain illustrative, as its
caption states; none are presented as experimental measurements.
