# DM-Call naming (2026-11-03)

The method is now called **DM-Call**, and the manuscript title is
**DM-Call: Decision-Model-Guided Function Calling with Cumulative Verification Feedback**.

## Python library

| Use | New name | Earlier names that still work |
|---|---|---|
| Distribution | `dmcall` | `dmgfc` (until 2026-11-03), `dmfc` |
| Import | `import dmcall` | `import dmgfc`, `import dmfc`, `import jbfc` |
| Class | `dmcall.DMCall` | `dmgfc.DMGFC`, `dmfc.DMFC`, `jbfc.JBFC` |
| Methods | `dmcall`, `dmcall-f`, `dmcall-a`, `dmcall-latest` | `dmgfc*`, `dmfc*`, `jbfc*` and the recorded `jev_*` IDs |
| Evaluation command | `dmcall-eval`, `python -m dmcall` | `dmgfc-eval`, `dmfc-eval`, `jev-fc` |

`dmcall` re-exports the implementation in `src/dmgfc`, so every name above runs the same
code. `src/dmgfc` and the frozen engine `src/jev_fc` are left unchanged because recorded
experiments pin their source hashes. `DMCall` differs from `DMGFC` only in its default
method name (`"dmcall"`), which selects the same loop as `"dmgfc"`.

## Not renamed

- Recorded experiment IDs, result directories, manifests and historical reports keep their
  original names for reproducibility.
- The README images (`docs/assets/*.png`) were rendered before the rename and still show
  DMGFC. Their sources in `docs/assets/src/` now say DM-Call; re-render them with Playwright
  and the Inter font to update the PNGs.
- The legacy manuscript pipeline under `paper/` and `scripts/` keeps its DMGFC labels. The
  current manuscript is maintained separately and uses DM-Call.
