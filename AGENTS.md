# Research repository instructions

Keep paper/main.tex and the implementation consistent. The current manuscript is
an unrun experimental protocol, not a completed empirical claim.

- Never fabricate benchmark values or promote synthetic/demo outputs into a paper.
- Never send ground_truth, category labels, example IDs, offline scores or gold-derived feedback to models.
- Keep the pinned BFCL inputs and official evaluator unchanged; document annotation anomalies.
- Three repairs means four proposals including the first. Re-evaluate both checks on each candidate.
- Only the final dispatch boundary can invoke an application callback. Benchmark runs do not execute tools.
- Never print, commit or package API keys or .env. Keep .env.example blank.
- Run appropriate offline tests after code changes. Use the native LaTeX editor for the standalone manuscript,
  and verify the exported PDF has four body pages with references beginning on page five (six pages maximum).
