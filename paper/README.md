# Current manuscript

Read the latest paper in **[main.pdf](main.pdf)**. Its source entry point is
**[main.tex](main.tex)**; the Abstract is written directly there, and the other active
sections are loaded from `sec/`. `sec/abstract.tex` is unused. The `sigconf*.tex` files
are ACM template examples, not entry points for this paper.

The active upload manuscript is the **anonymous review version**:
`sigconf,anonymous,review`, with line numbers, ACM CCS and reference-format metadata,
and four total pages including references. Author names, affiliations, email addresses
and acknowledgements are omitted from the active source and PDF. The author-facing
version is preserved in [`versions/2026-10-20-author-review/`](versions/2026-10-20-author-review/).
Page one explains Web relevance for the Web Infrastructure and Agentic Systems track;
this is the working track choice until the authors confirm their submission track.

The 2026-10-20 Overleaf revision updates the Abstract and Introduction, integrates
related work into Methodology, and reports three generators from two providers
(GPT-5-nano, GPT-5.4-mini, and Claude Sonnet 5.5) with one verifier (`jev-1.13.0`).
Direct and DMFC are compared over three runs on BFCL and an exploratory 156-case
API-Bank first-call adaptation. The current PDF has **four total pages including
references**. The manuscript uses DMFC labels; recorded experiment IDs retain
historical names while the public API is `import dmfc`, `dmfc.DMFC`.

See the [comment resolution record](../docs/overleaf-comment-resolution-2026-10-20.md)
for completed changes, hidden screenshot text still to check, and remaining experiments.
API-Bank uses adapted call exact match, not the official execution score, and its
input instruction was refined after exploratory outcomes. Claude's BFCL accuracy
decrease is included in the Abstract, results, and conclusion.

Previous source/PDF copies are preserved under [`versions/`](versions/), including
the original JBFC paper, the ACM import, and the revision before the Overleaf comments.

From the repository root:

```sh
make paper   # export the ACM project with the installed LaTeX toolchain, including BibTeX
make verify  # verify sections, measured tables, research figures, and resolved references
make final   # derive recorded results; preserve the combined benchmark table
```

Table 1 is generated from the historical and added experiments by
`scripts/additional_paper_tables.py`. `make final` preserves it while updating the
historical single-run ablation and repair-limit plot. Recorded trajectories and
previous versions remain unchanged.

The native editor cannot compile every dependency in this multi-file ACM project;
`make paper` exports the PDF with the existing local LaTeX installation. `make verify`
checks the short-paper budget, required content, author leaks in text/metadata, and
first-page Web/track relevance. All four rendered pages are inspected. The DOI/ISBN
fields are blank pending assignment after acceptance; the class and bibliography
style are unchanged from the supplied ACM template. See the
[upload checklist](../docs/overleaf-upload-checklist-2026-10-20.md) for exact files and verification.
