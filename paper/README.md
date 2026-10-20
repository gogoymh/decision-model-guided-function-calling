# Current manuscript

Read the latest paper in **[main.pdf](main.pdf)**. Its source entry point is
**[main.tex](main.tex)**; the Abstract is written directly there, and the other active
sections are loaded from `sec/`. `sec/abstract.tex` is unused. The `sigconf*.tex` files
are ACM template examples, not entry points for this paper.

The 2026-10-20 Overleaf revision updates the Abstract and Introduction, integrates
related work into Methodology, and reports three generators from two providers
(GPT-5-nano, GPT-5.4-mini, and Claude Sonnet 5.5) with one verifier (`jev-1.13.0`).
Direct and DMFC are compared over three runs on BFCL and an exploratory 156-case
API-Bank first-call adaptation. The current PDF has **four total pages including
references**. The manuscript uses DMFC labels; implementation and recorded experiment
IDs retain JBFC/Jev names.

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
`make paper` exports the PDF with the existing local LaTeX installation. The Overleaf
revision passed 748 offline tests and PDF verification; all four rendered pages were
checked. This is an author-facing copy. Anonymous review formatting and submission
track requirements still need a separate check.
