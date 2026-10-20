# ACM manuscript draft

The active manuscript is `main.tex`, using the ACM files and section layout supplied on 2026-10-20.
The supplied abstract, Introduction, authors, red draft passages and ACM sample front matter are
preserved. Missing Related Work, Methodology, experiments, Conclusion, acknowledgements and
bibliography entries were copied from the prior manuscript. The copied manuscript uses DMFC labels;
the implementation and recorded experiment IDs still use JBFC.

The old manuscript is saved unchanged in [`versions/2026-10-18-jbfc/`](versions/2026-10-18-jbfc/):
`main.tex`, `main.pdf`, and `references.bib`, from repository commit `76521c8`.

From the repository root:

```sh
make paper   # export the ACM project with the installed LaTeX toolchain, including BibTeX
make verify  # verify sections, measured tables, research figures, and resolved references
make final   # regenerate the measured tables in sec/experiment_setting_and_results.tex
```

The draft intentionally retains the supplied `xxx` passage, references to multiple verifier models,
red review text, placeholder conference/CCS/keywords, running author label, sample teaser and received
dates. These need author review before submission. The experiments copied here use one verifier and
the BFCL subset. The old four-body-page budget is not asserted for this preserved ACM draft.

The native LaTeX editor remains the source-editing entry point. Its compiler currently cannot read
all local project files or the supplied ACM class version; `make paper` exports the PDF using the
existing local LaTeX installation instead.

Validation during the migration: the PDF builds with six total pages, with References on page six;
the ACM structure checks and 63 offline table-generation tests pass. A full `make final` rerun stopped
because existing repair-limit replay caches differ from the expected source/path policy. The recorded
trajectories and those caches were not replaced; all four migrated table/plot blocks retain the prior
manuscript's numeric values. The split-file writer is covered by an offline integration test.
