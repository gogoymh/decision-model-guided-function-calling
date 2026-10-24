# Decision-Model-Guided Function Calling

Read the latest paper in **[main.pdf](main.pdf)**. Its source entry point is
**[main.tex](main.tex)**; the original Abstract is written directly there and also
loads the supplied empty `sec/abstract.tex`. Other sections are loaded from `sec/`.
The `sigconf*.tex` files are ACM template examples.

On **2026-10-21**, the user requested restoration of text already supplied in the
original WWW ZIP. The Abstract block, full Introduction, and Methodology's
`Task Definition` heading are restored exactly. At that revision, the title and original bibliography
entry were unchanged. The originally empty method/results/conclusion
sections retain the added material and measured results.

The original red text, `xxx`, and unrun multi-verifier claim in the Introduction
are preserved at the user's request. Actual experiments use one Jev verifier.
Claude and exploratory API-Bank results remain in the experiment section and
Claude's BFCL decrease remains in the conclusion. The frozen Abstract and
Introduction do not include these later results. Do not silently rewrite them.

On **2026-10-24**, the current title was shortened to **Decision-Model-Guided
Function Calling**, and its displayed method labels became **DMGFC**, **DMGFC-F**,
**DMGFC-A**, and **DMGFC-L**. This naming-only revision preserves all other
restored Abstract and Introduction wording, experiment data, and archived papers.

The current manuscript keeps `sigconf,anonymous,review`, ACM CCS and WWW 2027
metadata, and omits author identities and acknowledgements. It is a **four-page
restored draft**, including references, rather than a submission-ready manuscript.
The earlier Web/track paragraph was removed as part of restoring the entire
Introduction. The previous four-page revision is preserved at
[`versions/2026-10-21-before-polished-text-restore/`](versions/2026-10-21-before-polished-text-restore/),
and the author-facing revision at
[`versions/2026-10-20-author-review/`](versions/2026-10-20-author-review/).

The current upload ZIP, `DMGFC-Overleaf-source-2026-10-24.zip`, contains **11 source
files**, including empty `sec/abstract.tex`. It is rebuilt and checked from its own
files. See [current naming and build notes](../docs/dmgfc-naming-2026-10-24.md); the
[earlier restoration checklist](../docs/overleaf-polished-text-restore-2026-10-21.md)
records the original text restoration.
API-Bank remains an exploratory 156-case first-call exact-match adaptation,
rather than the official execution benchmark. The public display name is DMGFC;
the compatible Python API remains
`import dmfc`, `dmfc.DMFC`, with historical experiment IDs preserved.

Previous source/PDF copies are preserved under [`versions/`](versions/), including
the original JBFC paper, the ACM import, and the revision before the Overleaf comments.

From the repository root:

```sh
make paper   # export the ACM project with the installed LaTeX toolchain, including BibTeX
make verify  # strict submission check; the restored draft lacks the added Web/track paragraph
make final   # derive recorded results; preserve the combined benchmark table
```

Table 1 is generated from the historical and added experiments by
`scripts/additional_paper_tables.py`. `make final` preserves it while updating the
historical single-run ablation and repair-limit plot. Recorded trajectories and
previous versions remain unchanged.

The native editor cannot compile every dependency in this multi-file ACM project;
`make paper` exports the PDF with the existing local LaTeX installation. `make verify`
checks the short-paper budget, required content, author leaks in text/metadata, and
first-page Web/track relevance. The restored draft has four rendered pages. The strict
submission check still fails because the original Introduction lacks the added
Web/track paragraph. The DOI/ISBN
fields are blank pending assignment after acceptance; the class and bibliography
style are unchanged from the supplied ACM template. See the
[upload checklist](../docs/overleaf-polished-text-restore-2026-10-21.md) for exact files and verification.
