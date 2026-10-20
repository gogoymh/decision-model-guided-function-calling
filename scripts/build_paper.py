"""Export the manuscript, including local section files and BibTeX references, as PDF.

The native Codex LaTeX editor is the default editing and preview environment.
This optional script exports a portable PDF file; no TeX installation is performed.
"""

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(command, paper_dir):
    result = subprocess.run(command, cwd=paper_dir, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        print((result.stdout + result.stderr)[-6000:])
        raise SystemExit(result.returncode)


def build(paper_dir=None):
    paper_dir = Path(paper_dir) if paper_dir is not None else ROOT / "paper"
    options = ["-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape"]
    latexmk = shutil.which("latexmk")
    if latexmk:
        _run([latexmk, "-pdf", *options, "main.tex"], paper_dir)
    else:
        engine = shutil.which("pdflatex")
        if not engine:
            raise SystemExit("pdflatex was not found. Open paper/main.tex in the native LaTeX editor "
                             "and export its PDF.")
        command = [engine, *options, "main.tex"]
        _run(command, paper_dir)
        # Inline thebibliography documents need no BibTeX; ACM section files can use a .bib database.
        if r"\bibdata{" in (paper_dir / "main.aux").read_text():
            bibtex = shutil.which("bibtex")
            if not bibtex:
                raise SystemExit("bibtex was not found; the manuscript's bibliography cannot be exported.")
            _run([bibtex, "main"], paper_dir)
        for _ in range(2):
            _run(command, paper_dir)
    path = paper_dir / "main.pdf"
    print(path)
    return path


if __name__ == "__main__":
    build()
