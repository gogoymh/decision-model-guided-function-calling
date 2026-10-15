"""Export the self-contained manuscript as PDF with an existing TeX installation.

The native Codex LaTeX editor is the default editing and preview environment.
This optional script exports a portable PDF file; no TeX installation is performed.
"""

import shutil
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
engine = shutil.which("pdflatex")
if not engine:
    raise SystemExit("pdflatex was not found. Open paper/main.tex in the native LaTeX editor and export its PDF.")
for _ in range(2):
    result = subprocess.run([engine, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                            cwd=root / "paper", capture_output=True, text=True)
    if result.returncode:
        print(result.stdout[-6000:])
        raise SystemExit(result.returncode)
print(root / "paper" / "main.pdf")
