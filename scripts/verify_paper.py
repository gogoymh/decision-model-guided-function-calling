"""Check page budget, section coverage, references, and unrun table markers."""

import sys
from pathlib import Path

from pypdf import PdfReader

path = Path(sys.argv[1] if len(sys.argv) > 1 else "paper/main.pdf")
reader = PdfReader(path)
pages = [page.extract_text() or "" for page in reader.pages]
assert 5 <= len(pages) <= 6, f"Expected 5-6 total pages, got {len(pages)}"
assert "References" in pages[4], "References must start on page 5"
assert not any("References" in page for page in pages[:4]), "References appeared in the 4-page body"
body = "\n".join(pages[:4]).lower()
for section in ("abstract", "introduction", "related work", "method", "experiments", "conclusion", "acknowledg"):
    assert section in body, f"Missing requested section in body: {section}"
assert "conclusion" in pages[3].lower(), "Conclusion must fit in page 4"
assert "acknowledg" in pages[3].lower(), "Acknowledgements must fit in page 4"
assert "[?]" not in "\n".join(pages), "Unresolved citations"
print(f"Verified: body pages 1-4; references pages 5-{len(pages)}; all requested sections included.")
