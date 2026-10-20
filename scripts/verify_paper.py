"""Check the original page budget or an ACM draft's structure and rendered research objects."""

import argparse
import re
from pathlib import Path

from pypdf import PdfReader


def verify_original(pages):
    assert 5 <= len(pages) <= 6, f"Expected 5-6 total pages, got {len(pages)}"
    assert "References" in pages[4], "References must start on page 5"
    assert not any("References" in page for page in pages[:4]), "References appeared in the 4-page body"
    body = "\n".join(pages[:4]).lower()
    sections = ("abstract", "introduction", "related work", "method", "experiments", "conclusion", "acknowledg")
    for section in sections:
        assert section in body, f"Missing requested section in body: {section}"
    assert "conclusion" in pages[3].lower(), "Conclusion must fit in page 4"
    assert "acknowledg" in pages[3].lower(), "Acknowledgements must fit in page 4"
    assert "[?]" not in "\n".join(pages), "Unresolved citations"
    print(f"Verified: body pages 1-4; references pages 5-{len(pages)}; all requested sections included.")


def verify_acm_draft(pages):
    text = "\n".join(pages)
    references = re.search(r"(?im)^\s*(?:\d+\s+)?references\s*$", text)
    assert references, "Missing References heading"
    body = text[:references.start()]
    normalized = " ".join(body.lower().split())
    for section in ("abstract", "introduction", "related work", "method", "conclusion", "acknowledg"):
        assert section in normalized, f"Missing requested section in body: {section}"
    assert re.search(r"\bexperiments?\b|\bexperiment setting\s*(?:&|and)\s*results\b", normalized), \
        "Missing Experiments or Experiment Setting & Results section"
    for number in (1, 2, 3):
        assert re.search(rf"\btable\s+{number}\s*[:.]", normalized), f"Missing Table {number} caption"
    # ACM keeps the supplied teaser as Figure 1; check the research captions by their content.
    captions = re.findall(r"\bfigure\s+\d+\s*[:.]\s*(.*?)(?=\bfigure\s+\d+\s*[:.]|$)", normalized)
    assert any("cumulative prompt feedback" in caption for caption in captions), \
        "Missing feedback-loop figure"
    assert any("accuracy (%) by repair limit" in caption for caption in captions), \
        "Missing repair-limit figure"
    assert re.search(r"\balgorithm\s+1\b", normalized), "Missing Algorithm 1"
    assert not re.search(r"\[\s*\?(?:\s*[,;]\s*\?)*\s*\]", text), "Unresolved citations or references"
    assert re.search(r"\[\d+(?:\s*[,;\u2013-]\s*\d+)*\]", body), "No resolved numeric citations in body"
    assert "not yet run" not in normalized and "unfilled" not in normalized, "Unfilled result tables"
    reference_page = next(index for index, page in enumerate(pages, 1)
                          if re.search(r"(?im)^\s*(?:\d+\s+)?references\s*$", page))
    print(f"Verified ACM draft: {len(pages)} total pages; references start on page {reference_page}; "
          "sections, Tables 1-3, both research figures, Algorithm 1, and citations present. "
          "The original 4-page body budget is not applied to this draft.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="?", default="paper/main.pdf", type=Path)
    parser.add_argument("--format", choices=("original", "acm-draft"), default="original")
    args = parser.parse_args(argv)
    pages = [page.extract_text() or "" for page in PdfReader(args.pdf).pages]
    if args.format == "acm-draft":
        verify_acm_draft(pages)
    else:
        verify_original(pages)


if __name__ == "__main__":
    main()
