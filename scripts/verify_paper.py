"""Check the original, imported ACM draft, or four-page ACM short-paper format."""

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
    assert any("cumulative prompt feedback" in caption
               or "comparison of evaluated mechanisms" in caption for caption in captions), \
        "Missing feedback-loop figure"
    assert any("accuracy (%) by repair limit" in caption
               or ("accuracy" in caption and "api time" in caption) for caption in captions), \
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


def verify_acm_short(pages):
    assert 1 <= len(pages) <= 4, f"Expected at most 4 total pages including references, got {len(pages)}"
    text = "\n".join(pages)
    references = re.search(r"(?im)^\s*(?:\d+\s+)?references\s*$", text)
    assert references, "Missing References heading"
    body = text[:references.start()]
    assert not re.search(r"(?im)^\s*(?:\d+(?:\.\d+)*\s+)?related\s+work\s*$", body), \
        "ACM short paper must not have a separate Related Work section"
    headings = {
        "Abstract": r"abstract",
        "Introduction": r"introduction",
        "Methodology": r"methods?|methodology",
        "Experiments": r"experiments?|experiment(?:al)?\s+(?:setting|setup)\s*(?:&|and)\s*results",
        "Conclusion": r"conclusions?",
    }
    for name, pattern in headings.items():
        assert re.search(rf"(?im)^\s*(?:\d+(?:\.\d+)*\s+)?(?:{pattern})\s*$", body), \
            f"Missing requested section heading: {name}"
    normalized = " ".join(body.lower().split())
    # ACM separates caption numbers with a colon; a prose reference such as "Table 2." is not a table.
    tables = re.findall(r"\btable\s+(\d+)\s*:", normalized)
    assert sorted(tables) == ["1", "2"], "ACM short paper must contain exactly Tables 1 and 2"
    first_page = " ".join(pages[0].lower().split())
    first_figure = re.search(r"\bfigure\s+1\s*:\s*(.*?)(?=\bfigure\s+\d+\s*:|$)", first_page)
    assert first_figure and "comparison" in first_figure[1][:900] and "schematic" in first_figure[1][:900], \
        "Missing first-page comparison schematic (Figure 1)"
    figures = re.findall(r"\bfigure\s+(\d+)\s*:\s*(.*?)(?=\bfigure\s+\d+\s*:|$)", normalized)
    assert any(number != "1" and "accuracy" in caption[:900] and "api time" in caption[:900]
               for number, caption in figures), "Missing accuracy/API-time plot"
    assert re.search(r"\balgorithm\s+1\b", normalized), "Missing Algorithm 1"
    assert not re.search(r"\[\s*\?(?:\s*[,;]\s*\?)*\s*\]|\b(?:table|figure|algorithm)\s*\?+", text,
                         re.IGNORECASE), "Unresolved citations or cross-references"
    assert re.search(r"\[\d+(?:\s*[,;\u2013-]\s*\d+)*\]", body), "No resolved numeric citations in body"
    assert "not yet run" not in normalized and "unfilled" not in normalized, "Unfilled result tables"
    reference_page = next(index for index, page in enumerate(pages, 1)
                          if re.search(r"(?im)^\s*(?:\d+\s+)?references\s*$", page))
    print(f"Verified ACM short paper: {len(pages)} total pages including references "
          f"(starting on page {reference_page}); required sections, two tables, first-page comparison, "
          "accuracy/API-time plot, Algorithm 1, and citations present.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="?", default="paper/main.pdf", type=Path)
    parser.add_argument("--format", choices=("original", "acm-draft", "acm-short"), default="original")
    args = parser.parse_args(argv)
    pages = [page.extract_text() or "" for page in PdfReader(args.pdf).pages]
    if args.format == "acm-short":
        verify_acm_short(pages)
    elif args.format == "acm-draft":
        verify_acm_draft(pages)
    else:
        verify_original(pages)


if __name__ == "__main__":
    main()
