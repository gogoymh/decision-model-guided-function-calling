"""PDF checks distinguish the original page budget from the ACM short-paper requirements."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_paper", ROOT / "scripts/verify_paper.py")
verify_paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify_paper)


@pytest.fixture
def short_pages():
    return [
        "ABSTRACT\n1 Introduction\nCited work [1, 2].\n"
        "Figure 1: Comparison of evaluated mechanisms (schematic; not measured results).",
        "2 Methodology\nAlgorithm 1 DMGFC: verification and prompt repair\n"
        "3 Experiment Setting & Results\nTable 1: Main experiment.\nTable 2: Ablation study.",
        "Figure 2: DMGFC accuracy and API time by repair limit; r=3 matches Table 2.\n"
        "4 Conclusion\nReferences\n[1] A paper.",
    ]


def test_short_budget_includes_references(short_pages, capsys):
    verify_paper.verify_acm_short([*short_pages[:2], "", short_pages[2]])
    assert "4 total pages including references (starting on page 4)" in capsys.readouterr().out
    with pytest.raises(AssertionError, match="at most 4 total pages"):
        verify_paper.verify_acm_short([*short_pages[:2], "", "", short_pages[2]])


@pytest.mark.parametrize("old,new,message", [
    ("2 Methodology", "", "Missing requested section heading: Methodology"),
    ("3 Experiment Setting & Results", "", "Missing requested section heading: Experiments"),
    ("4 Conclusion", "", "Missing requested section heading: Conclusion"),
    ("Table 2: Ablation study.", "", "exactly Tables 1 and 2"),
    ("Table 2: Ablation study.", "Table 2: Ablation study.\nTable 3: Old rules table.", "exactly Tables 1 and 2"),
    ("Figure 2: DMGFC accuracy and API time by repair limit; r=3 matches Table 2.", "",
     "Missing accuracy/API-time plot"),
    ("Algorithm 1 DMGFC", "Algorithm unavailable DMGFC", "Missing Algorithm 1"),
    ("Cited work [1, 2].", "Cited work [?].", "Unresolved citations"),
    ("Cited work [1, 2].", "See Table ?? and cited work [1].", "Unresolved citations"),
    ("Cited work [1, 2].", "No numbered citation.", "No resolved numeric citations"),
])
def test_short_rejects_missing_or_unresolved_content(short_pages, old, new, message):
    pages = [page.replace(old, new) for page in short_pages]
    with pytest.raises(AssertionError, match=message):
        verify_paper.verify_acm_short(pages)


def test_comparison_must_be_on_first_page(short_pages):
    figure = "Figure 1: Comparison of evaluated mechanisms (schematic; not measured results)."
    short_pages[0] = short_pages[0].replace(figure, "")
    short_pages[1] += "\n" + figure
    with pytest.raises(AssertionError, match="Missing first-page comparison schematic"):
        verify_paper.verify_acm_short(short_pages)


def test_related_work_prose_allowed_but_separate_section_rejected(short_pages):
    short_pages[0] += "\nWe discuss related work within the introduction."
    verify_paper.verify_acm_short(short_pages)
    short_pages[1] += "\n2 Related Work\n"
    with pytest.raises(AssertionError, match="separate Related Work"):
        verify_paper.verify_acm_short(short_pages)


def test_original_format_keeps_its_four_body_page_budget():
    pages = ["Abstract\nIntroduction\nRelated Work\nMethod\nExperiments\n", "", "",
             "Conclusion\nAcknowledgements\n", "References\n"]
    verify_paper.verify_original(pages)
    with pytest.raises(AssertionError, match="Expected 5-6 total pages"):
        verify_paper.verify_original(pages[:4])
    with pytest.raises(AssertionError, match="References must start on page 5"):
        verify_paper.verify_original([pages[0], "References", "", pages[3], ""])


@pytest.mark.parametrize("framework,tradeoff", [
    ("JBFC and cumulative prompt feedback", "JBFC accuracy (%) by repair limit"),
    ("Comparison of evaluated mechanisms (schematic)", "DMFC accuracy and API time"),
])
def test_imported_draft_accepts_archived_and_revised_figure_captions(framework, tradeoff):
    pages = ["Abstract\nIntroduction\nRelated Work\nMethodology\nExperiments\n"
             "Table 1: Categories\nTable 2: Methods\nTable 3: Rules\n"
             f"Figure 2: {framework}.\nFigure 3: {tradeoff}.\nAlgorithm 1 DMFC\nCited [1].",
             "", "", "", "", "Conclusion\nAcknowledgements\nReferences\n[1] A paper."]
    verify_paper.verify_acm_draft(pages)


@pytest.fixture
def anonymous_pages(short_pages):
    short_pages[0] = ("Anonymous Author(s)\nWeb agents need verification before dispatch. "
                      "Web Infrastructure and Agentic Systems.\n" + short_pages[0])
    return short_pages


def test_anonymous_review_accepts_nonidentifying_pdf(anonymous_pages):
    verify_paper.verify_anonymous_review(anonymous_pages, {"/Author": "Anonymous Author(s)"})


@pytest.mark.parametrize("identity", ["Minhyeong Yu", "Wonduk Seo", "Enhans", "University College London",
                                     "Pseudo Lab", "No-More-Slops"])
def test_anonymous_review_rejects_author_leaks(anonymous_pages, identity):
    anonymous_pages[1] += "\n" + identity
    with pytest.raises(AssertionError, match="Author-identifying text"):
        verify_paper.verify_anonymous_review(anonymous_pages)


def test_anonymous_review_rejects_metadata_leak(anonymous_pages):
    with pytest.raises(AssertionError, match="PDF metadata"):
        verify_paper.verify_anonymous_review(anonymous_pages, {"/Author": "Minhyeong Yu"})


def test_anonymous_review_rejects_missing_first_page_track(anonymous_pages):
    anonymous_pages[0] = anonymous_pages[0].replace("Web Infrastructure and Agentic Systems.", "")
    with pytest.raises(AssertionError, match="selected-track relevance"):
        verify_paper.verify_anonymous_review(anonymous_pages)


def test_anonymous_review_rejects_acknowledgements(anonymous_pages):
    anonymous_pages[1] += "\nAcknowledgments\nSupport details."
    with pytest.raises(AssertionError, match="Acknowledgements"):
        verify_paper.verify_anonymous_review(anonymous_pages)
