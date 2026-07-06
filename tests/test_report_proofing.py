"""
Credibility / "unproofed draft" defect tests for the OASIS PDF report.

These guard against the class of errors a client CFO spots instantly:
  1. Mis-numbered subsection headings (Discussion 9.x, Conclusions 10.x).
  2. Table of Contents entries that do not match real body headings.
  3. Raw code identifiers (relative_ascendency, number_of_roles, ...) leaking
     into user-facing recommendation / narrative text.
  4. Benchmark framing that leads with ecological wetlands instead of the
     Fath (2019) organizational anchor (alpha 0.30-0.45).
"""
import io
import re

import numpy as np
import pytest

from src.ulanowicz_calculator import UlanowiczCalculator
from src.publication_report import PublicationReportGenerator
from src.pdf_generator import generate_pdf_report


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _make_generator():
    flow = np.array([
        [0, 10, 0, 0, 5],
        [0, 0, 8, 2, 0],
        [0, 0, 0, 7, 1],
        [3, 0, 0, 0, 6],
        [0, 4, 0, 0, 0],
    ], dtype=float)
    nodes = ['A', 'B', 'C', 'D', 'E']
    calc = UlanowiczCalculator(flow, nodes)
    metrics = calc.get_extended_metrics()
    assessments = calc.assess_regenerative_health()
    rg = PublicationReportGenerator(
        calculator=calc, metrics=metrics, assessments=assessments,
        org_name='Test Org', flow_matrix=flow, node_names=nodes)
    return rg, calc, metrics


def _render_pdf_text():
    rg, calc, metrics = _make_generator()
    pdf = generate_pdf_report(rg, calc, metrics, charts=None)
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


# ---------------------------------------------------------------------------
# 1. Subsection heading numbering
# ---------------------------------------------------------------------------
def test_discussion_subheadings_are_9x():
    rg, _, _ = _make_generator()
    disc = rg.generate_discussion()
    # No leaked earlier-section numbering
    assert not re.search(r'(?m)^\s*4\.\d\s', disc), \
        "Discussion still contains 4.x subsection headers"
    # Real parent is section 9
    assert re.search(r'(?m)^\s*9\.1\s', disc)
    assert re.search(r'(?m)^\s*9\.2\s', disc)
    assert re.search(r'(?m)^\s*9\.3\s', disc)


def test_conclusions_subheadings_are_10x():
    rg, _, _ = _make_generator()
    conc = rg.generate_conclusions()
    assert not re.search(r'(?m)^\s*5\.\d\s', conc), \
        "Conclusions still contains 5.x subsection headers"
    assert re.search(r'(?m)^\s*10\.1\s', conc)
    assert re.search(r'(?m)^\s*10\.2\s', conc)
    assert re.search(r'(?m)^\s*10\.3\s', conc)


# ---------------------------------------------------------------------------
# 2. Table of Contents matches the real body
# ---------------------------------------------------------------------------
def test_toc_has_no_phantom_entries():
    from src.pdf_generator import build_toc_items
    titles = [t for t, _ in build_toc_items()]
    joined = " | ".join(titles)
    assert "Network Structure" not in joined
    assert "System Organization" not in joined


def test_toc_entries_match_body_headings():
    """Every TOC entry must correspond to an actual heading rendered in the body."""
    from src.pdf_generator import build_toc_items, BODY_HEADINGS
    toc_titles = {t.strip() for t, _ in build_toc_items()}
    body = {h.strip() for h in BODY_HEADINGS}
    missing = toc_titles - body
    assert not missing, f"TOC entries with no matching body heading: {missing}"


def test_toc_body_headings_appear_in_rendered_pdf():
    text = _render_pdf_text()
    from src.pdf_generator import BODY_HEADINGS
    # Top-level numbered sections must literally appear in the rendered PDF.
    for h in BODY_HEADINGS:
        if re.match(r'^\d+\.\s', h) or h in (
                'Executive Summary', 'References'):
            token = h.split('&')[0].strip()[:18]
            assert token in text, f"body heading not found in PDF: {h!r}"


# ---------------------------------------------------------------------------
# 3. No leaked metric identifiers in user-facing text
# ---------------------------------------------------------------------------
LEAKED = ['relative_ascendency', 'number_of_roles', 'finn_cycling_index',
          'regenerative_capacity', 'flow_diversity', 'gini_coefficient',
          'clustering_coefficient', 'flow_reciprocity', 'mutualism_ratio',
          'functional_diversity', 'overhead_ratio', 'connectance']


def test_no_leaked_identifiers_in_pdf_text():
    text = _render_pdf_text()
    hits = [ident for ident in LEAKED if ident in text]
    assert not hits, f"Raw metric identifiers leaked into report text: {hits}"


def test_recommendation_metrics_humanized():
    from src.pdf_generator import humanize_metric_name
    assert humanize_metric_name('number_of_roles') == 'number of functional roles'
    assert 'relative ascendency' in humanize_metric_name('relative_ascendency')
    assert '_' not in humanize_metric_name('finn_cycling_index')


# ---------------------------------------------------------------------------
# 4. Benchmark framing: Fath organizational anchor is primary
# ---------------------------------------------------------------------------
def test_benchmark_leads_with_org_anchor():
    text = _render_pdf_text()
    # Fath organizational anchor present in benchmark framing
    assert 'Fath' in text
    assert '0.30' in text and '0.45' in text
    # Ecological anchors must be captioned as illustrative reference points.
    assert re.search(r'illustrative', text, re.IGNORECASE)
