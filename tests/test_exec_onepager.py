"""
TDD for R8 (executive one-pager) + R9 (credibility keystone to the front).

The FIRST content page after the cover must be a self-contained executive
one-pager carrying, in order:
  1. a reconciled headline verdict (the CAPPED OASIS status, not the raw mean);
  2. KPI cards with reference anchors + the alpha gradient position;
  3. an embedded Window-of-Viability chart (the marquee visual);
  4. top-3 risks in Evidence -> Implication form;
  5. prioritized next steps (roadmap, time-horizoned);
then a clear "Detailed analysis follows" divider.

R9: a "Why this applies to your organization" keystone paragraph must appear on
the front matter (cover or first exec page), led by the organizational evidence.

Honesty guardrail: the exec summary must NOT render a bare absolute-fail
"Non-Viable" / "UNSUSTAINABLE" verdict (the gradient reframe already shipped).
"""
import io
import json

import numpy as np
import pytest

from src.ulanowicz_calculator import UlanowiczCalculator
from src.oasis_calculator import OASISCalculator
from src.publication_report import PublicationReportGenerator
from src.pdf_generator import generate_pdf_report


# Three contrasting sample orgs (one capped-status org, one balanced, one viable).
SAMPLE_ORGS = [
    'data/synthetic_organizations/combined_flows/tech_company_combined_matrix.json',
    'data/synthetic_organizations/combined_flows/balanced_org_test.json',
    'data/ecosystem_samples/cone_spring_original.json',
]

# A capped org: SUSTAINABLE vetoes the raw HEALTHY mean down to WARNING.
CAPPED_ORG = 'data/synthetic_organizations/combined_flows/tech_company_combined_matrix.json'


def _load_sample(path):
    d = json.load(open(path))
    return (np.array(d['flows'], dtype=float),
            d['nodes'],
            d.get('organization', 'Org'))


def _build(flow, nodes, org_name):
    calc = UlanowiczCalculator(flow, nodes)
    metrics = calc.get_extended_metrics()
    assessments = calc.assess_regenerative_health()
    profile = OASISCalculator(calc).get_oasis_profile()
    rg = PublicationReportGenerator(
        calculator=calc, metrics=metrics, assessments=assessments,
        org_name=org_name, flow_matrix=flow, node_names=nodes,
        oasis_profile=profile)
    return rg, calc, metrics, profile


def _pdf_text(pdf_bytes):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return [(page.extract_text() or "") for page in reader.pages]


def _count_pdf_images(pdf_bytes):
    return (pdf_bytes.count(b'/Subtype /Image')
            + pdf_bytes.count(b'/Subtype/Image'))


def _render(path):
    flow, nodes, org = _load_sample(path)
    rg, calc, metrics, profile = _build(flow, nodes, org)
    pdf = generate_pdf_report(rg, calc, metrics, charts=None)
    assert pdf and pdf[:4] == b'%PDF'
    return pdf, profile


# ---------------------------------------------------------------------------
# The executive one-pager front section
# ---------------------------------------------------------------------------

def _front_text(pdf_bytes):
    """Text of the cover + first exec content page (the one-pager lives here)."""
    pages = _pdf_text(pdf_bytes)
    # cover (page 0) + the exec one-pager page(s) that precede the divider
    joined = "\n".join(pages[:3])
    return joined


def test_exec_onepager_has_reconciled_capped_verdict():
    pdf, profile = _render(CAPPED_ORG)
    front = _front_text(pdf)
    # The capped status must be the stated verdict (not the raw mean's label).
    assert profile['overall_status_capped'] is True
    assert profile['overall_status'] in front            # WARNING (capped)
    assert 'capped' in front.lower()
    # A dimension name that drove the cap must be named.
    assert any(d.upper() in front.upper() for d in profile['capped_by'])


def test_exec_onepager_has_alpha_gradient_position():
    pdf, profile = _render(SAMPLE_ORGS[0])
    front = _front_text(pdf)
    # Gradient framing, never a bare pass/fail. One of the three positions shows.
    assert any(p in front.lower()
               for p in ('under-organized', 'over-organized', 'balanced'))
    assert 'α' in front or 'alpha' in front.lower()


def test_exec_onepager_embeds_wov_image_on_front():
    pdf, _ = _render(SAMPLE_ORGS[0])
    pages = _pdf_text(pdf)
    front = "\n".join(pages[:3])
    # The marquee WoV visual caption appears in the exec summary.
    assert 'Window of Viability' in front or 'Robustness' in front
    # And there is at least one embedded raster image in the whole PDF.
    assert _count_pdf_images(pdf) >= 1


def test_exec_onepager_has_top_risk_line():
    pdf, _ = _render(SAMPLE_ORGS[0])
    front = _front_text(pdf)
    assert 'Evidence' in front and 'Implication' in front


def test_exec_onepager_has_next_step():
    pdf, _ = _render(SAMPLE_ORGS[0])
    front = _front_text(pdf)
    # A time-horizoned next step (roadmap) appears in the one-pager.
    assert ('Next Steps' in front or 'Action' in front)
    assert any(h in front for h in
               ('Immediate', 'Short-Term', 'Short-term', 'Medium-Term',
                'Medium-term', '0–3', '0-3'))


def test_exec_onepager_has_divider():
    pdf, _ = _render(SAMPLE_ORGS[0])
    front = _front_text(pdf)
    assert 'Detailed analysis follows' in front


def test_keystone_on_front_matter():
    pdf, _ = _render(SAMPLE_ORGS[0])
    front = _front_text(pdf)
    assert 'Why this applies to your organization' in front
    # Led by the organizational evidence (Fath 2019), with the indicative caveat.
    assert 'Fath' in front
    assert 'indicative' in front.lower() or 'directional' in front.lower()


def test_no_bare_absolute_fail_verdict_in_exec():
    pdf, _ = _render(CAPPED_ORG)
    front = _front_text(pdf).lower()
    # The reframe forbids bare absolute-fail organizational verdicts.
    assert 'non-viable' not in front
    assert 'unsustainable' not in front


# ---------------------------------------------------------------------------
# Robustness: degenerate-safe + all three sample orgs still build with images
# ---------------------------------------------------------------------------

def test_degenerate_network_onepager_safe():
    flow = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    nodes = ['X', 'Y', 'Z']
    rg, calc, metrics, _ = _build(flow, nodes, 'Tiny Org')
    pdf = generate_pdf_report(rg, calc, metrics, charts=None)
    assert pdf and pdf[:4] == b'%PDF'


@pytest.mark.parametrize('path', SAMPLE_ORGS)
def test_three_sample_orgs_build_with_images(path):
    pdf, _ = _render(path)
    assert _count_pdf_images(pdf) >= 1
