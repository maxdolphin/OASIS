import numpy as np
from src.ulanowicz_calculator import UlanowiczCalculator
from src.oasis_calculator import OASISCalculator
from src.oasis_pdf_report import OASISPDFReport


def _build_report(detailed=True):
    flow = np.array([
        [0, 10, 0, 0, 5],
        [0, 0, 8, 2, 0],
        [0, 0, 0, 7, 1],
        [3, 0, 0, 0, 6],
        [0, 4, 0, 0, 0],
    ], dtype=float)
    nodes = ['A', 'B', 'C', 'D', 'E']
    uc = UlanowiczCalculator(flow, nodes)
    oc = OASISCalculator(uc)
    return OASISPDFReport(
        org_name='Test Org',
        oasis_profile=oc.get_oasis_profile(),
        ulanowicz_metrics=uc.get_extended_metrics(),
        interpretations=oc.get_oasis_interpretation(),
        recommendations=oc.get_recommendations(),
        detailed=detailed,
    )


def test_detailed_report_contains_new_sections():
    html = _build_report(detailed=True).generate_html()
    assert 'Benchmarking' in html
    assert 'Risk &amp; Resilience' in html or 'Risk & Resilience' in html
    assert 'Action Roadmap' in html
    assert 'Framework Mapping' in html or 'ESG' in html


def test_lean_report_excludes_new_sections():
    html = _build_report(detailed=False).generate_html()
    assert 'Action Roadmap' not in html


def test_convenience_function_detailed_default(tmp_path):
    from src.oasis_pdf_report import generate_oasis_pdf_report

    flow = np.array([[0, 10, 5], [2, 0, 8], [6, 1, 0]], dtype=float)
    uc = UlanowiczCalculator(flow, ['A', 'B', 'C'])
    oc = OASISCalculator(uc)
    out = tmp_path / "r.pdf"
    generate_oasis_pdf_report(oc, uc, org_name='X', output_path=str(out))
    html = (tmp_path / "r.html").read_text()
    assert 'Action Roadmap' in html  # detailed=True is the default
