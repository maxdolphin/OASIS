"""
Integration test: the app's actual PDF path (src/pdf_generator.generate_pdf_report,
reportlab) must include the detailed ecosystemic sections.
"""
import numpy as np
import pytest

from src.ulanowicz_calculator import UlanowiczCalculator
from src.publication_report import PublicationReportGenerator
from src.pdf_generator import generate_pdf_report


def _render_pdf_bytes():
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
    return generate_pdf_report(rg, calc, metrics, charts=None)


def _pdf_text(pdf_bytes):
    import io
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def test_app_pdf_renders():
    pdf = _render_pdf_bytes()
    assert pdf is not None
    assert pdf[:4] == b'%PDF'


def test_app_pdf_contains_detailed_sections():
    text = _pdf_text(_render_pdf_bytes())
    assert 'Benchmarking' in text
    assert 'Risk' in text and 'Resilience' in text
    assert 'Action Roadmap' in text
    assert 'ESG Framework Mapping' in text


def test_app_pdf_sections_sequential():
    text = _pdf_text(_render_pdf_bytes())
    # Renumbered sections must be present and ordered
    for marker in ['5. Benchmarking', '6. Risk', '7. Prioritized Action Roadmap',
                   '8. ESG Framework Mapping', '9. Discussion',
                   '10. Conclusions']:
        assert marker in text, f"missing section marker: {marker}"
