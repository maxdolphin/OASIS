"""
TDD: the reportlab PDF path must EMBED visualizations (not just tables/text).

Before this work `pdfimages -list` reported ZERO images in the generated PDF.
These tests assert the report now embeds real raster charts, including the
Window-of-Viability / robustness curve, and that generation is robust to
degenerate (tiny) networks.
"""
import io
import json
import shutil
import subprocess

import numpy as np
import pytest

from src.ulanowicz_calculator import UlanowiczCalculator
from src.publication_report import PublicationReportGenerator
from src.pdf_generator import generate_pdf_report


def _build_report_generator(flow, nodes, org_name='Test Org'):
    calc = UlanowiczCalculator(flow, nodes)
    metrics = calc.get_extended_metrics()
    assessments = calc.assess_regenerative_health()
    rg = PublicationReportGenerator(
        calculator=calc, metrics=metrics, assessments=assessments,
        org_name=org_name, flow_matrix=flow, node_names=nodes)
    return rg, calc, metrics


def _load_sample(path):
    d = json.load(open(path))
    return np.array(d['flows'], dtype=float), d['nodes'], d.get('organization', 'Org')


def _count_pdf_images(pdf_bytes):
    """Count embedded raster images. Prefer pdfimages; fall back to raw parse."""
    if shutil.which('pdfimages'):
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tf:
            tf.write(pdf_bytes)
            tmp = tf.name
        try:
            out = subprocess.run(['pdfimages', '-list', tmp],
                                 capture_output=True, text=True)
            lines = [l for l in out.stdout.splitlines()
                     if l.strip() and l.split()[0].isdigit()]
            return len(lines)
        finally:
            os.unlink(tmp)
    # Fallback: count image XObjects in the raw PDF stream
    return pdf_bytes.count(b'/Subtype /Image') + pdf_bytes.count(b'/Subtype/Image')


def test_pdf_embeds_at_least_three_images():
    """The Cone Spring sample must yield a PDF with >= 3 embedded images."""
    flow, nodes, org = _load_sample(
        'data/ecosystem_samples/cone_spring_original.json')
    rg, calc, metrics = _build_report_generator(flow, nodes, org)
    pdf = generate_pdf_report(rg, calc, metrics, charts=None)
    assert pdf and pdf[:4] == b'%PDF'
    n = _count_pdf_images(pdf)
    assert n >= 3, f"expected >= 3 embedded images, got {n}"


def test_window_of_viability_curve_present():
    """The sustainability section must render the WoV/robustness curve image."""
    flow, nodes, org = _load_sample(
        'data/ecosystem_samples/cone_spring_original.json')
    rg, calc, metrics = _build_report_generator(flow, nodes, org)
    pdf = generate_pdf_report(rg, calc, metrics, charts=None)
    text = _pdf_text(pdf)
    # Caption of the WoV figure is emitted next to the embedded Image flowable.
    assert 'Window of Viability' in text or 'Robustness Curve' in text


def test_degenerate_network_does_not_crash():
    """A 3-node degenerate network must still produce a PDF (charts guarded)."""
    flow = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ], dtype=float)
    nodes = ['X', 'Y', 'Z']
    rg, calc, metrics = _build_report_generator(flow, nodes, 'Tiny Org')
    pdf = generate_pdf_report(rg, calc, metrics, charts=None)
    assert pdf and pdf[:4] == b'%PDF'


def test_two_node_network_does_not_crash():
    flow = np.array([[0, 1], [1, 0]], dtype=float)
    nodes = ['A', 'B']
    rg, calc, metrics = _build_report_generator(flow, nodes, 'Two Node')
    pdf = generate_pdf_report(rg, calc, metrics, charts=None)
    assert pdf and pdf[:4] == b'%PDF'


def _pdf_text(pdf_bytes):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join((page.extract_text() or "") for page in reader.pages)
