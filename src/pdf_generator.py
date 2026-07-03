"""
Professional PDF Report Generator
Generates publication-quality PDF reports using reportlab.
Designed for organizational network analysis based on Ulanowicz-Fath framework.
"""

import base64
import re
from io import BytesIO
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


def _pdf_gradient(alpha):
    """Gradient classifier (position + direction-of-travel) — single source of
    truth from report_intelligence. Reframes the old binary viability verdict."""
    try:
        import report_intelligence as _ri
    except ImportError:  # pragma: no cover
        from src import report_intelligence as _ri
    return _ri.assess_alpha_position(alpha)


# ── Color palette ──────────────────────────────────────────────────────────
FOREST_GREEN = '#1a5f35'
MEDIUM_GREEN = '#2d8a4e'
ACCENT_GREEN = '#2ecc71'
GOLD = '#d4a843'
TEAL = '#48c9b0'
DARK_TEXT = '#1a1a2e'
BODY_TEXT = '#333333'
MUTED = '#666666'
LIGHT_BG = '#f8faf9'
TABLE_HEADER_BG = '#1a5f35'
TABLE_ALT_ROW = '#f0f5f2'
STATUS_HEALTHY = '#2ecc71'
STATUS_WARNING = '#f5b041'
STATUS_CRITICAL = '#e74c3c'

# OASIS dimension colors (lighter tints for print)
OASIS_COLORS = {
    'open': '#3498db',
    'autonomous': '#9b59b6',
    'symbiotic': '#27ae60',
    'intelligent': '#e67e22',
    'sustainable': '#1abc9c',
}


# ── Human-readable metric labels ───────────────────────────────────────────
# Maps raw code identifiers (dict keys) to reader-facing labels. Used ONLY for
# display text; dict keys/access are never changed.
_METRIC_LABELS = {
    'relative_ascendency': 'relative ascendency (α)',
    'ascendency_ratio': 'relative ascendency (α)',
    'number_of_roles': 'number of functional roles',
    'functional_diversity': 'functional diversity',
    'finn_cycling_index': 'resource cycling (Finn cycling index)',
    'flow_reciprocity': 'flow reciprocity',
    'regenerative_capacity': 'regenerative capacity',
    'flow_diversity': 'flow diversity',
    'connectance': 'network connectance',
    'clustering_coefficient': 'clustering coefficient',
    'gini_coefficient': 'resource distribution (Gini coefficient)',
    'mutualism_ratio': 'mutualism ratio',
    'robustness': 'robustness',
    'redundancy': 'pathway redundancy',
    'overhead_ratio': 'reserve overhead',
}


def humanize_metric_name(name):
    """Convert a raw metric identifier into a reader-facing label.

    Falls back to a title-cased, underscore-stripped version for any key not in
    the explicit map, guaranteeing no raw ``snake_case`` identifier reaches the
    reader.
    """
    if not isinstance(name, str):
        return str(name)
    key = name.strip()
    if key in _METRIC_LABELS:
        return _METRIC_LABELS[key]
    return key.replace('_', ' ').strip()


def build_toc_items():
    """Table-of-Contents entries mirroring the ACTUAL body headings, in order.

    Kept in sync with :data:`BODY_HEADINGS`; every entry here must correspond to
    a heading rendered in the report body.
    """
    return [
        ("Executive Summary", ""),
        ("1. Introduction", ""),
        ("2. Methodology", ""),
        ("3. Results", ""),
        ("   3.1 Core Network Metrics", ""),
        ("   3.2 Sustainability Assessment", ""),
        ("   3.3 Visualizations", ""),
        ("   3.4 Flow Distribution Analysis", ""),
        ("4. OASIS Organizational Health Assessment", ""),
        ("5. Benchmarking & Position", ""),
        ("6. Risk & Resilience Analysis", ""),
        ("7. Prioritized Action Roadmap", ""),
        ("8. ESG Framework Mapping", ""),
        ("9. Discussion", ""),
        ("10. Conclusions & Recommendations", ""),
        ("References", ""),
        ("Appendix: Detailed Data", ""),
    ]


# Canonical list of body headings actually rendered (top-level + subsections),
# used by the TOC and by proofing tests to guarantee TOC↔body consistency.
BODY_HEADINGS = [t.strip() for t, _ in build_toc_items()]


def _hex_to_rgb(hex_color):
    """Convert hex color string to reportlab Color."""
    from reportlab.lib.colors import HexColor
    return HexColor(hex_color)


def _get_status_color(status):
    """Return color for a status string."""
    s = status.upper() if isinstance(status, str) else ''
    if 'HEALTHY' in s or 'VIABLE' in s or 'HIGH' in s or 'OPTIMAL' in s or 'STRONG' in s:
        return STATUS_HEALTHY
    elif 'WARNING' in s or 'MODERATE' in s or 'DEVELOPING' in s:
        return STATUS_WARNING
    elif 'CRITICAL' in s or 'LOW' in s or 'NON' in s:
        return STATUS_CRITICAL
    return MUTED


def generate_pdf_report(report_generator, calculator, metrics, charts=None):
    """
    Generate a professional PDF report using reportlab.

    Args:
        report_generator: PublicationReportGenerator instance
        calculator: RegenerativeMetricsCalculator instance
        metrics: Dictionary of calculated metrics
        charts: Optional dictionary of plotly figures to include

    Returns:
        bytes: PDF file content, or None on failure
    """
    try:
        return _build_reportlab_pdf(report_generator, calculator, metrics, charts)
    except Exception as e:
        # Fallback: try simple PDF
        try:
            full_text = report_generator.generate_full_report()
            return create_simple_pdf(full_text, report_generator.org_name)
        except Exception:
            return None


def _build_reportlab_pdf(report_generator, calculator, metrics, charts=None):
    """Build a professionally formatted PDF with reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm, mm
    from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib import colors
    from reportlab.platypus import (
        BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
        PageBreak, Table, TableStyle, Image, KeepTogether, HRFlowable,
        NextPageTemplate,
    )
    from reportlab.graphics.shapes import Drawing, Rect, String, Line
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.pdfgen import canvas as pdfcanvas

    PAGE_W, PAGE_H = A4
    MARGIN_L = 2.2 * cm
    MARGIN_R = 2.2 * cm
    MARGIN_T = 2.5 * cm
    MARGIN_B = 2.5 * cm
    CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R

    buffer = BytesIO()
    org_name = report_generator.org_name
    timestamp = datetime.now().strftime('%B %d, %Y')

    # ── Custom styles ────────────────────────────────────────────────────
    styles = getSampleStyleSheet()

    s_title = ParagraphStyle(
        'ReportTitle', parent=styles['Title'],
        fontName='Helvetica-Bold', fontSize=28, leading=34,
        textColor=_hex_to_rgb(FOREST_GREEN),
        alignment=TA_CENTER, spaceAfter=16,
    )
    s_subtitle = ParagraphStyle(
        'ReportSubtitle', parent=styles['Normal'],
        fontName='Helvetica', fontSize=16, leading=20,
        textColor=_hex_to_rgb(MUTED),
        alignment=TA_CENTER, spaceAfter=8,
    )
    s_cover_detail = ParagraphStyle(
        'CoverDetail', parent=styles['Normal'],
        fontName='Helvetica', fontSize=11, leading=14,
        textColor=_hex_to_rgb(MUTED),
        alignment=TA_CENTER, spaceAfter=4,
    )
    s_h1 = ParagraphStyle(
        'H1', parent=styles['Heading1'],
        fontName='Helvetica-Bold', fontSize=20, leading=24,
        textColor=_hex_to_rgb(FOREST_GREEN),
        spaceBefore=24, spaceAfter=12,
        keepWithNext=1,
    )
    s_h2 = ParagraphStyle(
        'H2', parent=styles['Heading2'],
        fontName='Helvetica-Bold', fontSize=15, leading=19,
        textColor=_hex_to_rgb(MEDIUM_GREEN),
        spaceBefore=18, spaceAfter=8,
        keepWithNext=1,
    )
    s_h3 = ParagraphStyle(
        'H3', parent=styles['Heading3'],
        fontName='Helvetica-Bold', fontSize=12, leading=15,
        textColor=_hex_to_rgb(DARK_TEXT),
        spaceBefore=12, spaceAfter=6,
        keepWithNext=1,
    )
    s_body = ParagraphStyle(
        'Body', parent=styles['BodyText'],
        fontName='Times-Roman', fontSize=10.5, leading=14,
        textColor=_hex_to_rgb(BODY_TEXT),
        alignment=TA_JUSTIFY, spaceAfter=8,
    )
    s_body_italic = ParagraphStyle(
        'BodyItalic', parent=s_body,
        fontName='Times-Italic',
    )
    s_caption = ParagraphStyle(
        'Caption', parent=styles['Normal'],
        fontName='Helvetica-Oblique', fontSize=9, leading=11,
        textColor=_hex_to_rgb(MUTED),
        alignment=TA_CENTER, spaceAfter=12, spaceBefore=4,
    )
    s_kpi_value = ParagraphStyle(
        'KPIValue', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=22, leading=26,
        alignment=TA_CENTER, spaceAfter=2,
    )
    s_kpi_label = ParagraphStyle(
        'KPILabel', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8.5, leading=11,
        textColor=_hex_to_rgb(MUTED),
        alignment=TA_CENTER, spaceAfter=2,
    )
    s_kpi_status = ParagraphStyle(
        'KPIStatus', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=8, leading=10,
        alignment=TA_CENTER, spaceAfter=0,
    )
    s_reference = ParagraphStyle(
        'Reference', parent=s_body,
        fontSize=9.5, leading=12,
        leftIndent=24, firstLineIndent=-24,
        spaceAfter=6,
    )
    s_table_note = ParagraphStyle(
        'TableNote', parent=styles['Normal'],
        fontName='Helvetica-Oblique', fontSize=8.5, leading=10,
        textColor=_hex_to_rgb(MUTED),
        spaceAfter=6,
    )

    # ── Header / Footer drawing ─────────────────────────────────────────
    def _draw_header_footer(cvs, doc):
        """Draw page header and footer on every content page."""
        cvs.saveState()
        # Header line
        cvs.setStrokeColor(_hex_to_rgb(FOREST_GREEN))
        cvs.setLineWidth(1.5)
        cvs.line(MARGIN_L, PAGE_H - MARGIN_T + 8, PAGE_W - MARGIN_R, PAGE_H - MARGIN_T + 8)
        # Header text
        cvs.setFont('Helvetica', 8)
        cvs.setFillColor(_hex_to_rgb(MUTED))
        cvs.drawString(MARGIN_L, PAGE_H - MARGIN_T + 12,
                        f"Organizational Network Analysis — {org_name}")
        cvs.drawRightString(PAGE_W - MARGIN_R, PAGE_H - MARGIN_T + 12, timestamp)
        # Footer
        cvs.setStrokeColor(_hex_to_rgb(FOREST_GREEN))
        cvs.setLineWidth(0.5)
        cvs.line(MARGIN_L, MARGIN_B - 10, PAGE_W - MARGIN_R, MARGIN_B - 10)
        cvs.setFont('Helvetica', 8)
        cvs.setFillColor(_hex_to_rgb(MUTED))
        cvs.drawString(MARGIN_L, MARGIN_B - 22,
                        "Based on Ulanowicz-Fath Regenerative Economics Framework")
        cvs.drawRightString(PAGE_W - MARGIN_R, MARGIN_B - 22,
                            f"Page {doc.page}")
        cvs.restoreState()

    def _draw_cover(cvs, doc):
        """Draw cover page background (no header/footer)."""
        cvs.saveState()
        # Top accent bar
        cvs.setFillColor(_hex_to_rgb(FOREST_GREEN))
        cvs.rect(0, PAGE_H - 6 * mm, PAGE_W, 6 * mm, fill=True, stroke=False)
        # Bottom accent bar
        cvs.setFillColor(_hex_to_rgb(GOLD))
        cvs.rect(0, 0, PAGE_W, 3 * mm, fill=True, stroke=False)
        # Thin green line below gold
        cvs.setFillColor(_hex_to_rgb(FOREST_GREEN))
        cvs.rect(0, 3 * mm, PAGE_W, 1 * mm, fill=True, stroke=False)
        cvs.restoreState()

    # ── Document template ────────────────────────────────────────────────
    cover_frame = Frame(
        MARGIN_L, MARGIN_B, CONTENT_W, PAGE_H - MARGIN_T - MARGIN_B,
        id='cover',
    )
    content_frame = Frame(
        MARGIN_L, MARGIN_B, CONTENT_W, PAGE_H - MARGIN_T - MARGIN_B,
        id='content',
    )

    doc = BaseDocTemplate(
        buffer, pagesize=A4,
        leftMargin=MARGIN_L, rightMargin=MARGIN_R,
        topMargin=MARGIN_T, bottomMargin=MARGIN_B,
        title=f"Network Analysis Report — {org_name}",
        author="Adaptive Organization Analysis System",
    )

    doc.addPageTemplates([
        PageTemplate(id='cover', frames=[cover_frame], onPage=_draw_cover),
        PageTemplate(id='content', frames=[content_frame], onPage=_draw_header_footer),
    ])

    # ── Light-theme color sequence for PDF charts ───────────────────────
    _PDF_COLORS = [
        '#1a5f35', '#2d8a4e', '#3498db', '#e67e22', '#9b59b6',
        '#1abc9c', '#c0392b', '#2c3e50', '#d4a843', '#7f8c8d',
    ]

    # ── Helper to convert Plotly fig to reportlab Image ──────────────────
    def _chart_image(fig, width=CONTENT_W, height=280):
        """Convert a Plotly figure to a reportlab Image flowable."""
        try:
            import copy
            fig_copy = go.Figure(fig)

            # Force light theme layout
            fig_copy.update_layout(
                template='plotly_white',
                paper_bgcolor='white',
                plot_bgcolor='#fafafa',
                font=dict(color='#333333'),
                margin=dict(l=70, r=30, t=60, b=50),
                legend=dict(font=dict(color='#333333')),
            )

            # Override trace-level colors for light background readability
            for i, trace in enumerate(fig_copy.data):
                color = _PDF_COLORS[i % len(_PDF_COLORS)]
                if trace.type == 'pie':
                    # Pie charts: assign full color sequence and dark text
                    trace.marker = dict(
                        colors=_PDF_COLORS[:len(trace.labels)] if trace.labels else _PDF_COLORS,
                        line=dict(color='white', width=2),
                    )
                    trace.textfont = dict(color='#333333')
                    trace.outsidetextfont = dict(color='#333333')
                elif trace.type == 'heatmap':
                    trace.colorscale = 'Greens'
                elif trace.type in ('bar', 'scatter', 'scatterpolar'):
                    # Keep existing colors if they are already light-friendly,
                    # otherwise only fix text/font colors
                    if hasattr(trace, 'textfont'):
                        trace.textfont = dict(color='#333333')

            # Fix axis colors for light background
            fig_copy.update_xaxes(
                color='#333333', gridcolor='#e0e0e0',
                linecolor='#999999', zerolinecolor='#cccccc',
            )
            fig_copy.update_yaxes(
                color='#333333', gridcolor='#e0e0e0',
                linecolor='#999999', zerolinecolor='#cccccc',
            )

            # Render at higher pixel resolution for sharpness
            render_w = max(int(width * 2), 900)
            render_h = max(int(height * 2), 500)
            img_bytes = pio.to_image(fig_copy, format='png',
                                     width=render_w, height=render_h, scale=2)
            img_buf = BytesIO(img_bytes)
            return Image(img_buf, width=width, height=height)
        except Exception as _e:
            import logging
            logging.getLogger(__name__).warning(
                "PDF chart (plotly/kaleido) export failed, skipping: %s", _e)
            return None

    def _mpl_image(fig, width=CONTENT_W, height=280, dpi=150):
        """Convert a matplotlib Figure to a reportlab Image flowable.

        Used for charts that already have a native matplotlib builder (e.g. the
        Window-of-Viability curve) and as a kaleido-free fallback path.
        """
        try:
            img_buf = BytesIO()
            fig.savefig(img_buf, format='png', dpi=dpi,
                        bbox_inches='tight', facecolor='white')
            img_buf.seek(0)
            try:
                import matplotlib.pyplot as _plt
                _plt.close(fig)
            except Exception:
                pass
            return Image(img_buf, width=width, height=height)
        except Exception as _e:
            import logging
            logging.getLogger(__name__).warning(
                "PDF chart (matplotlib) export failed, skipping: %s", _e)
            return None

    def _guarded_chart_block(builder, caption, story_list,
                             heading=None, heading_style=None):
        """Build one chart via *builder* (returns a reportlab Image or None),
        wrap it with a caption, and append as a KeepTogether block.

        Each chart is individually guarded so one failure logs a warning and is
        skipped rather than aborting the whole PDF. Returns True if embedded.
        """
        img = None
        try:
            img = builder()
        except Exception as _e:
            import logging
            logging.getLogger(__name__).warning(
                "PDF chart builder raised, skipping: %s", _e)
            img = None
        if img is None:
            return False
        block = []
        if heading is not None:
            block.append(Paragraph(heading, heading_style or s_h2))
        block.append(img)
        block.append(Paragraph(caption, s_caption))
        story_list.append(KeepTogether(block))
        return True

    # ── Build story ──────────────────────────────────────────────────────
    story = []

    # ════════════════════════════════════════════════════════════════════
    # COVER PAGE
    # ════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 3.5 * cm))
    story.append(Paragraph("ORGANIZATIONAL<br/>NETWORK ANALYSIS", s_title))
    story.append(Spacer(1, 0.3 * cm))
    # Green horizontal rule
    story.append(HRFlowable(
        width='60%', thickness=2,
        color=_hex_to_rgb(FOREST_GREEN),
        spaceBefore=4, spaceAfter=12,
    ))
    story.append(Paragraph(org_name, s_subtitle))
    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph(
        "A Quantitative Assessment Using<br/>"
        "Ulanowicz-Fath Regenerative Economics Principles", s_cover_detail))
    story.append(Spacer(1, 0.8 * cm))
    story.append(Paragraph(
        f"Report Date: {timestamp}", s_cover_detail))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Adaptive Organization Analysis System", s_cover_detail))

    # Quick stats box on cover
    story.append(Spacer(1, 2 * cm))
    n_nodes = len(calculator.node_names)
    n_edges = int(np.count_nonzero(calculator.flow_matrix))
    viability = _pdf_gradient(metrics.get('relative_ascendency',
                              metrics.get('ascendency_ratio', 0)))['position']
    cover_data = [
        ['Network Nodes', 'Active Connections', 'Gradient Position', 'Robustness'],
        [str(n_nodes), str(n_edges), viability, f"{metrics.get('robustness', 0):.3f}"],
    ]
    cover_table = Table(cover_data, colWidths=[CONTENT_W / 4] * 4)
    cover_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('TEXTCOLOR', (0, 0), (-1, 0), _hex_to_rgb(MUTED)),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, 1), 14),
        ('TEXTCOLOR', (0, 1), (-1, 1), _hex_to_rgb(FOREST_GREEN)),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LINEBELOW', (0, 0), (-1, 0), 0.5, _hex_to_rgb(FOREST_GREEN)),
    ]))
    story.append(cover_table)

    story.append(NextPageTemplate('content'))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════
    # EXECUTIVE ONE-PAGER  (R8 + R9)
    # A self-contained, demo-ready first content page composed of five
    # elements, in order:
    #   0. The credibility keystone (R9) — "Why this applies to your org".
    #   1. Reconciled headline verdict — the CAPPED OASIS status + capped_by.
    #   2. KPI cards with reference anchors (gradient framing, never bare fail).
    #   3. The marquee "you are here" Window-of-Viability curve.
    #   4. Top-3 risks in Evidence -> Implication form.
    #   5. Prioritized next steps (roadmap, time-horizoned).
    # Then a clear "Detailed analysis follows" divider; the existing detailed
    # sections continue after it.
    # All inputs are the precomputed oasis_profile + report_intelligence views;
    # nothing here recomputes a metric.
    # ════════════════════════════════════════════════════════════════════

    # Shared intelligence module + the precomputed profile (read, do not recompute).
    try:
        import report_intelligence as _ri_exec
    except ImportError:  # pragma: no cover
        from src import report_intelligence as _ri_exec

    _exec_profile = getattr(report_generator, 'oasis_profile', None)
    if not (isinstance(_exec_profile, dict) and 'dimension_scores' in _exec_profile):
        _exec_profile = None
    if _exec_profile is None:
        try:
            from oasis_calculator import OASISCalculator as _OC_exec
        except Exception:
            try:
                from src.oasis_calculator import OASISCalculator as _OC_exec
            except Exception:
                _OC_exec = None
        if _OC_exec is not None:
            try:
                _exec_profile = _OC_exec(calculator).get_oasis_profile()
            except Exception:
                _exec_profile = None

    rob = metrics.get('robustness', 0)
    rob_status = _ri_exec.categorize_robustness_label(rob)
    eff = metrics.get('network_efficiency', 0)
    eff_status = _ri_exec.categorize_efficiency_label(eff)
    alpha = metrics.get('ascendency_ratio', 0)
    _grad_exec = _ri_exec.assess_alpha_position(alpha)

    story.append(Paragraph("Executive Summary", s_h1))
    story.append(HRFlowable(
        width='100%', thickness=1, color=_hex_to_rgb(FOREST_GREEN),
        spaceBefore=0, spaceAfter=10,
    ))

    # ── 0. Credibility keystone (R9): "Why this applies to your organization" ──
    #    Lead with the ORGANIZATIONAL evidence (Fath 2019), not wetlands; frame
    #    the window as an indicative directional reference (honesty guardrail).
    s_keystone = ParagraphStyle(
        'Keystone', parent=s_body, fontSize=9.5, leading=13,
        textColor=_hex_to_rgb(DARK_TEXT),
        leftIndent=6, rightIndent=6, spaceBefore=2, spaceAfter=8,
        borderColor=_hex_to_rgb(TEAL), borderWidth=0.5, borderPadding=5,
        backColor=_hex_to_rgb('#f4fbf9'),
    )
    story.append(Paragraph(
        "<b>Why this applies to your organization.</b> High-performing "
        "organizations analyzed with this same efficiency&ndash;resilience "
        "framework cluster in a characteristic range (relative ascendency "
        "&alpha; &asymp; 0.30&ndash;0.45; Fath et al., 2019, regenerative "
        "economics). OASIS reads how your organization is <i>structurally "
        "wired</i> &mdash; the balance between coordinating efficiency and "
        "adaptive reserve computed from real flow data &mdash; a network lens "
        "that <i>complements</i>, and does not replace, culture and engagement "
        "measures. The viability band is an <b>indicative, directional</b> "
        "reference (calibrated on ecological systems; organizational "
        "calibration is an open question), so read your position as a "
        "direction of travel, not a compliance grade.",
        s_keystone))

    # ── 1. Reconciled headline verdict — capped status + business meaning ──
    if _exec_profile is not None:
        _overall = float(_exec_profile.get('overall_score', 0.0))
        _capped_status = str(_exec_profile.get('overall_status', 'UNKNOWN'))
        _capped = bool(_exec_profile.get('overall_status_capped', False))
        _capped_by = _exec_profile.get('capped_by', []) or []
        _verdict_color = _get_status_color(_capped_status)
        if _capped and _capped_by:
            _cap_names = ', '.join(d.capitalize() for d in _capped_by)
            _headline = (
                f"<font color=\"{_verdict_color}\"><b>{_capped_status} "
                f"&mdash; {_overall:.0f}/100</b></font>, capped by a critical "
                f"<b>{_cap_names}</b> dimension."
            )
            _sowhat = (
                "So what: the overall label is held below its raw average "
                "because a core pillar is critical &mdash; a weak pillar cannot "
                "be averaged away, and it sets the near-term priority."
            )
        else:
            _headline = (
                f"<font color=\"{_verdict_color}\"><b>{_capped_status} "
                f"&mdash; {_overall:.0f}/100</b></font>."
            )
            _sowhat = (
                "So what: no single dimension is critical; the priority is to "
                "hold the balance and address the weakest pillar before it "
                "drifts."
            )
        _s_verdict = ParagraphStyle(
            'Verdict', parent=s_body, fontSize=13, leading=17,
            spaceBefore=2, spaceAfter=2, alignment=TA_LEFT)
        story.append(Paragraph("Headline Verdict", ParagraphStyle(
            'vh', parent=s_h3, spaceBefore=2, spaceAfter=2)))
        story.append(Paragraph(_headline, _s_verdict))
        story.append(Paragraph(_sowhat, ParagraphStyle(
            's', parent=s_body_italic, fontSize=9.5, leading=12, spaceAfter=8)))

    # ── 2. KPI cards with reference anchors (gradient framing) ──
    def _kpi_cell(label, value, status, anchor, color=None):
        c = color or _get_status_color(status)
        return [
            Paragraph(str(value), ParagraphStyle(
                'kv', parent=s_kpi_value, textColor=_hex_to_rgb(c))),
            Paragraph(label, s_kpi_label),
            Paragraph(status, ParagraphStyle(
                'ks', parent=s_kpi_status, textColor=_hex_to_rgb(c))),
            Paragraph(anchor, ParagraphStyle(
                'ka', parent=s_kpi_label, fontSize=7.2, leading=8.5,
                textColor=_hex_to_rgb(MUTED))),
        ]

    # α gradient position drives the alpha card's status word (never "Non-Viable").
    _alpha_pos = {
        'under-organized': 'Under-organized',
        'over-organized': 'Over-organized',
        'balanced': 'Balanced',
    }[_grad_exec['position']]
    _overall_kpi = (f"{_exec_profile.get('overall_score', 0):.0f}"
                    if _exec_profile is not None else '—')
    _overall_status_kpi = (str(_exec_profile.get('overall_status', ''))
                           if _exec_profile is not None else '')

    kpi_cells = [
        _kpi_cell('OASIS Overall', _overall_kpi, _overall_status_kpi,
                  'HEALTHY ≥ 60 / WARNING ≥ 40 / else CRITICAL'),
        _kpi_cell('Rel. Ascendency (α)', f"{alpha:.3f}", _alpha_pos,
                  'indicative band 0.2–0.6; high-perf. orgs 0.30–0.45'),
        _kpi_cell('Robustness (R)', f"{rob:.3f}", rob_status,
                  'peaks ≈0.37 (=1/e); High ≥ 0.25'),
        _kpi_cell('Efficiency', f"{eff:.3f}", eff_status,
                  'balanced 0.2–0.6; high = brittle'),
    ]

    # Flatten into table rows (4 rows per card: value/label/status/anchor).
    kpi_data = [[cell[i] for cell in kpi_cells] for i in range(4)]
    kpi_table = Table(kpi_data, colWidths=[CONTENT_W / 4] * 4)
    kpi_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('BOX', (0, 0), (0, -1), 0.5, _hex_to_rgb('#e0e0e0')),
        ('BOX', (1, 0), (1, -1), 0.5, _hex_to_rgb('#e0e0e0')),
        ('BOX', (2, 0), (2, -1), 0.5, _hex_to_rgb('#e0e0e0')),
        ('BOX', (3, 0), (3, -1), 0.5, _hex_to_rgb('#e0e0e0')),
        ('BACKGROUND', (0, 0), (-1, -1), _hex_to_rgb('#fafcfb')),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 0.35 * cm))

    # ── 3. The marquee "you are here" Window-of-Viability curve ──
    def _build_exec_wov_image():
        try:
            png = _ri_exec.render_window_of_viability_png(alpha, rob)
        except Exception:
            return None
        if not png:
            return None
        try:
            _w = CONTENT_W * 0.72
            return Image(BytesIO(png), width=_w, height=_w * 4.0 / 7.2)
        except Exception:
            return None

    _guarded_chart_block(
        _build_exec_wov_image,
        "<i>You are here.</i> The organization's position (red marker) on the "
        "robustness curve R(&alpha;) = &minus;&alpha;&middot;ln(&alpha;), with "
        "the indicative reference band shaded. " + _grad_exec['caveat'],
        story)

    # ── 4. Top-3 risks (Evidence -> Implication) ──
    story.append(Paragraph("Top Risks", ParagraphStyle(
        'trh', parent=s_h3, spaceBefore=4, spaceAfter=3)))
    _exec_risks_rendered = False
    if _exec_profile is not None:
        try:
            _risk_view = _ri_exec.build_risk_view(metrics, _exec_profile)
            for _it in _risk_view['items'][:3]:
                _sc = _get_status_color(_it['severity'])
                story.append(Paragraph(
                    f"<font color=\"{_sc}\"><b>{_it['severity']}</b></font> "
                    f"&mdash; {_it['title']}. "
                    f"<b>Evidence:</b> {_it['evidence']} "
                    f"<b>Implication:</b> {_it['implication']}",
                    ParagraphStyle('rk', parent=s_body, fontSize=9, leading=11.5,
                                   spaceBefore=1, spaceAfter=3,
                                   leftIndent=10, firstLineIndent=-10)))
                _exec_risks_rendered = True
        except Exception:
            _exec_risks_rendered = False
    if not _exec_risks_rendered:
        story.append(Paragraph(
            "<b>Evidence:</b> risk view unavailable for this network. "
            "<b>Implication:</b> see the detailed Risk &amp; Resilience section.",
            ParagraphStyle('rk0', parent=s_body, fontSize=9, leading=11.5,
                           spaceAfter=3)))

    # ── 5. Prioritized next steps (roadmap, time-horizoned) ──
    story.append(Paragraph("Prioritized Next Steps", ParagraphStyle(
        'nsh', parent=s_h3, spaceBefore=4, spaceAfter=3)))
    _exec_steps_rendered = False
    if _exec_profile is not None:
        try:
            try:
                from oasis_calculator import OASISCalculator as _OC_rec
            except Exception:
                from src.oasis_calculator import OASISCalculator as _OC_rec
            _exec_recs = _exec_profile.get('recommendations')
            if _exec_recs is None:
                _exec_recs = _OC_rec(calculator).get_recommendations()
            _exec_roadmap = _ri_exec.build_action_roadmap(_exec_recs, _exec_profile)
            _horizon_labels = [
                ('immediate', 'Immediate (0–3 mo)'),
                ('short_term', 'Short-Term (3–9 mo)'),
                ('medium_term', 'Medium-Term (9–18 mo)'),
            ]
            _flat_steps = []
            for _hkey, _hlabel in _horizon_labels:
                for _st in _exec_roadmap.get(_hkey, []):
                    _flat_steps.append((_hlabel, _st))
            for _hlabel, _st in _flat_steps[:3]:
                _pc = _get_status_color(_st.get('priority', ''))
                story.append(Paragraph(
                    f"<font color=\"{_pc}\"><b>{_hlabel}</b></font> &middot; "
                    f"<b>{_st.get('dimension', '')}</b> &mdash; "
                    f"{_st.get('action', '')}",
                    ParagraphStyle('ns', parent=s_body, fontSize=9, leading=11.5,
                                   spaceBefore=1, spaceAfter=3,
                                   leftIndent=10, firstLineIndent=-10)))
                _exec_steps_rendered = True
        except Exception:
            _exec_steps_rendered = False
    if not _exec_steps_rendered:
        story.append(Paragraph(
            "<b>Immediate (0–3 mo)</b> &middot; Establish a recurring "
            "assessment cadence and confirm the reconciled verdict with "
            "leadership before acting.",
            ParagraphStyle('ns0', parent=s_body, fontSize=9, leading=11.5,
                           spaceAfter=3)))

    # ── Divider: analyst depth gated behind this line ──
    story.append(Spacer(1, 0.2 * cm))
    story.append(HRFlowable(
        width='100%', thickness=1.2, color=_hex_to_rgb(GOLD),
        dash=(3, 2), spaceBefore=1, spaceAfter=3))
    story.append(Paragraph(
        "— Detailed analysis follows —",
        ParagraphStyle('divider', parent=s_caption, fontSize=10,
                       textColor=_hex_to_rgb(MEDIUM_GREEN), spaceAfter=2)))

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════
    # TABLE OF CONTENTS (simple)
    # ════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Table of Contents", s_h1))
    story.append(HRFlowable(
        width='100%', thickness=1, color=_hex_to_rgb(FOREST_GREEN),
        spaceBefore=0, spaceAfter=16,
    ))
    toc_items = build_toc_items()
    for item, _ in toc_items:
        indent = 24 if item.startswith('   ') else 0
        toc_style = ParagraphStyle(
            'toc', parent=s_body,
            fontName='Helvetica' if indent else 'Helvetica-Bold',
            fontSize=10.5 if indent else 11,
            leftIndent=indent,
            spaceAfter=4,
        )
        story.append(Paragraph(item.strip(), toc_style))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════
    # HELPER: Parse plain text sections into paragraphs
    # ════════════════════════════════════════════════════════════════════
    def _render_text_section(text, story_list):
        """Convert plain text with headers and bullets into styled paragraphs."""
        lines = text.strip().split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()

            # Skip decorative lines
            if line.startswith('====') or line.startswith('----') or line.startswith('────'):
                i += 1
                continue

            # Skip the section number/title lines that match s_h1/s_h2 patterns
            # (we add those manually)
            if not line.strip():
                i += 1
                continue

            # Sub-section headers like "2.3 Analytical Measures"
            if re.match(r'^\d+\.\d+(\.\d+)?\s', line.strip()):
                story_list.append(Paragraph(line.strip(), s_h3))
                i += 1
                continue

            # Section headers like "1. INTRODUCTION" or "ABSTRACT"
            if line.strip().isupper() and len(line.strip()) < 60:
                i += 1
                continue

            # Table detection — lines with consistent spacing or "Table N."
            if line.strip().startswith('Table ') and '.' in line:
                story_list.append(Paragraph(f"<b>{line.strip()}</b>", s_table_note))
                i += 1
                # Collect table lines
                table_lines = []
                while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('Table '):
                    if lines[i].startswith('---') or lines[i].startswith('==='):
                        i += 1
                        continue
                    table_lines.append(lines[i])
                    i += 1
                if table_lines:
                    _render_text_table(table_lines, story_list)
                continue

            # Bullet points
            if line.strip().startswith('- ') or line.strip().startswith('• '):
                bullet_text = line.strip()[2:]
                # Handle markdown bold
                bullet_text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', bullet_text)
                bullet_style = ParagraphStyle(
                    'bullet', parent=s_body,
                    leftIndent=18, firstLineIndent=-12,
                    spaceBefore=2, spaceAfter=2,
                )
                story_list.append(Paragraph(f"•  {bullet_text}", bullet_style))
                i += 1
                continue

            # Numbered items (1. 2. etc)
            m = re.match(r'^(\d+)\.\s+(.+)', line.strip())
            if m and len(line.strip()) < 120:
                num_text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', m.group(2))
                num_style = ParagraphStyle(
                    'numbered', parent=s_body,
                    leftIndent=18, firstLineIndent=-18,
                    spaceBefore=3, spaceAfter=3,
                )
                # Collect continuation lines
                full_text = num_text
                while i + 1 < len(lines) and lines[i + 1].strip() and \
                        not re.match(r'^\d+\.?\s', lines[i + 1].strip()) and \
                        not lines[i + 1].strip().startswith('-') and \
                        not lines[i + 1].strip().startswith('Table'):
                    i += 1
                    cont = lines[i].strip()
                    cont = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', cont)
                    full_text += ' ' + cont
                story_list.append(Paragraph(f"{m.group(1)}.  {full_text}", num_style))
                i += 1
                continue

            # Regular paragraph — collect consecutive non-blank lines
            para_lines = [line.strip()]
            while i + 1 < len(lines) and lines[i + 1].strip() and \
                    not lines[i + 1].startswith('Table') and \
                    not re.match(r'^\d+\.\d+', lines[i + 1].strip()) and \
                    not lines[i + 1].strip().isupper():
                i += 1
                para_lines.append(lines[i].strip())
            text_joined = ' '.join(para_lines)
            # Handle markdown bold
            text_joined = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text_joined)
            if text_joined.strip():
                story_list.append(Paragraph(text_joined, s_body))
            i += 1

    def _render_text_table(table_lines, story_list):
        """Convert fixed-width text table lines into a styled reportlab Table."""
        if not table_lines:
            return
        # Parse by splitting on 2+ spaces
        parsed = []
        for tl in table_lines:
            cols = re.split(r'\s{2,}', tl.strip())
            if cols and any(c.strip() for c in cols):
                parsed.append([c.strip() for c in cols])
        if not parsed:
            return
        # Normalize column count
        max_cols = max(len(r) for r in parsed)
        for row in parsed:
            while len(row) < max_cols:
                row.append('')

        # Convert to Paragraph cells
        cell_style = ParagraphStyle('tcell', parent=s_body, fontSize=9, leading=11, spaceAfter=0)
        cell_bold = ParagraphStyle('tcellb', parent=cell_style, fontName='Helvetica-Bold')
        data = []
        for ri, row in enumerate(parsed):
            data.append([
                Paragraph(c, cell_bold if ri == 0 else cell_style) for c in row
            ])

        col_w = CONTENT_W / max_cols
        t = Table(data, colWidths=[col_w] * max_cols)
        style_cmds = [
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BACKGROUND', (0, 0), (-1, 0), _hex_to_rgb(TABLE_HEADER_BG)),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.white, _hex_to_rgb(TABLE_ALT_ROW)]),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.3, _hex_to_rgb('#cccccc')),
        ]
        t.setStyle(TableStyle(style_cmds))
        story_list.append(Spacer(1, 4))
        story_list.append(t)
        story_list.append(Spacer(1, 8))

    # ════════════════════════════════════════════════════════════════════
    # 1. INTRODUCTION
    # ════════════════════════════════════════════════════════════════════
    story.append(Paragraph("1. Introduction", s_h1))
    story.append(HRFlowable(
        width='100%', thickness=1, color=_hex_to_rgb(FOREST_GREEN),
        spaceBefore=0, spaceAfter=8,
    ))
    _render_text_section(report_generator.generate_introduction(), story)
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════
    # 2. METHODOLOGY
    # ════════════════════════════════════════════════════════════════════
    story.append(Paragraph("2. Methodology", s_h1))
    story.append(HRFlowable(
        width='100%', thickness=1, color=_hex_to_rgb(FOREST_GREEN),
        spaceBefore=0, spaceAfter=8,
    ))
    _render_text_section(report_generator.generate_methodology(), story)
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════
    # 3. RESULTS
    # ════════════════════════════════════════════════════════════════════
    story.append(Paragraph("3. Results", s_h1))
    story.append(HRFlowable(
        width='100%', thickness=1, color=_hex_to_rgb(FOREST_GREEN),
        spaceBefore=0, spaceAfter=8,
    ))

    # ── Core Metrics Summary Table (manually built for better formatting) ──
    story.append(Paragraph("3.1 Core Network Metrics", s_h2))

    # Use .get() with defaults for metrics that may not always be present
    _m = lambda k, d=0.0: metrics.get(k, d)
    redundancy = _m('redundancy')
    regen_cap = _m('regenerative_capacity')
    eld = _m('effective_link_density')
    td = _m('trophic_depth')

    core_data = [
        ['Metric', 'Value', 'Interpretation'],
        ['Total System Throughput (TST)',
         f"{metrics['total_system_throughput']:.3f}", 'System scale'],
        ['Average Mutual Information (AMI)',
         f"{metrics['average_mutual_information']:.3f}", 'bits'],
        ['Flow Diversity (H)',
         f"{_m('flow_diversity'):.3f}", 'bits'],
        ['Ascendency (A)',
         f"{metrics['ascendency']:.3f}",
         f"{metrics['ascendency_ratio']*100:.1f}% of capacity"],
        ['Development Capacity (C)',
         f"{metrics['development_capacity']:.3f}", '100% (theoretical max)'],
        ['Overhead (Φ)',
         f"{metrics['overhead']:.3f}",
         f"{metrics['overhead_ratio']*100:.1f}% of capacity"],
        ['Robustness (R)',
         f"{_m('robustness'):.3f}", rob_status],
        ['Network Efficiency',
         f"{_m('network_efficiency'):.3f}", eff_status],
        ['Redundancy',
         f"{redundancy:.3f}",
         'High' if redundancy > 0.6 else 'Moderate' if redundancy > 0.3 else 'Low'],
        ['Effective Link Density',
         f"{eld:.3f}", 'ratio'],
        ['Trophic Depth',
         f"{td:.3f}", 'levels'],
        ['Regenerative Capacity',
         f"{regen_cap:.3f}",
         'Strong' if regen_cap > 0.4 else 'Moderate' if regen_cap > 0.2 else 'Limited'],
    ]

    cell_s = ParagraphStyle('cs', parent=s_body, fontSize=9, leading=11, spaceAfter=0)
    cell_b = ParagraphStyle('cb', parent=cell_s, fontName='Helvetica-Bold')
    cell_h = ParagraphStyle('ch', parent=cell_s, fontName='Helvetica-Bold',
                             textColor=colors.white)

    core_table_data = []
    for ri, row in enumerate(core_data):
        if ri == 0:
            core_table_data.append([Paragraph(c, cell_h) for c in row])
        else:
            core_table_data.append([
                Paragraph(row[0], cell_b),
                Paragraph(row[1], cell_s),
                Paragraph(row[2], cell_s),
            ])

    core_t = Table(core_table_data, colWidths=[CONTENT_W * 0.45, CONTENT_W * 0.2, CONTENT_W * 0.35])
    core_style = [
        ('BACKGROUND', (0, 0), (-1, 0), _hex_to_rgb(TABLE_HEADER_BG)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.3, _hex_to_rgb('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.white, _hex_to_rgb(TABLE_ALT_ROW)]),
    ]
    core_t.setStyle(TableStyle(core_style))
    story.append(core_t)
    story.append(Paragraph(
        "<i>Table 1. Core Network Metrics</i> — Key information-theoretic and organizational measures derived from the Ulanowicz-Fath framework, with interpretive assessments for each indicator.", s_caption))

    # ── Viability Assessment ──
    story.append(Paragraph("3.2 Sustainability Assessment", s_h2))
    viab_data = [
        ['Parameter', 'Value', 'Status'],
        ['Current Position (α)', f"{alpha:.3f}",
         'Optimal' if 0.30 <= alpha <= 0.45 else 'Developing' if alpha < 0.35 else 'Efficient'],
        ['Reference Lower Edge', f"{metrics['viability_lower_bound']:.3f}",
         'above' if alpha > metrics['viability_lower_bound'] else 'below'],
        ['Reference Upper Edge', f"{metrics['viability_upper_bound']:.3f}",
         'below' if alpha < metrics['viability_upper_bound'] else 'above'],
        ['Gradient Position', _pdf_gradient(alpha)['position'],
         'Direction of travel: ' + _pdf_gradient(alpha)['direction_of_travel']],
    ]
    viab_table_data = []
    for ri, row in enumerate(viab_data):
        if ri == 0:
            viab_table_data.append([Paragraph(c, cell_h) for c in row])
        else:
            status_color = _get_status_color(row[2])
            viab_table_data.append([
                Paragraph(row[0], cell_b),
                Paragraph(row[1], cell_s),
                Paragraph(row[2], ParagraphStyle(
                    'status', parent=cell_s,
                    fontName='Helvetica-Bold',
                    textColor=_hex_to_rgb(status_color))),
            ])
    viab_t = Table(viab_table_data, colWidths=[CONTENT_W * 0.45, CONTENT_W * 0.25, CONTENT_W * 0.30])
    viab_style = list(core_style)  # reuse
    viab_t.setStyle(TableStyle(viab_style))
    story.append(viab_t)
    story.append(Paragraph(
        "<i>Table 2. Gradient Position</i> — Position of the organization on the "
        "efficiency-resilience gradient relative to the indicative reference band, "
        "with direction of travel. " + _pdf_gradient(alpha)['caveat'], s_caption))

    # ── Window-of-Viability / robustness curve (most important credibility
    #    visual). Prefer the native matplotlib builder; fall back to the
    #    Plotly robustness curve via kaleido if matplotlib is unavailable.
    _wov_num = [0]  # figure counter carried into 3.3

    def _build_wov_image():
        try:
            from visualizer import SustainabilityVisualizer as _SV
        except Exception:
            from src.visualizer import SustainabilityVisualizer as _SV
        try:
            viz = _SV(calculator)
            mpl_fig = viz.plot_sustainability_curve_matplotlib(figsize=(11, 4.5))
            img = _mpl_image(mpl_fig, width=CONTENT_W * 0.95, height=CONTENT_W * 0.95 * 4.5 / 11)
            if img is not None:
                return img
            # Fallback: plotly robustness curve through kaleido
            return _chart_image(viz.create_robustness_curve(),
                                width=CONTENT_W * 0.9, height=250)
        except Exception:
            return None

    if _guarded_chart_block(
            _build_wov_image,
            "<i>Figure 1. Window of Viability &amp; Robustness Curve</i> — Left: the "
            "organization's ascendency (A) versus development capacity (C) with the "
            "green band marking the empirical window of viability. Right: key "
            "sustainability metrics. This visual anchors the efficiency&ndash;resilience "
            "trade-off central to the Ulanowicz-Fath framework.",
            story):
        _wov_num[0] = 1

    # ── 3.3 Visualizations ──
    # Self-sufficient: charts are built internally from the calculator so the
    # report embeds real images even when the caller passes charts=None
    # (previously this whole block was skipped -> zero images in the PDF).
    # Any caller-supplied plotly figures are embedded in addition.
    _figure_notes = {
        "System Robustness Curve": "The organization's position (red marker) relative to the theoretical robustness function R = -α·log(α), with the empirical optimum at α ≈ 0.37.",
        "Core Metrics Analysis": "Comparative bar chart of key information-theoretic indicators, enabling rapid identification of metrics that deviate from healthy-system benchmarks.",
        "Flow Distribution": "Distribution of resource flows across the top network nodes, illustrating concentration patterns and potential structural dependencies.",
    }

    story.append(Paragraph("3.3 Visualizations", s_h2))
    fig_num = [_wov_num[0]]  # continue numbering after the WoV figure

    # (a) Internal flow / Sankey diagram built from the calculator.
    def _build_flow_image():
        try:
            from visualizer import SustainabilityVisualizer as _SV
        except Exception:
            from src.visualizer import SustainabilityVisualizer as _SV
        viz = _SV(calculator)
        return _chart_image(viz.create_sankey_diagram(),
                            width=CONTENT_W * 0.95, height=300)

    fig_num[0] += 1
    _flow_ok = _guarded_chart_block(
        _build_flow_image,
        f"<i>Figure {fig_num[0]}. Network Flow Diagram (Sankey)</i> — Directed "
        "resource flows between nodes, revealing structural pathways, hubs and "
        "dependencies across the organizational network.",
        story)
    if not _flow_ok:
        fig_num[0] -= 1  # don't burn a figure number on a skipped chart

    # (b) Any caller-supplied plotly figures (app path).
    embedded_any = _flow_ok or _wov_num[0] > 0
    if charts:
        for chart_name, fig in charts.items():
            if fig is None:
                continue

            def _build(_f=fig):
                return _chart_image(_f, width=CONTENT_W * 0.92, height=250)

            fig_num[0] += 1
            note = _figure_notes.get(
                chart_name,
                f"Visualization of {chart_name.lower()} for the analyzed network.")
            ok = _guarded_chart_block(
                _build,
                f"<i>Figure {fig_num[0]}. {chart_name}</i> — {note}",
                story)
            if ok:
                embedded_any = True
            else:
                fig_num[0] -= 1

    if not embedded_any:
        story.append(Paragraph(
            "Chart images could not be generated for this report.",
            s_body_italic))

    # ── Remaining results text ──
    story.append(Paragraph("3.4 Flow Distribution Analysis", s_h2))
    flows = calculator.flow_matrix[calculator.flow_matrix > 0]
    if len(flows) > 0:
        gini = report_generator._calculate_gini()
        flow_stats = [
            ['Statistic', 'Value'],
            ['Mean Flow', f"{np.mean(flows):.3f}"],
            ['Median Flow', f"{np.median(flows):.3f}"],
            ['Standard Deviation', f"{np.std(flows):.3f}"],
            ['Coefficient of Variation', f"{np.std(flows)/np.mean(flows):.3f}"],
            ['Maximum Flow', f"{np.max(flows):.3f}"],
            ['Gini Coefficient', f"{gini:.3f}"],
        ]
        flow_td = []
        for ri, row in enumerate(flow_stats):
            if ri == 0:
                flow_td.append([Paragraph(c, cell_h) for c in row])
            else:
                flow_td.append([Paragraph(row[0], cell_b), Paragraph(row[1], cell_s)])
        flow_t = Table(flow_td, colWidths=[CONTENT_W * 0.55, CONTENT_W * 0.45])
        flow_style_cmds = [
            ('BACKGROUND', (0, 0), (-1, 0), _hex_to_rgb(TABLE_HEADER_BG)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.3, _hex_to_rgb('#cccccc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.white, _hex_to_rgb(TABLE_ALT_ROW)]),
        ]
        flow_t.setStyle(TableStyle(flow_style_cmds))
        story.append(flow_t)
        story.append(Paragraph(
            "<i>Table 3. Flow Distribution Statistics</i> — Summary of resource flow concentration and inequality across the network, including Gini coefficient and entropy-based dispersion measures.", s_caption))

        ineq = ('high inequality' if gini > 0.6 else
                'moderate inequality' if gini > 0.3 else
                'relatively equal distribution')
        story.append(Paragraph(
            f"The Gini coefficient of {gini:.3f} indicates {ineq} in flow "
            f"distribution across the network.", s_body))

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════
    # 4. OASIS HEALTH ASSESSMENT
    # ════════════════════════════════════════════════════════════════════
    story.append(Paragraph("4. OASIS Organizational Health Assessment", s_h1))
    story.append(HRFlowable(
        width='100%', thickness=1, color=_hex_to_rgb(FOREST_GREEN),
        spaceBefore=0, spaceAfter=8,
    ))

    # Try to get OASIS data — prefer the precomputed profile (computed once at
    # provision) carried on the report_generator; recompute only on a miss.
    try:
        profile = getattr(report_generator, 'oasis_profile', None)
        if not (isinstance(profile, dict) and 'dimension_scores' in profile):
            profile = None
        interpretations = profile.get('interpretation') if profile else None
        recommendations = profile.get('recommendations') if profile else None
        if profile is None or interpretations is None or recommendations is None:
            from oasis_calculator import OASISCalculator
            oasis = OASISCalculator(calculator)
            if profile is None:
                profile = oasis.get_oasis_profile()
            if interpretations is None:
                interpretations = oasis.get_oasis_interpretation()
            if recommendations is None:
                recommendations = oasis.get_recommendations()

        scores = profile['dimension_scores']
        overall = profile['overall_score']
        overall_status = profile['overall_status']
        dim_status = profile['dimension_status']

        story.append(Paragraph(
            f"The OASIS framework assesses organizational health across five dimensions "
            f"derived from Fath et al.'s (2019) 10 Principles of Regenerative Economics. "
            f"The overall health score is <b>{overall:.0f}/100</b> ({overall_status}).",
            s_body))

        # OASIS scores table
        oasis_data = [['Dimension', 'Score', 'Status', 'Key Focus']]
        dim_descriptions = {
            'open': 'Interconnectivity & exchange',
            'autonomous': 'Learning & routine encoding',
            'symbiotic': 'Integration & balance',
            'intelligent': 'Functional diversity',
            'sustainable': 'Order-freedom balance',
        }
        for dim in ['open', 'autonomous', 'symbiotic', 'intelligent', 'sustainable']:
            sc = scores[dim]
            st_text = dim_status[dim]
            status_c = _get_status_color(st_text)
            oasis_data.append([
                Paragraph(dim.upper(), ParagraphStyle(
                    'od', parent=cell_b,
                    textColor=_hex_to_rgb(OASIS_COLORS.get(dim, FOREST_GREEN)))),
                Paragraph(f"{sc:.0f}", cell_s),
                Paragraph(st_text, ParagraphStyle(
                    'os', parent=cell_s, fontName='Helvetica-Bold',
                    textColor=_hex_to_rgb(status_c))),
                Paragraph(dim_descriptions.get(dim, ''), cell_s),
            ])

        oasis_t = Table(oasis_data, colWidths=[
            CONTENT_W * 0.22, CONTENT_W * 0.12, CONTENT_W * 0.18, CONTENT_W * 0.48])
        oasis_style_cmds = [
            ('BACKGROUND', (0, 0), (-1, 0), _hex_to_rgb(TABLE_HEADER_BG)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.3, _hex_to_rgb('#cccccc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.white, _hex_to_rgb(TABLE_ALT_ROW)]),
        ]
        oasis_t.setStyle(TableStyle(oasis_style_cmds))
        story.append(oasis_t)
        story.append(Paragraph(
            "<i>Table 4. OASIS Organizational Health Profile</i> — Composite scores across the five OASIS dimensions (Open, Autonomous, Symbiotic, Intelligent, Sustainable), with status indicators benchmarked against healthy-system thresholds.", s_caption))

        # ── OASIS radar + dimension gauges (embedded images) ──
        try:
            from oasis_visualizer import (
                create_oasis_radar_chart as _radar,
                create_all_dimension_gauges as _gauges)
        except Exception:
            try:
                from src.oasis_visualizer import (
                    create_oasis_radar_chart as _radar,
                    create_all_dimension_gauges as _gauges)
            except Exception:
                _radar = _gauges = None

        if _radar is not None:
            _guarded_chart_block(
                lambda: _chart_image(
                    _radar(scores, title="OASIS Health Profile"),
                    width=CONTENT_W * 0.7, height=CONTENT_W * 0.7),
                "<i>Figure O1. OASIS Radar</i> — Five-dimension health profile "
                "(Open, Autonomous, Symbiotic, Intelligent, Sustainable) plotted "
                "against healthy-system thresholds. A balanced pentagon indicates "
                "well-rounded organizational health.",
                story)
        if _gauges is not None:
            _guarded_chart_block(
                lambda: _chart_image(
                    _gauges(profile),
                    width=CONTENT_W * 0.98, height=200),
                "<i>Figure O2. OASIS Dimension Gauges</i> — Per-dimension scores "
                "(0&ndash;100) with color-coded status bands for at-a-glance "
                "identification of strengths and weaknesses.",
                story)

        # Dimension interpretations
        story.append(Paragraph("4.1 Dimension Interpretations", s_h2))
        for dim in ['open', 'autonomous', 'symbiotic', 'intelligent', 'sustainable']:
            dim_color = OASIS_COLORS.get(dim, FOREST_GREEN)
            story.append(Paragraph(
                f'<font color="{dim_color}"><b>{dim.upper()}</b></font>',
                ParagraphStyle('dimh', parent=s_h3, textColor=_hex_to_rgb(dim_color))))
            interp = interpretations.get(dim, 'No interpretation available.')
            story.append(Paragraph(interp, s_body))

        # Recommendations
        if recommendations:
            story.append(Paragraph("4.2 OASIS-Based Recommendations", s_h2))
            for i, rec in enumerate(recommendations, 1):
                priority_c = _get_status_color(rec.get('priority', ''))
                story.append(Paragraph(
                    f"<b>Recommendation {i}</b> "
                    f"(<font color=\"{priority_c}\">{rec.get('priority', 'N/A')} priority</font>)",
                    s_h3))
                rec_items = [
                    f"<b>Dimension:</b> {rec.get('dimension', 'N/A')}",
                    f"<b>Issue:</b> {rec.get('issue', 'N/A')}",
                    f"<b>Action:</b> {rec.get('action', 'N/A')}",
                ]
                metrics_to_improve = rec.get('metrics_to_improve', [])
                if metrics_to_improve:
                    _mnames = ', '.join(
                        humanize_metric_name(_m) for _m in metrics_to_improve)
                    rec_items.append(
                        f"<b>Metrics to improve:</b> {_mnames}")
                for item in rec_items:
                    story.append(Paragraph(f"•  {item}", ParagraphStyle(
                        'ri', parent=s_body, leftIndent=18, firstLineIndent=-12,
                        spaceBefore=2, spaceAfter=2)))

    except Exception:
        story.append(Paragraph(
            "OASIS assessment could not be completed for this analysis. "
            "The framework requires additional network metrics that may not be available.",
            s_body_italic))

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════
    # 5-8. DETAILED ECOSYSTEMIC ANALYSIS
    # (benchmarking, risk & resilience, action roadmap, ESG mapping)
    # Built from src/report_intelligence.py on metrics already computed.
    # ════════════════════════════════════════════════════════════════════
    _ri = None
    try:
        import report_intelligence as _ri
        from oasis_calculator import OASISCalculator as _OC
    except Exception:
        try:
            from src import report_intelligence as _ri
            from src.oasis_calculator import OASISCalculator as _OC
        except Exception:
            _ri = None

    if _ri is not None:
        try:
            # Prefer the precomputed OASIS profile carried on report_generator.
            _profile = getattr(report_generator, 'oasis_profile', None)
            if not (isinstance(_profile, dict) and 'dimension_scores' in _profile):
                _profile = None
            _recs = _profile.get('recommendations') if _profile else None
            if _profile is None or _recs is None:
                _oasis = _OC(calculator)
                if _profile is None:
                    _profile = _oasis.get_oasis_profile()
                if _recs is None:
                    _recs = _oasis.get_recommendations()
            _bench = _ri.build_benchmark_view(metrics, _profile)
            _risk = _ri.build_risk_view(metrics, _profile)
            _roadmap = _ri.build_action_roadmap(_recs, _profile)
            _esg = _ri.build_esg_crosswalk(_profile, metrics)

            def _sec_rule():
                story.append(HRFlowable(
                    width='100%', thickness=1, color=_hex_to_rgb(FOREST_GREEN),
                    spaceBefore=0, spaceAfter=8))

            # ---- 5. Benchmarking & Position ----
            story.append(Paragraph("5. Benchmarking &amp; Position", s_h1))
            _sec_rule()
            _pos = {
                'within': 'within the Window of Viability',
                'above': 'above the viability band (tending rigid / over-organized)',
                'below': 'below the viability band (tending chaotic / under-organized)',
            }.get(_bench['position'], 'undetermined')
            story.append(Paragraph(
                f"The organization's relative ascendency is "
                f"<b>&alpha; = {_bench['alpha']:.3f}</b>, placing it {_pos} "
                f"(viable band {_bench['lower']}&ndash;{_bench['upper']}; robustness "
                f"optimum &alpha; &asymp; {_bench['optimum']:.2f}). Distance to the "
                f"robustness optimum is <b>{_bench['distance_to_optimum']:.3f}</b>.",
                s_body))
            # ── PRIMARY comparator: organizational anchor (Fath et al. 2019) ──
            _org_lo, _org_hi = 0.30, 0.45
            _org_in = _org_lo <= _bench['alpha'] <= _org_hi
            story.append(Paragraph(
                "<b>Primary benchmark &mdash; organizational reference.</b> "
                "High-performing organizations analyzed with this framework exhibit "
                "relative ascendency &alpha; in the range "
                f"<b>0.30&ndash;0.45</b> (Fath et al., 2019). At &alpha; = "
                f"{_bench['alpha']:.3f}, {org_name} "
                f"{'sits within' if _org_in else 'sits outside'} this "
                "organizational band, which is the appropriate headline comparator "
                "for interpreting these results.", s_body))
            _org_data = [
                ['Reference', 'Relative Ascendency (α)', 'Source'],
                [Paragraph('High-performing organizations', cell_b),
                 Paragraph('0.30–0.45', cell_s),
                 Paragraph('Fath et al., 2019', cell_s)],
                [Paragraph(f'<b>{org_name} (this assessment)</b>', cell_b),
                 Paragraph(f"<b>{_bench['alpha']:.3f}</b>", cell_s),
                 Paragraph('—', cell_s)],
            ]
            _ot = Table(_org_data, colWidths=[
                CONTENT_W * 0.40, CONTENT_W * 0.30, CONTENT_W * 0.30])
            _ot.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), _hex_to_rgb(TABLE_HEADER_BG)),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.3, _hex_to_rgb('#cccccc')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                 [colors.white, _hex_to_rgb(TABLE_ALT_ROW)]),
            ]))
            story.append(_ot)
            story.append(Paragraph(
                "<i>Table 5. Primary benchmark &mdash; organizational reference "
                "band (Fath et al., 2019).</i>", s_caption))
            # ── Secondary: ecological anchors, clearly demoted to illustrative ──
            story.append(Spacer(1, 0.3 * cm))
            story.append(Paragraph(
                "The ecological values below are provided only as <b>illustrative "
                "methodology reference points</b> that calibrate the viability "
                "scale. They are <b>not</b> the benchmark for this organization and "
                "should not be read as targets.", s_body_italic))
            _anchor_data = [['Ecological Reference Point (illustrative)',
                             'Relative Ascendency (α)', 'Source']]
            for _a in _bench['reference_anchors']:
                _anchor_data.append([
                    Paragraph(_a['label'], cell_b),
                    Paragraph(f"{_a['relative_ascendency']:.3f}", cell_s),
                    Paragraph(_a.get('source', ''), cell_s)])
            if len(_anchor_data) > 1:
                _at = Table(_anchor_data, colWidths=[
                    CONTENT_W * 0.40, CONTENT_W * 0.30, CONTENT_W * 0.30])
                _at.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), _hex_to_rgb(MUTED)),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('TOPPADDING', (0, 0), (-1, -1), 5),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('GRID', (0, 0), (-1, -1), 0.3, _hex_to_rgb('#cccccc')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                     [colors.white, _hex_to_rgb(TABLE_ALT_ROW)]),
                ]))
                story.append(_at)
                story.append(Paragraph(
                    "<i>Table 5b. Ecological reference points (illustrative). "
                    "Scientific calibration values for the viability scale, "
                    "not organizational targets.</i>", s_caption))
            story.append(PageBreak())

            # ---- 6. Risk & Resilience Analysis ----
            story.append(Paragraph("6. Risk &amp; Resilience Analysis", s_h1))
            _sec_rule()
            story.append(Paragraph(
                f"Overall fragility classification: <b>{_risk['fragility']}</b>. "
                f"Adaptive reserve indicators &mdash; overhead ratio "
                f"{_risk['overhead_ratio'] * 100:.1f}%, redundancy "
                f"{_risk['redundancy']:.3f}.", s_body))
            for _it in _risk['items']:
                _sc = _get_status_color(_it['severity'])
                story.append(Paragraph(
                    f"<font color=\"{_sc}\"><b>{_it['severity']}</b></font> "
                    f"&mdash; {_it['title']}", s_h3))
                story.append(Paragraph(f"<b>Evidence:</b> {_it['evidence']}", s_body))
                story.append(Paragraph(
                    f"<b>Implication:</b> {_it['implication']}", s_body))
            story.append(PageBreak())

            # ---- 7. Prioritized Action Roadmap ----
            story.append(Paragraph("7. Prioritized Action Roadmap", s_h1))
            _sec_rule()
            for _htitle, _hkey in [
                ('7.1 Immediate (0–3 months)', 'immediate'),
                ('7.2 Short-Term (3–9 months)', 'short_term'),
                ('7.3 Medium-Term (9–18 months)', 'medium_term'),
            ]:
                story.append(Paragraph(_htitle, s_h2))
                _items = _roadmap[_hkey]
                if not _items:
                    story.append(Paragraph(
                        "No actions in this horizon.", s_body_italic))
                    continue
                for _it in _items:
                    _pc = _get_status_color(_it['priority'])
                    story.append(Paragraph(
                        f"<font color=\"{_pc}\"><b>{_it['priority']}</b></font> "
                        f"&middot; {_it['dimension']}", s_h3))
                    story.append(Paragraph(f"<b>Issue:</b> {_it['issue']}", s_body))
                    story.append(Paragraph(f"<b>Action:</b> {_it['action']}", s_body))
                    story.append(Paragraph(
                        f"<b>Expected impact:</b> {_it['expected_impact']}", s_body))
                    _m = ', '.join(
                        humanize_metric_name(_x)
                        for _x in _it['metrics_to_improve']) or 'N/A'
                    story.append(Paragraph(
                        f"<b>Metrics to improve:</b> {_m}", s_body))
            story.append(PageBreak())

            # ---- 8. ESG Framework Mapping ----
            story.append(Paragraph("8. ESG Framework Mapping", s_h1))
            _sec_rule()
            story.append(Paragraph(
                "Indicative crosswalk linking OASIS findings to recognized disclosure "
                "frameworks (GRI, ESRS/CSRD, TCFD). Provided for navigation and context "
                "only; not a compliance attestation.", s_body_italic))
            _esg_data = [['OASIS Finding', 'GRI', 'ESRS / CSRD', 'TCFD']]
            for _row in _esg:
                _esg_data.append([
                    Paragraph(
                        f"<b>{_row['oasis_dimension']}</b><br/>"
                        f"{_row['finding_summary']}", cell_s),
                    Paragraph(_row['gri_ref'], cell_s),
                    Paragraph(_row['esrs_ref'], cell_s),
                    Paragraph(_row['tcfd_ref'], cell_s)])
            _et = Table(_esg_data, colWidths=[
                CONTENT_W * 0.34, CONTENT_W * 0.22, CONTENT_W * 0.22, CONTENT_W * 0.22])
            _et.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), _hex_to_rgb(TABLE_HEADER_BG)),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.3, _hex_to_rgb('#cccccc')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                 [colors.white, _hex_to_rgb(TABLE_ALT_ROW)]),
            ]))
            story.append(_et)
            story.append(Paragraph(
                "<i>Table 6. Indicative OASIS-to-ESG framework crosswalk.</i>",
                s_caption))
            story.append(PageBreak())
        except Exception:
            # Detailed analysis is additive; never break the base report.
            pass

    # ════════════════════════════════════════════════════════════════════
    # 9. DISCUSSION
    # ════════════════════════════════════════════════════════════════════
    story.append(Paragraph("9. Discussion", s_h1))
    story.append(HRFlowable(
        width='100%', thickness=1, color=_hex_to_rgb(FOREST_GREEN),
        spaceBefore=0, spaceAfter=8,
    ))
    _render_text_section(report_generator.generate_discussion(), story)
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════
    # 6. CONCLUSIONS & RECOMMENDATIONS
    # ════════════════════════════════════════════════════════════════════
    story.append(Paragraph("10. Conclusions &amp; Recommendations", s_h1))
    story.append(HRFlowable(
        width='100%', thickness=1, color=_hex_to_rgb(FOREST_GREEN),
        spaceBefore=0, spaceAfter=8,
    ))
    _render_text_section(report_generator.generate_conclusions(), story)
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════
    # REFERENCES
    # ════════════════════════════════════════════════════════════════════
    story.append(Paragraph("References", s_h1))
    story.append(HRFlowable(
        width='100%', thickness=1, color=_hex_to_rgb(FOREST_GREEN),
        spaceBefore=0, spaceAfter=8,
    ))
    refs_text = report_generator.generate_references()
    # Parse references individually
    for line in refs_text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('===') or line.startswith('---') or \
                line.startswith('REFERENCES') or line.startswith('Additional'):
            continue
        if line.startswith('- '):
            story.append(Paragraph(f"•  {line[2:]}", ParagraphStyle(
                'refb', parent=s_reference, leftIndent=18, firstLineIndent=-12)))
        elif line[0].isupper() and '(' in line:
            story.append(Paragraph(line, s_reference))
        elif line:
            story.append(Paragraph(line, s_body))

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════
    # APPENDIX
    # ════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Appendix: Detailed Data", s_h1))
    story.append(HRFlowable(
        width='100%', thickness=1, color=_hex_to_rgb(FOREST_GREEN),
        spaceBefore=0, spaceAfter=8,
    ))

    # Node statistics table (manually built for quality)
    story.append(Paragraph("A1. Node-Level Statistics", s_h2))
    node_data = [['#', 'Node Name', 'In-Flow', 'Out-Flow', 'Total Flow']]
    for i, name in enumerate(calculator.node_names[:30]):  # cap at 30
        in_f = float(np.sum(calculator.flow_matrix[:, i]))
        out_f = float(np.sum(calculator.flow_matrix[i, :]))
        node_data.append([
            str(i + 1),
            Paragraph(name[:30], cell_s),
            f"{in_f:.2f}",
            f"{out_f:.2f}",
            f"{in_f + out_f:.2f}",
        ])
    if len(calculator.node_names) > 30:
        node_data.append(['...', f'({len(calculator.node_names) - 30} more nodes)', '', '', ''])

    node_t = Table(node_data, colWidths=[
        CONTENT_W * 0.06, CONTENT_W * 0.38, CONTENT_W * 0.18,
        CONTENT_W * 0.18, CONTENT_W * 0.20])
    node_style = [
        ('BACKGROUND', (0, 0), (-1, 0), _hex_to_rgb(TABLE_HEADER_BG)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
        ('ALIGN', (2, 0), (-1, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.3, _hex_to_rgb('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.white, _hex_to_rgb(TABLE_ALT_ROW)]),
    ]
    node_t.setStyle(TableStyle(node_style))
    story.append(node_t)
    story.append(Paragraph(
        "<i>Table A1. Node-Level Flow Statistics</i> — Disaggregated inflow and outflow volumes for each node in the network, enabling identification of dominant actors and potential bottlenecks.", s_caption))

    # Assessment categories
    story.append(Paragraph("A2. Assessment Categories", s_h2))
    assess_data = [['Category', 'Assessment']]
    for cat, assessment in report_generator.assessments.items():
        assess_data.append([
            cat.replace('_', ' ').title(),
            assessment,
        ])
    assess_t = Table(assess_data, colWidths=[CONTENT_W * 0.40, CONTENT_W * 0.60])
    assess_style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), _hex_to_rgb(TABLE_HEADER_BG)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.3, _hex_to_rgb('#cccccc')),
    ]
    assess_style_cmds.append(
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.white, _hex_to_rgb(TABLE_ALT_ROW)]))
    assess_t.setStyle(TableStyle(assess_style_cmds))
    story.append(assess_t)
    story.append(Paragraph(
        "<i>Table A2. Metric Assessment Categories</i> — Classification of each computed metric into qualitative assessment bands (e.g., Healthy, Developing, Critical) used throughout this report.", s_caption))

    # ── Build the PDF ────────────────────────────────────────────────────
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def create_simple_pdf(report_text, org_name):
    """
    Create a simple text-based PDF using reportlab.
    Fallback method for when the full report builder fails.

    Args:
        report_text: Full text of the report
        org_name: Organization name

    Returns:
        bytes: PDF content, or None if reportlab unavailable
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            rightMargin=72, leftMargin=72,
            topMargin=72, bottomMargin=72,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle', parent=styles['Heading1'],
            fontSize=24, textColor=colors.HexColor(FOREST_GREEN),
            spaceAfter=30, alignment=TA_CENTER,
        )
        heading_style = ParagraphStyle(
            'CustomHeading', parent=styles['Heading2'],
            fontSize=16, textColor=colors.HexColor(FOREST_GREEN),
            spaceAfter=12, spaceBefore=12,
        )
        body_style = ParagraphStyle(
            'CustomBody', parent=styles['BodyText'],
            fontSize=11, alignment=TA_JUSTIFY, spaceAfter=12,
        )

        story = []
        story.append(Paragraph("Organizational Network Analysis Report", title_style))
        story.append(Paragraph(org_name, heading_style))
        story.append(Spacer(1, 0.5 * inch))

        sections = report_text.split('\n\n')
        for section in sections:
            if not section.strip():
                continue
            stripped = section.strip()
            if stripped.startswith('===') or stripped.startswith('---'):
                continue
            if stripped.isupper() or (stripped and stripped[0].isdigit() and len(stripped) < 80):
                story.append(PageBreak())
                story.append(Paragraph(stripped, heading_style))
            else:
                clean = stripped.replace('\n', '<br/>').replace('&', '&amp;')
                clean = clean.replace('<', '&lt;').replace('>', '&gt;')
                clean = clean.replace('&lt;br/&gt;', '<br/>')
                story.append(Paragraph(clean, body_style))
            story.append(Spacer(1, 0.1 * inch))

        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes

    except ImportError:
        return None
