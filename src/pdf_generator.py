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
        except Exception:
            return None

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
    viability = 'Viable' if metrics['is_viable'] else 'Non-Viable'
    cover_data = [
        ['Network Nodes', 'Active Connections', 'Viability Status', 'Robustness'],
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
    # EXECUTIVE SUMMARY (KPI Cards)
    # ════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Executive Summary", s_h1))
    story.append(HRFlowable(
        width='100%', thickness=1, color=_hex_to_rgb(FOREST_GREEN),
        spaceBefore=0, spaceAfter=12,
    ))

    # Build KPI cards as a table
    def _kpi_cell(label, value, status, color=None):
        c = color or _get_status_color(status)
        return [
            Paragraph(str(value), ParagraphStyle(
                'kv', parent=s_kpi_value, textColor=_hex_to_rgb(c))),
            Paragraph(label, s_kpi_label),
            Paragraph(status, ParagraphStyle(
                'ks', parent=s_kpi_status, textColor=_hex_to_rgb(c))),
        ]

    rob = metrics.get('robustness', 0)
    # E-20: unified robustness "high" threshold (0.25) — shared source of truth.
    try:
        import report_intelligence as _ri_rob
    except ImportError:  # pragma: no cover
        from src import report_intelligence as _ri_rob
    rob_status = ('High' if rob >= _ri_rob.ROBUSTNESS_HIGH_THRESHOLD
                  else 'Moderate' if rob > 0.15 else 'Low')
    eff = metrics.get('network_efficiency', 0)
    eff_status = 'Optimal' if 0.2 <= eff <= 0.6 else 'Sub-optimal'
    alpha = metrics.get('ascendency_ratio', 0)

    kpi_cells = [
        _kpi_cell('Viability Status', viability, viability),
        _kpi_cell('Robustness (R)', f"{rob:.3f}", rob_status),
        _kpi_cell('Network Efficiency', f"{eff:.3f}", eff_status),
        _kpi_cell('Rel. Ascendency (α)', f"{alpha:.3f}",
                   'Optimal' if 0.30 <= alpha <= 0.45 else 'Warning'),
    ]

    # Flatten into table rows (3 rows per card, 4 columns)
    kpi_data = [[cell[i] for cell in kpi_cells] for i in range(3)]
    kpi_table = Table(kpi_data, colWidths=[CONTENT_W / 4] * 4)
    kpi_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('BOX', (0, 0), (0, -1), 0.5, _hex_to_rgb('#e0e0e0')),
        ('BOX', (1, 0), (1, -1), 0.5, _hex_to_rgb('#e0e0e0')),
        ('BOX', (2, 0), (2, -1), 0.5, _hex_to_rgb('#e0e0e0')),
        ('BOX', (3, 0), (3, -1), 0.5, _hex_to_rgb('#e0e0e0')),
        ('BACKGROUND', (0, 0), (-1, -1), _hex_to_rgb('#fafcfb')),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 0.5 * cm))

    # Executive summary text
    exec_text = (
        f"This report presents a comprehensive network analysis of <b>{org_name}</b> "
        f"using the Ulanowicz-Fath regenerative economics framework. The organization's "
        f"network comprises <b>{n_nodes} nodes</b> and <b>{n_edges} directed connections</b>, "
        f"with a total system throughput of <b>{metrics['total_system_throughput']:.1f} units</b>."
    )
    story.append(Paragraph(exec_text, s_body))

    viab_text = (
        f"The system {'operates within' if metrics['is_viable'] else 'falls outside'} the "
        f"window of viability (α = {alpha:.3f}, bounds: {metrics['viability_lower_bound']:.2f}–"
        f"{metrics['viability_upper_bound']:.2f}), indicating "
        f"{'sustainable operational characteristics' if metrics['is_viable'] else 'need for structural adaptation'}. "
        f"Robustness of R = {rob:.3f} suggests {rob_status.lower()} resilience to perturbations."
    )
    story.append(Paragraph(viab_text, s_body))

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════
    # TABLE OF CONTENTS (simple)
    # ════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Table of Contents", s_h1))
    story.append(HRFlowable(
        width='100%', thickness=1, color=_hex_to_rgb(FOREST_GREEN),
        spaceBefore=0, spaceAfter=16,
    ))
    toc_items = [
        ("Executive Summary", ""),
        ("1. Introduction", ""),
        ("2. Methodology", ""),
        ("3. Results", ""),
        ("   3.1 Network Structure", ""),
        ("   3.2 Information-Theoretic Analysis", ""),
        ("   3.3 System Organization", ""),
        ("   3.4 Sustainability Assessment", ""),
        ("   3.5 Resilience Metrics", ""),
        ("   3.6 Flow Distribution", ""),
        ("4. OASIS Health Assessment", ""),
        ("5. Benchmarking & Position", ""),
        ("6. Risk & Resilience Analysis", ""),
        ("7. Prioritized Action Roadmap", ""),
        ("8. ESG Framework Mapping", ""),
        ("9. Discussion", ""),
        ("10. Conclusions & Recommendations", ""),
        ("References", ""),
        ("Appendix", ""),
    ]
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
        ['Lower Bound', f"{metrics['viability_lower_bound']:.3f}",
         'PASS' if alpha > metrics['viability_lower_bound'] else 'FAIL'],
        ['Upper Bound', f"{metrics['viability_upper_bound']:.3f}",
         'PASS' if alpha < metrics['viability_upper_bound'] else 'FAIL'],
        ['Within Window of Viability', 'Yes' if metrics['is_viable'] else 'No',
         'Sustainable' if metrics['is_viable'] else 'Needs attention'],
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
        "<i>Table 2. Viability Assessment</i> — Position of the organization relative to the empirically derived window of viability, indicating whether current efficiency-resilience dynamics are sustainable.", s_caption))

    # ── Charts ──
    # Professional figure caption mapping: chart_name -> interpretive note
    _figure_notes = {
        "System Robustness Curve": "The organization's position (red marker) relative to the theoretical robustness function R = -α·log(α), with the empirical optimum at α ≈ 0.37.",
        "Core Metrics Analysis": "Comparative bar chart of key information-theoretic indicators, enabling rapid identification of metrics that deviate from healthy-system benchmarks.",
        "Flow Distribution": "Distribution of resource flows across the top network nodes, illustrating concentration patterns and potential structural dependencies.",
    }
    if charts:
        fig_num = 1
        first_chart = True
        for chart_name, fig in charts.items():
            if fig is None:
                continue
            img = _chart_image(fig, width=CONTENT_W * 0.92, height=250)
            if img:
                note = _figure_notes.get(chart_name, f"Visualization of {chart_name.lower()} for the analyzed network.")
                chart_block = [
                    img,
                    Paragraph(
                        f"<i>Figure {fig_num}. {chart_name}</i> — {note}", s_caption),
                ]
                if first_chart:
                    # Keep section heading with the first chart
                    chart_block.insert(
                        0, Paragraph("3.3 Visualizations", s_h2))
                    first_chart = False
                story.append(KeepTogether(chart_block))
                fig_num += 1
        if first_chart:
            # No valid charts — still emit the heading
            story.append(Paragraph("3.3 Visualizations", s_h2))
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

    # Try to get OASIS data
    try:
        from oasis_calculator import OASISCalculator
        oasis = OASISCalculator(calculator)
        profile = oasis.get_oasis_profile()
        interpretations = oasis.get_oasis_interpretation()
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
                    rec_items.append(
                        f"<b>Metrics to improve:</b> {', '.join(metrics_to_improve)}")
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
            _oasis = _OC(calculator)
            _profile = _oasis.get_oasis_profile()
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
            story.append(Paragraph(
                "Published ecosystem values below are scientific reference points for "
                "the viability scale&mdash;not organizational targets.", s_body_italic))
            _anchor_data = [['Reference Network', 'Relative Ascendency (α)', 'Source']]
            for _a in _bench['reference_anchors']:
                _anchor_data.append([
                    Paragraph(_a['label'], cell_b),
                    Paragraph(f"{_a['relative_ascendency']:.3f}", cell_s),
                    Paragraph(_a.get('source', ''), cell_s)])
            if len(_anchor_data) > 1:
                _at = Table(_anchor_data, colWidths=[
                    CONTENT_W * 0.40, CONTENT_W * 0.30, CONTENT_W * 0.30])
                _at.setStyle(TableStyle([
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
                story.append(_at)
                story.append(Paragraph(
                    "<i>Table 5. Published reference networks (relative ascendency). "
                    "Shown as scientific reference points, not targets.</i>", s_caption))
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
                    _m = ', '.join(_it['metrics_to_improve']) or 'N/A'
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
