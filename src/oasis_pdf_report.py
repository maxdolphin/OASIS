"""
OASIS Professional PDF Report Generator

Generates publication-quality PDF reports for the OASIS (Open, Autonomous,
Symbiotic, Intelligent, Sustainable) organizational health assessment.

Design specifications:
- Light theme (white background) for professional printing
- A4 page size with consulting-firm aesthetics (McKinsey/BCG style)
- Color palette: Deep Forest Green (#1a5f35), Gold (#d4a843), Teal (#48c9b0)
- Professional serif body text, clean sans-serif headers
- Page numbers, headers/footers, alternating-row tables
- High-resolution embedded chart images

Based on:
- Fath et al. (2019) "Measuring regenerative economics"
- Ulanowicz et al. (2009) "Quantifying sustainability"
- Zorach & Ulanowicz (2003) "Quantifying complexity of flow networks"
"""

import base64
import math
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# DESIGN TOKENS
# ---------------------------------------------------------------------------

# -- Page Layout (A4: 210 x 297 mm) ----------------------------------------
PAGE = {
    "width_mm": 210,
    "height_mm": 297,
    "margin_top_mm": 25,
    "margin_bottom_mm": 20,
    "margin_left_mm": 25,
    "margin_right_mm": 25,
    "header_height_mm": 12,
    "footer_height_mm": 10,
    "content_width_mm": 160,     # 210 - 25 - 25
    "content_height_mm": 252,    # 297 - 25 - 20
    "gutter_mm": 6,              # space between columns when using 2-col
    "column_width_mm": 77,       # (160 - 6) / 2
}

# -- Color Palette ----------------------------------------------------------
COLORS = {
    # Primary accents
    "forest_green":   "#1a5f35",
    "gold":           "#d4a843",
    "teal":           "#48c9b0",

    # Extended palette (derived)
    "forest_light":   "#e8f5ec",   # 95% tint of forest green
    "forest_medium":  "#a3d4b5",   # 50% tint
    "gold_light":     "#faf3e0",   # 95% tint of gold
    "teal_light":     "#e0f7f1",   # 95% tint of teal
    "teal_dark":      "#1a8a6e",   # 30% shade of teal

    # Neutrals
    "black":          "#1a1a2e",   # near-black for body text
    "dark_gray":      "#3d3d56",   # secondary text
    "medium_gray":    "#6c6c80",   # captions, annotations
    "light_gray":     "#e8e8ee",   # table alt rows, rules
    "pale_gray":      "#f5f5f8",   # background fills
    "white":          "#ffffff",

    # Semantic (status indicators)
    "healthy":        "#1a5f35",   # forest green
    "warning":        "#d4a843",   # gold
    "critical":       "#c0392b",   # muted red (not neon)
}

# -- Typography Hierarchy ---------------------------------------------------
# Fonts: Headers use a clean sans-serif (Helvetica Neue / Arial / Inter)
#         Body uses a professional serif (Georgia / Charter / Palatino)
TYPOGRAPHY = {
    "cover_title": {
        "family": "sans-serif",
        "size_pt": 36,
        "weight": "bold",
        "color": COLORS["forest_green"],
        "line_height": 1.15,
        "letter_spacing": "-0.5px",
    },
    "cover_subtitle": {
        "family": "serif",
        "size_pt": 16,
        "weight": "normal",
        "color": COLORS["dark_gray"],
        "line_height": 1.4,
    },
    "h1_section": {
        "family": "sans-serif",
        "size_pt": 22,
        "weight": "bold",
        "color": COLORS["forest_green"],
        "line_height": 1.2,
        "margin_top_mm": 16,
        "margin_bottom_mm": 6,
        "border_bottom": f"2px solid {COLORS['forest_green']}",
    },
    "h2_subsection": {
        "family": "sans-serif",
        "size_pt": 15,
        "weight": "600",  # semibold
        "color": COLORS["forest_green"],
        "line_height": 1.25,
        "margin_top_mm": 10,
        "margin_bottom_mm": 4,
    },
    "h3_sub_subsection": {
        "family": "sans-serif",
        "size_pt": 12,
        "weight": "600",
        "color": COLORS["dark_gray"],
        "line_height": 1.3,
        "margin_top_mm": 6,
        "margin_bottom_mm": 3,
    },
    "body": {
        "family": "serif",
        "size_pt": 10.5,
        "weight": "normal",
        "color": COLORS["black"],
        "line_height": 1.55,
        "text_align": "justify",
    },
    "body_small": {
        "family": "serif",
        "size_pt": 9,
        "weight": "normal",
        "color": COLORS["dark_gray"],
        "line_height": 1.45,
    },
    "caption": {
        "family": "sans-serif",
        "size_pt": 8.5,
        "weight": "normal",
        "color": COLORS["medium_gray"],
        "line_height": 1.35,
        "style": "italic",
    },
    "metric_value": {
        "family": "sans-serif",
        "size_pt": 28,
        "weight": "bold",
        "line_height": 1.0,
    },
    "metric_label": {
        "family": "sans-serif",
        "size_pt": 9,
        "weight": "600",
        "color": COLORS["dark_gray"],
        "text_transform": "uppercase",
        "letter_spacing": "0.5px",
    },
    "table_header": {
        "family": "sans-serif",
        "size_pt": 9,
        "weight": "bold",
        "color": COLORS["white"],
    },
    "table_body": {
        "family": "serif",
        "size_pt": 9.5,
        "weight": "normal",
        "color": COLORS["black"],
    },
    "header_footer": {
        "family": "sans-serif",
        "size_pt": 7.5,
        "weight": "normal",
        "color": COLORS["medium_gray"],
    },
    "footnote": {
        "family": "serif",
        "size_pt": 8,
        "weight": "normal",
        "color": COLORS["medium_gray"],
    },
}

# -- Table Styling ----------------------------------------------------------
TABLE_STYLE = {
    "header_bg": COLORS["forest_green"],
    "header_text": COLORS["white"],
    "row_even_bg": COLORS["white"],
    "row_odd_bg": COLORS["pale_gray"],
    "border_color": COLORS["light_gray"],
    "border_width": "0.5px",
    "cell_padding": "6px 10px",
    "corner_radius": "0px",   # sharp corners for professional look
}

# -- KPI / Metric Card Styling ---------------------------------------------
METRIC_CARD = {
    "width_mm": 36,
    "height_mm": 28,
    "bg": COLORS["white"],
    "border": f"1px solid {COLORS['light_gray']}",
    "border_top_width": "3px",   # colored top accent
    "border_radius": "2px",
    "padding": "8px 6px",
    "shadow": "0 1px 3px rgba(0,0,0,0.06)",
}

# -- Chart Placement --------------------------------------------------------
CHART_GUIDELINES = {
    "max_width_single_col_mm": 160,       # full content width
    "max_width_half_col_mm": 77,          # half content width
    "max_height_single_col_mm": 110,      # ~43% of content height
    "dpi": 300,                            # high-res for print
    "format": "png",
    "margin_top_mm": 4,
    "margin_bottom_mm": 6,
    "caption_position": "below",
    "border": f"0.5px solid {COLORS['light_gray']}",
}


# ---------------------------------------------------------------------------
# HTML/CSS REPORT BUILDER
# ---------------------------------------------------------------------------

class OASISPDFReport:
    """
    Builds a professional PDF report as an HTML document that can be rendered
    to PDF via WeasyPrint, wkhtmltopdf, or browser print-to-PDF.

    The design follows consulting-firm report aesthetics:
    - Clean white background
    - Forest Green / Gold / Teal accent palette
    - Strong typographic hierarchy
    - Data-dense but visually organized layouts
    """

    def __init__(
        self,
        org_name: str,
        oasis_profile: Dict[str, Any],
        ulanowicz_metrics: Dict[str, Any],
        interpretations: Dict[str, str],
        recommendations: List[Dict[str, Any]],
        chart_images: Optional[Dict[str, bytes]] = None,
        logo_path: Optional[str] = None,
        analyst_name: str = "OASIS Analysis System",
        detailed: bool = True,
    ):
        """
        Initialize the report builder.

        Args:
            org_name: Name of the organization analyzed
            oasis_profile: Full OASIS profile from OASISCalculator.get_oasis_profile()
            ulanowicz_metrics: Extended metrics from UlanowiczCalculator.get_extended_metrics()
            interpretations: Dimension interpretations from get_oasis_interpretation()
            recommendations: Recommendations from get_recommendations()
            chart_images: Dict mapping chart name to PNG bytes (rendered at 300 dpi)
            logo_path: Optional file path to organization logo
            analyst_name: Name/label for the analyst or system
        """
        self.org_name = org_name
        self.profile = oasis_profile
        self.metrics = ulanowicz_metrics
        self.interpretations = interpretations
        self.recommendations = recommendations
        self.charts = chart_images or {}
        self.logo_path = logo_path
        self.analyst_name = analyst_name
        self.timestamp = datetime.now()
        self.page_number = 0

        self.detailed = detailed
        # Lazily computed report-intelligence views (built on existing data only)
        from src import report_intelligence as _ri
        self._ri = _ri
        if detailed:
            self.benchmark = _ri.build_benchmark_view(self.metrics, self.profile)
            self.risk = _ri.build_risk_view(self.metrics, self.profile)
            self.roadmap = _ri.build_action_roadmap(self.recommendations, self.profile)
            self.esg = _ri.build_esg_crosswalk(self.profile, self.metrics)
            # Render the WoV chart once, kept separate from self.charts so the Results
            # "Visualizations" loop does not render it a second time.
            try:
                self._wov_chart_png = _ri.render_window_of_viability_png(
                    self.benchmark['alpha'], self.benchmark['robustness'])
            except Exception:
                self._wov_chart_png = None
        else:
            self._wov_chart_png = None

    # ------------------------------------------------------------------
    # CSS STYLESHEET
    # ------------------------------------------------------------------
    def _build_css(self) -> str:
        """Generate the complete CSS stylesheet for the report."""
        c = COLORS
        t = TYPOGRAPHY
        ts = TABLE_STYLE

        return f"""
        /* ============================================================
           OASIS PDF REPORT STYLESHEET
           A4, light theme, consulting-firm aesthetics
           ============================================================ */

        @page {{
            size: A4;
            margin: {PAGE['margin_top_mm']}mm {PAGE['margin_right_mm']}mm
                    {PAGE['margin_bottom_mm']}mm {PAGE['margin_left_mm']}mm;

            @top-center {{
                content: none;
            }}
            @bottom-center {{
                content: counter(page);
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                font-size: {t['header_footer']['size_pt']}pt;
                color: {t['header_footer']['color']};
            }}
        }}

        @page :first {{
            margin: 0;
            @bottom-center {{ content: none; }}
        }}

        /* --- RESET & BASE ---------------------------------------- */
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: Georgia, 'Times New Roman', serif;
            font-size: {t['body']['size_pt']}pt;
            line-height: {t['body']['line_height']};
            color: {t['body']['color']};
            background: {c['white']};
            text-align: justify;
            -webkit-print-color-adjust: exact;
            print-color-adjust: exact;
            counter-reset: section figure table;
        }}

        /* --- AUTOMATIC SECTION / FIGURE / TABLE NUMBERING ------- */
        /* Numbered main sections (appendices opt out via .appendix). */
        h1:not(.appendix) {{
            counter-increment: section;
            counter-reset: subsection;
        }}
        h1:not(.appendix)::before {{
            content: counter(section) ". ";
        }}
        h2:not(.appendix) {{
            counter-increment: subsection;
        }}
        h2:not(.appendix)::before {{
            content: counter(section) "." counter(subsection) "\\00a0\\00a0";
        }}
        table caption {{
            counter-increment: table;
        }}
        table caption::before {{
            content: "Table " counter(table) ". ";
        }}
        .figure-caption {{
            counter-increment: figure;
        }}
        .figure-caption::before {{
            content: "Figure " counter(figure) ". ";
        }}

        /* --- COVER PAGE ------------------------------------------ */
        .cover-page {{
            width: 210mm;
            height: 297mm;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            page-break-after: always;
            position: relative;
            background: {c['white']};
        }}

        .cover-accent-bar {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 8mm;
            background: linear-gradient(90deg,
                {c['forest_green']} 0%,
                {c['forest_green']} 60%,
                {c['teal']} 80%,
                {c['gold']} 100%);
        }}

        .cover-accent-bottom {{
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 3mm;
            background: {c['forest_green']};
        }}

        .cover-logo {{
            margin-bottom: 20mm;
        }}

        .cover-logo img {{
            max-height: 25mm;
            max-width: 60mm;
        }}

        .cover-logo-placeholder {{
            width: 50mm;
            height: 18mm;
            border: 1.5px dashed {c['light_gray']};
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: {c['medium_gray']};
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 8pt;
            margin: 0 auto;
        }}

        .cover-title {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: {t['cover_title']['size_pt']}pt;
            font-weight: {t['cover_title']['weight']};
            color: {t['cover_title']['color']};
            line-height: {t['cover_title']['line_height']};
            letter-spacing: {t['cover_title']['letter_spacing']};
            margin-top: 8mm;
            padding: 0 30mm;
        }}

        .cover-rule {{
            width: 60mm;
            height: 2px;
            background: {c['gold']};
            margin: 8mm auto;
        }}

        .cover-subtitle {{
            font-family: Georgia, 'Times New Roman', serif;
            font-size: {t['cover_subtitle']['size_pt']}pt;
            color: {t['cover_subtitle']['color']};
            line-height: {t['cover_subtitle']['line_height']};
            padding: 0 40mm;
        }}

        .cover-org-name {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 20pt;
            font-weight: 600;
            color: {c['dark_gray']};
            margin-top: 12mm;
        }}

        .cover-meta {{
            position: absolute;
            bottom: 25mm;
            width: 100%;
            text-align: center;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 9pt;
            color: {c['medium_gray']};
            line-height: 1.6;
        }}

        /* --- PAGE HEADER / FOOTER -------------------------------- */
        .page-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 3mm;
            border-bottom: 0.5px solid {c['light_gray']};
            margin-bottom: 6mm;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: {t['header_footer']['size_pt']}pt;
            color: {t['header_footer']['color']};
        }}

        .page-header-left {{
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .page-header-right {{
            color: {c['forest_green']};
            font-weight: 600;
        }}

        /* --- SECTION HEADINGS ------------------------------------ */
        h1 {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: {t['h1_section']['size_pt']}pt;
            font-weight: {t['h1_section']['weight']};
            color: {t['h1_section']['color']};
            line-height: {t['h1_section']['line_height']};
            margin-top: {t['h1_section']['margin_top_mm']}mm;
            margin-bottom: {t['h1_section']['margin_bottom_mm']}mm;
            padding-bottom: 2mm;
            border-bottom: {t['h1_section']['border_bottom']};
            page-break-after: avoid;
        }}

        h2 {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: {t['h2_subsection']['size_pt']}pt;
            font-weight: {t['h2_subsection']['weight']};
            color: {t['h2_subsection']['color']};
            line-height: {t['h2_subsection']['line_height']};
            margin-top: {t['h2_subsection']['margin_top_mm']}mm;
            margin-bottom: {t['h2_subsection']['margin_bottom_mm']}mm;
            page-break-after: avoid;
        }}

        h3 {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: {t['h3_sub_subsection']['size_pt']}pt;
            font-weight: {t['h3_sub_subsection']['weight']};
            color: {t['h3_sub_subsection']['color']};
            line-height: {t['h3_sub_subsection']['line_height']};
            margin-top: {t['h3_sub_subsection']['margin_top_mm']}mm;
            margin-bottom: {t['h3_sub_subsection']['margin_bottom_mm']}mm;
            page-break-after: avoid;
        }}

        p {{
            margin-bottom: 3mm;
        }}

        /* --- EXECUTIVE SUMMARY KPI CARDS ------------------------- */
        .kpi-row {{
            display: flex;
            justify-content: space-between;
            gap: {PAGE['gutter_mm']}mm;
            margin: 6mm 0;
        }}

        .kpi-card {{
            flex: 1;
            background: {c['white']};
            border: 1px solid {c['light_gray']};
            border-top: 3px solid {c['forest_green']};
            border-radius: 2px;
            padding: 5mm 4mm;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }}

        .kpi-card.status-healthy {{
            border-top-color: {c['healthy']};
        }}
        .kpi-card.status-warning {{
            border-top-color: {c['warning']};
        }}
        .kpi-card.status-critical {{
            border-top-color: {c['critical']};
        }}

        .kpi-value {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: {t['metric_value']['size_pt']}pt;
            font-weight: {t['metric_value']['weight']};
            line-height: 1.0;
            margin-bottom: 2mm;
        }}

        .kpi-label {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: {t['metric_label']['size_pt']}pt;
            font-weight: {t['metric_label']['weight']};
            color: {t['metric_label']['color']};
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .kpi-sublabel {{
            font-family: Georgia, serif;
            font-size: 7.5pt;
            color: {c['medium_gray']};
            margin-top: 1mm;
        }}

        /* --- OASIS DIMENSION CARDS ------------------------------- */
        .dimension-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 4mm;
            margin: 4mm 0 6mm 0;
        }}

        .dimension-card {{
            border: 1px solid {c['light_gray']};
            border-radius: 2px;
            padding: 4mm 5mm;
            page-break-inside: avoid;
        }}

        .dimension-card-header {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 2mm;
        }}

        .dimension-name {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 11pt;
            font-weight: bold;
        }}

        .dimension-score {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 18pt;
            font-weight: bold;
        }}

        .dimension-tagline {{
            font-family: Georgia, serif;
            font-size: 8.5pt;
            font-style: italic;
            color: {c['medium_gray']};
            margin-bottom: 2mm;
        }}

        .dimension-bar-bg {{
            width: 100%;
            height: 4px;
            background: {c['light_gray']};
            border-radius: 2px;
            margin: 2mm 0;
        }}

        .dimension-bar-fill {{
            height: 100%;
            border-radius: 2px;
        }}

        .dimension-interpretation {{
            font-size: 8.5pt;
            color: {c['dark_gray']};
            line-height: 1.4;
        }}

        /* Full-width card for SUSTAINABLE (central dimension) */
        .dimension-card-full {{
            grid-column: 1 / -1;
        }}

        /* --- STATUS BADGE ---------------------------------------- */
        .status-badge {{
            display: inline-block;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 7pt;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 1mm 3mm;
            border-radius: 2px;
            color: {c['white']};
        }}
        .status-badge.healthy {{ background: {c['healthy']}; }}
        .status-badge.warning {{ background: {c['warning']}; }}
        .status-badge.critical {{ background: {c['critical']}; }}

        /* --- TABLES ---------------------------------------------- */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 4mm 0 6mm 0;
            font-size: {t['table_body']['size_pt']}pt;
            page-break-inside: avoid;
        }}

        thead th {{
            background: {ts['header_bg']};
            color: {ts['header_text']};
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: {t['table_header']['size_pt']}pt;
            font-weight: bold;
            text-align: left;
            padding: {ts['cell_padding']};
            border: none;
        }}

        tbody td {{
            padding: {ts['cell_padding']};
            border-bottom: {ts['border_width']} solid {ts['border_color']};
            font-family: Georgia, serif;
            vertical-align: top;
        }}

        tbody tr:nth-child(odd) {{
            background: {ts['row_odd_bg']};
        }}

        tbody tr:nth-child(even) {{
            background: {ts['row_even_bg']};
        }}

        table caption {{
            caption-side: bottom;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: {t['caption']['size_pt']}pt;
            font-style: italic;
            color: {t['caption']['color']};
            text-align: left;
            padding-top: 2mm;
        }}

        td.numeric {{
            font-family: 'Courier New', monospace;
            text-align: right;
        }}

        /* --- CHARTS / FIGURES ------------------------------------ */
        .figure {{
            margin: {CHART_GUIDELINES['margin_top_mm']}mm 0
                    {CHART_GUIDELINES['margin_bottom_mm']}mm 0;
            text-align: center;
            page-break-inside: avoid;
        }}

        .figure img {{
            max-width: 100%;
            height: auto;
            border: {CHART_GUIDELINES['border']};
        }}

        .figure-caption {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: {t['caption']['size_pt']}pt;
            font-style: italic;
            color: {t['caption']['color']};
            text-align: center;
            margin-top: 2mm;
        }}

        .figure-half {{
            display: inline-block;
            width: 48%;
            vertical-align: top;
            margin: 0 0.5%;
        }}

        /* --- CALLOUT BOX ----------------------------------------- */
        .callout {{
            background: {c['forest_light']};
            border-left: 3px solid {c['forest_green']};
            padding: 4mm 5mm;
            margin: 4mm 0;
            page-break-inside: avoid;
        }}

        .callout-title {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 10pt;
            font-weight: bold;
            color: {c['forest_green']};
            margin-bottom: 2mm;
        }}

        .callout-gold {{
            background: {c['gold_light']};
            border-left-color: {c['gold']};
        }}

        .callout-gold .callout-title {{
            color: #8a6d1b;
        }}

        /* --- RECOMMENDATION CARDS -------------------------------- */
        .recommendation {{
            border: 1px solid {c['light_gray']};
            border-radius: 2px;
            padding: 4mm 5mm;
            margin: 3mm 0;
            page-break-inside: avoid;
        }}

        .recommendation.priority-critical {{
            border-left: 4px solid {c['critical']};
        }}
        .recommendation.priority-high {{
            border-left: 4px solid #e67e22;
        }}
        .recommendation.priority-medium {{
            border-left: 4px solid {c['gold']};
        }}
        .recommendation.priority-low {{
            border-left: 4px solid {c['teal']};
        }}

        .recommendation-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2mm;
        }}

        .recommendation-dimension {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 10pt;
            font-weight: bold;
            color: {c['dark_gray']};
        }}

        .priority-tag {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 7pt;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 1mm 2.5mm;
            border-radius: 2px;
            color: {c['white']};
        }}
        .priority-tag.critical {{ background: {c['critical']}; }}
        .priority-tag.high {{ background: #e67e22; }}
        .priority-tag.medium {{ background: {c['gold']}; color: #4a3b0f; }}
        .priority-tag.low {{ background: {c['teal']}; }}

        /* --- REFERENCES ------------------------------------------ */
        .reference {{
            font-size: 9pt;
            line-height: 1.5;
            padding-left: 8mm;
            text-indent: -8mm;
            margin-bottom: 2mm;
        }}

        /* --- UTILITIES ------------------------------------------- */
        .page-break {{
            page-break-before: always;
        }}

        .text-small {{ font-size: 9pt; }}
        .text-muted {{ color: {c['medium_gray']}; }}
        .text-center {{ text-align: center; }}
        .text-right {{ text-align: right; }}
        .mt-4 {{ margin-top: 4mm; }}
        .mb-4 {{ margin-bottom: 4mm; }}
        .mb-8 {{ margin-bottom: 8mm; }}

        .two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: {PAGE['gutter_mm']}mm;
        }}
        """

    # ------------------------------------------------------------------
    # SECTION BUILDERS
    # ------------------------------------------------------------------

    def _build_cover_page(self) -> str:
        """Build the cover page HTML."""
        logo_html = ""
        if self.logo_path:
            logo_html = f'<div class="cover-logo"><img src="{self.logo_path}" alt="Logo"></div>'
        else:
            logo_html = '<div class="cover-logo"><div class="cover-logo-placeholder">LOGO</div></div>'

        return f"""
        <div class="cover-page">
            <div class="cover-accent-bar"></div>

            {logo_html}

            <div class="cover-title">
                OASIS Organizational<br>Health Assessment
            </div>

            <div class="cover-rule"></div>

            <div class="cover-subtitle">
                Ecosystemic Sustainability Analysis Using Network Theory<br>
                and Regenerative Economics Principles
            </div>

            <div class="cover-org-name">{_escape(self.org_name)}</div>

            <div class="cover-accent-bottom"></div>

            <div class="cover-meta">
                {self.timestamp.strftime('%B %d, %Y')}<br>
                Prepared by {_escape(self.analyst_name)}<br>
                OASIS Framework v1.0
            </div>
        </div>
        """

    def _build_executive_summary(self) -> str:
        """Build the Executive Summary section."""
        scores = self.profile['dimension_scores']
        overall = self.profile['overall_score']
        overall_status = self.profile['overall_status']
        dim_status = self.profile['dimension_status']

        # Overall KPI card
        status_class = overall_status.lower()
        status_color = COLORS.get(status_class, COLORS['dark_gray'])

        # Dimension KPI cards
        dim_labels = {
            'open': 'Open',
            'autonomous': 'Autonomous',
            'symbiotic': 'Symbiotic',
            'intelligent': 'Intelligent',
            'sustainable': 'Sustainable',
        }
        dim_taglines = {
            'open': 'Interconnectivity',
            'autonomous': 'Learning Capacity',
            'symbiotic': 'Integration Balance',
            'intelligent': 'Functional Diversity',
            'sustainable': 'Order-Freedom Balance',
        }

        kpi_cards = ""
        for dim_key in ['open', 'autonomous', 'symbiotic', 'intelligent', 'sustainable']:
            s = scores[dim_key]
            st = dim_status[dim_key].lower()
            sc = COLORS.get(st, COLORS['dark_gray'])
            kpi_cards += f"""
            <div class="kpi-card status-{st}">
                <div class="kpi-value" style="color: {sc};">{s:.0f}</div>
                <div class="kpi-label">{dim_labels[dim_key]}</div>
                <div class="kpi-sublabel">{dim_taglines[dim_key]}</div>
            </div>
            """

        # Count statuses
        healthy_n = sum(1 for s in dim_status.values() if s == 'HEALTHY')
        warning_n = sum(1 for s in dim_status.values() if s == 'WARNING')
        critical_n = sum(1 for s in dim_status.values() if s == 'CRITICAL')

        return f"""
        <div class="page-break"></div>

        <h1>Executive Summary</h1>

        <div class="callout">
            <div class="callout-title">Overall OASIS Health Score</div>
            <div style="display: flex; align-items: center; gap: 6mm;">
                <div style="font-family: 'Helvetica Neue', sans-serif;
                            font-size: 48pt; font-weight: bold;
                            color: {status_color};">
                    {overall:.0f}<span style="font-size: 20pt; color: {COLORS['medium_gray']};">/100</span>
                </div>
                <div>
                    <span class="status-badge {status_class}">{overall_status}</span>
                    <div class="text-small text-muted mt-4">
                        {healthy_n} healthy &middot; {warning_n} warning &middot; {critical_n} critical
                    </div>
                </div>
            </div>
        </div>

        <h2>Dimension Scores at a Glance</h2>
        <div class="kpi-row">
            {kpi_cards}
        </div>

        <h2>Key Findings</h2>
        <p>
            The OASIS assessment of <strong>{_escape(self.org_name)}</strong> reveals an
            overall health score of <strong>{overall:.0f}/100</strong>,
            classified as <strong>{overall_status}</strong>.
            Of the five assessment dimensions, {healthy_n} are in healthy range,
            {warning_n} require attention, and {critical_n} are critical.
        </p>
        {self._build_key_findings_bullets()}
        """

    def _build_key_findings_bullets(self) -> str:
        """Generate bullet-point key findings."""
        scores = self.profile['dimension_scores']
        dim_status = self.profile['dimension_status']
        details = self.profile['dimension_details']

        bullets = []

        # Strongest dimension
        best_dim = max(scores, key=scores.get)
        bullets.append(
            f"<strong>Strongest dimension:</strong> {best_dim.upper()} "
            f"({scores[best_dim]:.0f}/100) -- "
            f"{self.interpretations.get(best_dim, '')[:120]}..."
        )

        # Weakest dimension
        worst_dim = min(scores, key=scores.get)
        if dim_status[worst_dim] in ('WARNING', 'CRITICAL'):
            bullets.append(
                f"<strong>Area of concern:</strong> {worst_dim.upper()} "
                f"({scores[worst_dim]:.0f}/100) -- "
                f"{self.interpretations.get(worst_dim, '')[:120]}..."
            )

        # Viability window
        sust = details.get('sustainable', {}).get('metrics', {})
        alpha = sust.get('relative_ascendency', 0)
        is_viable = sust.get('is_viable', False)
        if is_viable:
            bullets.append(
                f"The organization operates <strong>within the Window of Viability</strong> "
                f"(alpha = {alpha:.3f}), indicating a sustainable balance between "
                f"efficiency and resilience."
            )
        else:
            direction = "over-constrained (too rigid)" if alpha > 0.6 else "under-organized (too flexible)"
            bullets.append(
                f"The organization operates <strong>outside the Window of Viability</strong> "
                f"(alpha = {alpha:.3f}), appearing {direction}."
            )

        html = "<ul style='margin: 3mm 0 3mm 6mm; line-height: 1.6;'>"
        for b in bullets:
            html += f"<li style='margin-bottom: 2mm;'>{b}</li>"
        html += "</ul>"
        return html

    def _build_benchmarking(self) -> str:
        """Benchmarking & position vs the Window of Viability and reference points."""
        b = self.benchmark
        pos_text = {
            'within': 'within the Window of Viability',
            'above': 'above the viability band (tending rigid / over-organized)',
            'below': 'below the viability band (tending chaotic / under-organized)',
        }.get(b['position'], 'undetermined')

        anchor_rows = ""
        for a in b['reference_anchors']:
            anchor_rows += f"""
            <tr>
                <td>{_escape(a['label'])}</td>
                <td class="numeric">{a['relative_ascendency']:.3f}</td>
                <td>{_escape(a['source'])}</td>
            </tr>"""
        if not anchor_rows:
            anchor_rows = '<tr><td colspan="3">No reference data available.</td></tr>'

        return f"""
        <div class="page-break"></div>
        <h1>Benchmarking &amp; Position</h1>
        <p>
            The organization's relative ascendency is
            <strong>&alpha; = {b['alpha']:.3f}</strong>, placing it {pos_text}
            (viable band {b['lower']}&ndash;{b['upper']}; robustness optimum
            &alpha; &asymp; {b['optimum']:.2f}). Distance to the robustness optimum is
            <strong>{b['distance_to_optimum']:.3f}</strong>.
        </p>
        {self._build_wov_figure()}
        <h2>Ecological Reference Points</h2>
        <p class="text-small text-muted">
            Published ecosystem values are shown as scientific reference points for the
            viability scale&mdash;not as organizational targets.
        </p>
        <table>
            <thead><tr><th>Reference Network</th>
                <th style="text-align:right;">Relative Ascendency (&alpha;)</th>
                <th>Source</th></tr></thead>
            <tbody>{anchor_rows}</tbody>
            <caption>Published reference networks (relative ascendency).</caption>
        </table>
        """

    def _build_wov_figure(self) -> str:
        """Render the Window-of-Viability chart as a figure block (Benchmarking)."""
        if not self._wov_chart_png:
            return ""
        b64 = base64.b64encode(self._wov_chart_png).decode('utf-8')
        caption = ('Window of Viability with the organization positioned on the '
                   'robustness curve.')
        return f"""
        <div class="figure">
            <img src="data:image/png;base64,{b64}" alt="window_viability">
            <div class="figure-caption">{caption}</div>
        </div>
        """

    def _build_risk_resilience(self) -> str:
        """Risk & resilience analysis section."""
        r = self.risk
        items_html = ""
        for it in r['items']:
            sev = _escape(it['severity'])
            items_html += f"""
            <div class="recommendation priority-{sev.lower()}">
                <div class="recommendation-header">
                    <span class="recommendation-dimension">{_escape(it['title'])}</span>
                    <span class="priority-tag {sev.lower()}">{sev}</span>
                </div>
                <p style="margin:1mm 0;"><strong>Evidence:</strong> {_escape(it['evidence'])}</p>
                <p style="margin:1mm 0;"><strong>Implication:</strong> {_escape(it['implication'])}</p>
            </div>"""
        return f"""
        <div class="page-break"></div>
        <h1>Risk &amp; Resilience Analysis</h1>
        <p>
            Overall fragility classification: <strong>{_escape(r['fragility'])}</strong>.
            Adaptive reserve indicators &mdash; overhead ratio
            {r['overhead_ratio']*100:.1f}%, redundancy {r['redundancy']:.3f}.
        </p>
        {items_html}
        """

    def _build_action_roadmap(self) -> str:
        """Prioritized action roadmap section."""
        def horizon_html(title, items):
            if not items:
                return f"<h2>{title}</h2><p class='text-muted'>No actions in this horizon.</p>"
            rows = ""
            for it in items:
                metrics_txt = ', '.join(it['metrics_to_improve']) or 'N/A'
                rows += f"""
                <div class="recommendation priority-{it['priority'].lower()}">
                    <div class="recommendation-header">
                        <span class="recommendation-dimension">{_escape(it['dimension'])}</span>
                        <span class="priority-tag {it['priority'].lower()}">{_escape(it['priority'])}</span>
                    </div>
                    <p style="margin:1mm 0; font-weight:600;">{_escape(it['issue'])}</p>
                    <p style="margin:1mm 0;">{_escape(it['action'])}</p>
                    <p class="text-small text-muted">Expected impact: {_escape(it['expected_impact'])}<br>
                       Metrics to improve: {_escape(metrics_txt)}</p>
                </div>"""
            return f"<h2>{title}</h2>{rows}"

        return f"""
        <div class="page-break"></div>
        <h1>Prioritized Action Roadmap</h1>
        {horizon_html('Immediate (0&ndash;3 months)', self.roadmap['immediate'])}
        {horizon_html('Short-Term (3&ndash;9 months)', self.roadmap['short_term'])}
        {horizon_html('Medium-Term (9&ndash;18 months)', self.roadmap['medium_term'])}
        """

    def _build_esg_mapping(self) -> str:
        """ESG framework mapping section (indicative)."""
        rows = ""
        for row in self.esg:
            rows += f"""
            <tr>
                <td><strong>{_escape(row['oasis_dimension'])}</strong><br>
                    <span class="text-small text-muted">{_escape(row['finding_summary'])}</span></td>
                <td>{_escape(row['gri_ref'])}</td>
                <td>{_escape(row['esrs_ref'])}</td>
                <td>{_escape(row['tcfd_ref'])}</td>
            </tr>"""
        return f"""
        <div class="page-break"></div>
        <h1>ESG Framework Mapping</h1>
        <p class="text-small text-muted">
            Indicative crosswalk linking OASIS findings to recognized disclosure
            frameworks. Provided for navigation and context only; not a compliance
            attestation.
        </p>
        <table>
            <thead><tr><th>OASIS Finding</th><th>GRI</th><th>ESRS / CSRD</th><th>TCFD</th></tr></thead>
            <tbody>{rows}</tbody>
            <caption>Indicative OASIS-to-ESG framework crosswalk.</caption>
        </table>
        """

    def _build_methodology(self) -> str:
        """Build the Methodology section."""
        return f"""
        <div class="page-break"></div>

        <h1>Methodology</h1>

        <h2>Theoretical Framework</h2>
        <p>
            The OASIS (Open, Autonomous, Symbiotic, Intelligent, Sustainable) assessment
            framework integrates Ulanowicz's ecosystem network analysis with Fath et al.'s
            (2019) ten principles of regenerative economics. Organizations are modeled as
            directed weighted flow networks, where nodes represent functional units and edges
            represent resource, information, or influence flows.
        </p>

        <h2>Information-Theoretic Foundations</h2>
        <p>
            System health is quantified through information-theoretic measures derived from
            the flow matrix <em>F</em>. The core decomposition follows Ulanowicz (1986):
        </p>
        <div class="callout" style="font-family: 'Courier New', monospace; font-size: 9.5pt; line-height: 1.7;">
            Development Capacity:&nbsp;&nbsp; C = TST &times; H<br>
            Ascendency:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; A = TST &times; AMI<br>
            Overhead:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &Phi; = C &minus; A<br>
            Robustness:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; R = &minus;&alpha; &times; log(&alpha;)<br>
            where &alpha; = A / C (relative ascendency)
        </div>

        <h2>OASIS Dimension Mapping</h2>
        <table>
            <thead>
                <tr>
                    <th>Dimension</th>
                    <th>Fath Principles</th>
                    <th>Primary Metrics</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>OPEN</strong></td>
                    <td>P1, P3, P4 (Circulation)</td>
                    <td>Connectance, Flow Diversity, Betweenness, Clustering</td>
                </tr>
                <tr>
                    <td><strong>AUTONOMOUS</strong></td>
                    <td>P2, P9 (Re-investment)</td>
                    <td>Finn Cycling Index, Reciprocity, AMI, Autocatalytic Index</td>
                </tr>
                <tr>
                    <td><strong>SYMBIOTIC</strong></td>
                    <td>P5, P8 (Mutualism)</td>
                    <td>Gini Coefficient, Modularity, Node Utilization, Mutualism</td>
                </tr>
                <tr>
                    <td><strong>INTELLIGENT</strong></td>
                    <td>P7, P10 (Diversity)</td>
                    <td>Number of Roles, Functional Diversity, Conditional Entropy</td>
                </tr>
                <tr>
                    <td><strong>SUSTAINABLE</strong></td>
                    <td>P6 (Resilience-Efficiency)</td>
                    <td>Robustness, Window of Viability, Alpha Optimality</td>
                </tr>
            </tbody>
            <caption>OASIS dimensions mapped to Fath et al. (2019) regenerative economics principles.</caption>
        </table>

        <h2>Scoring Methodology</h2>
        <p>
            Each dimension is scored on a 0&ndash;100 scale using weighted combinations of
            normalized underlying metrics. Dimension weights default to equal (20% each) and
            can be adjusted for organizational context. Status thresholds are calibrated per
            dimension based on ecological reference ranges.
        </p>
        """

    def _build_results_core_metrics(self) -> str:
        """Build the core network metrics results sub-section."""
        m = self.metrics

        rows = [
            ("Total System Throughput (TST)", f"{m.get('total_system_throughput', 0):.3f}", "flow units"),
            ("Average Mutual Information (AMI)", f"{m.get('average_mutual_information', 0):.3f}", "bits"),
            ("Flow Diversity (H)", f"{m.get('flow_diversity', 0):.3f}", "bits"),
            ("Conditional Entropy", f"{m.get('conditional_entropy', 0):.3f}", "bits"),
            ("Ascendency (A)", f"{m.get('ascendency', 0):.3f}", "flow x bits"),
            ("Development Capacity (C)", f"{m.get('development_capacity', 0):.3f}", "flow x bits"),
            ("Overhead (Phi)", f"{m.get('overhead', 0):.3f}", "flow x bits"),
            ("Relative Ascendency (alpha)", f"{m.get('ascendency_ratio', 0):.4f}", "ratio"),
            ("Relative Overhead", f"{m.get('overhead_ratio', 0):.4f}", "ratio"),
            ("Robustness (R)", f"{m.get('robustness', 0):.4f}", "ratio"),
            ("Redundancy", f"{m.get('redundancy', 0):.4f}", "ratio"),
            ("Network Efficiency", f"{m.get('network_efficiency', 0):.4f}", "ratio"),
            ("Effective Link Density", f"{m.get('effective_link_density', 0):.4f}", "ratio"),
            ("Connectance", f"{m.get('connectance', 0):.4f}", "ratio"),
            ("Trophic Depth", f"{m.get('trophic_depth', 0):.3f}", "levels"),
        ]

        tbody = ""
        for label, val, unit in rows:
            tbody += f"""
            <tr>
                <td>{label}</td>
                <td class="numeric">{val}</td>
                <td>{unit}</td>
            </tr>"""

        return f"""
        <h2>Core Network Metrics</h2>
        <table>
            <thead>
                <tr><th>Metric</th><th style="text-align:right;">Value</th><th>Unit</th></tr>
            </thead>
            <tbody>{tbody}</tbody>
            <caption>Core Ulanowicz network analysis metrics.</caption>
        </table>
        """

    def _build_results_oasis(self) -> str:
        """Build the OASIS health assessment results sub-section."""
        scores = self.profile['dimension_scores']
        dim_status = self.profile['dimension_status']
        details = self.profile['dimension_details']

        dim_meta = {
            'open': {'label': 'OPEN', 'tagline': 'Ability to Interconnect and Exchange',
                     'color': '#5dade2'},
            'autonomous': {'label': 'AUTONOMOUS', 'tagline': 'Ability to Learn and Encode Routines',
                           'color': '#bb8fce'},
            'symbiotic': {'label': 'SYMBIOTIC', 'tagline': 'Human-Machine Integration and Balance',
                          'color': '#58d68d'},
            'intelligent': {'label': 'INTELLIGENT', 'tagline': 'Leverage Diverse Intelligence Types',
                            'color': '#f5b041'},
            'sustainable': {'label': 'SUSTAINABLE', 'tagline': 'Balance Between Order and Freedom',
                            'color': COLORS['teal']},
        }

        # Build dimension cards (2x2 grid + 1 full-width for SUSTAINABLE)
        cards_html = '<div class="dimension-grid">'

        for dim_key in ['open', 'autonomous', 'symbiotic', 'intelligent']:
            meta = dim_meta[dim_key]
            s = scores[dim_key]
            st = dim_status[dim_key]
            interp = self.interpretations.get(dim_key, '')
            bar_color = COLORS.get(st.lower(), meta['color'])

            cards_html += f"""
            <div class="dimension-card">
                <div class="dimension-card-header">
                    <span class="dimension-name" style="color: {meta['color']};">{meta['label']}</span>
                    <span class="dimension-score" style="color: {bar_color};">{s:.0f}</span>
                </div>
                <div class="dimension-tagline">{meta['tagline']}</div>
                <div class="dimension-bar-bg">
                    <div class="dimension-bar-fill" style="width: {min(s, 100)}%; background: {meta['color']};"></div>
                </div>
                <span class="status-badge {st.lower()}">{st}</span>
                <div class="dimension-interpretation mt-4">{interp}</div>
            </div>
            """

        # SUSTAINABLE card (full width)
        s_meta = dim_meta['sustainable']
        s_score = scores['sustainable']
        s_status = dim_status['sustainable']
        s_interp = self.interpretations.get('sustainable', '')
        s_bar_color = COLORS.get(s_status.lower(), s_meta['color'])
        sust_metrics = details.get('sustainable', {}).get('metrics', {})

        cards_html += f"""
        <div class="dimension-card dimension-card-full">
            <div class="dimension-card-header">
                <span class="dimension-name" style="color: {s_meta['color']};">{s_meta['label']}</span>
                <span class="dimension-score" style="color: {s_bar_color};">{s_score:.0f}</span>
            </div>
            <div class="dimension-tagline">{s_meta['tagline']} (Central Dimension)</div>
            <div class="dimension-bar-bg">
                <div class="dimension-bar-fill" style="width: {min(s_score, 100)}%; background: {s_meta['color']};"></div>
            </div>
            <span class="status-badge {s_status.lower()}">{s_status}</span>
            <div class="dimension-interpretation mt-4">{s_interp}</div>
            <div class="mt-4 text-small text-muted">
                Alpha = {sust_metrics.get('relative_ascendency', 0):.3f} |
                Robustness = {sust_metrics.get('robustness', 0):.4f} |
                In Window: {'Yes' if sust_metrics.get('is_viable', False) else 'No'} |
                Fitness = {sust_metrics.get('fitness_for_evolution', 0):.4f}
            </div>
        </div>
        """

        cards_html += "</div>"

        return f"""
        <h2>OASIS Health Assessment</h2>
        <p>
            The five OASIS dimensions provide a multifaceted view of organizational health,
            each mapped to specific Fath et al. (2019) regenerative economics principles.
        </p>
        {cards_html}
        """

    def _build_results_charts(self) -> str:
        """Build the chart/visualization section of results."""
        if not self.charts:
            return ""

        html = "<h2>Visualizations</h2>"

        chart_captions = {
            'radar': 'OASIS dimension radar chart showing health profile across all five dimensions.',
            'sustainability_curve': 'Sustainability curve with organization position relative to the Window of Viability.',
            'flow_network': 'Network flow visualization showing inter-unit resource and information flows.',
            'dimension_bars': 'Comparative bar chart of OASIS dimension scores with status thresholds.',
            'window_viability': 'Window of Viability analysis showing robustness as a function of relative ascendency.',
            'heatmap': 'Flow matrix heatmap showing intensity of pairwise flows.',
        }

        for chart_name, img_bytes in self.charts.items():
            if img_bytes:
                b64 = base64.b64encode(img_bytes).decode('utf-8')
                caption = chart_captions.get(chart_name, chart_name)
                html += f"""
                <div class="figure">
                    <img src="data:image/png;base64,{b64}" alt="{chart_name}">
                    <div class="figure-caption">{caption}</div>
                </div>
                """

        return html

    def _build_results(self) -> str:
        """Build the complete Results section."""
        return f"""
        <div class="page-break"></div>

        <h1>Results</h1>

        {self._build_results_core_metrics()}

        <h2>Network Flow Analysis</h2>
        <p>
            The flow matrix reveals the directed exchange patterns between organizational
            units. Total System Throughput (TST) of
            <strong>{self.metrics.get('total_system_throughput', 0):.2f}</strong> quantifies
            the overall scale of activity. The relative ascendency
            (alpha = {self.metrics.get('ascendency_ratio', 0):.3f}) indicates the proportion of
            system capacity utilized for organized, efficient behavior, with the remainder
            ({self.metrics.get('overhead_ratio', 0)*100:.1f}%) maintained as overhead providing
            adaptive capacity and resilience.
        </p>

        {self._build_results_oasis()}

        {self._build_results_charts()}
        """

    def _build_discussion(self) -> str:
        """Build the Discussion and Recommendations section."""
        recs_html = ""
        if self.recommendations:
            for rec in self.recommendations:
                prio = rec.get('priority', 'MEDIUM').lower()
                recs_html += f"""
                <div class="recommendation priority-{prio}">
                    <div class="recommendation-header">
                        <span class="recommendation-dimension">{rec.get('dimension', 'N/A')}</span>
                        <span class="priority-tag {prio}">{rec.get('priority', 'MEDIUM')}</span>
                    </div>
                    <p style="margin: 1mm 0; font-weight: 600;">{rec.get('issue', '')}</p>
                    <p style="margin: 1mm 0;">{rec.get('action', '')}</p>
                    <p class="text-small text-muted">
                        Metrics to improve: {', '.join(rec.get('metrics_to_improve', ['N/A']))}
                    </p>
                </div>
                """
        else:
            recs_html = """
            <div class="callout">
                <div class="callout-title">No Critical Recommendations</div>
                <p>The organization shows healthy patterns across all OASIS dimensions.
                   Continue monitoring and maintaining current practices.</p>
            </div>
            """

        return f"""
        <div class="page-break"></div>

        <h1>Discussion &amp; Recommendations</h1>

        <h2>Interpretation of Findings</h2>
        <p>
            The OASIS assessment provides a multidimensional view of organizational health
            grounded in network theory and information-theoretic principles. The overall
            score of <strong>{self.profile['overall_score']:.0f}/100</strong> reflects
            the weighted balance across all five dimensions.
        </p>

        <h2>Strategic Recommendations</h2>
        {recs_html}

        <h2>Limitations</h2>
        <p>
            This analysis represents a point-in-time snapshot. Longitudinal analysis is
            recommended to track organizational evolution. The meaning of flows (information,
            resources, influence) affects interpretation. System boundary definitions and node
            aggregation levels influence computed metrics.
        </p>
        """

    def _build_references(self) -> str:
        """Build the References section."""
        return f"""
        <div class="page-break"></div>

        <h1>References</h1>

        <div class="reference">
            Fath, B. D., Fiscus, D. A., Goerner, S. J., Berea, A., &amp; Ulanowicz, R. E.
            (2019). Measuring regenerative economics: 10 principles and measures undergirding
            systemic economic health. <em>Global Transitions</em>, 1, 15&ndash;27.
            https://doi.org/10.1016/j.glt.2019.06.002
        </div>

        <div class="reference">
            Holling, C. S. (1973). Resilience and stability of ecological systems.
            <em>Annual Review of Ecology and Systematics</em>, 4(1), 1&ndash;23.
        </div>

        <div class="reference">
            Ulanowicz, R. E. (1986). <em>Growth and Development: Ecosystems
            Phenomenology</em>. Springer-Verlag, New York.
        </div>

        <div class="reference">
            Ulanowicz, R. E. (1997). <em>Ecology, the Ascendent Perspective</em>.
            Columbia University Press, New York.
        </div>

        <div class="reference">
            Ulanowicz, R. E. (2009). <em>A Third Window: Natural Life beyond Newton
            and Darwin</em>. Templeton Foundation Press.
        </div>

        <div class="reference">
            Ulanowicz, R. E., Goerner, S. J., Lietaer, B., &amp; Gomez, R. (2009).
            Quantifying sustainability: Resilience, efficiency and the return of
            information theory. <em>Ecological Complexity</em>, 6(1), 27&ndash;36.
        </div>

        <div class="reference">
            Zorach, A. C., &amp; Ulanowicz, R. E. (2003). Quantifying the complexity
            of flow networks: How many roles are there? <em>Complexity</em>, 8(3),
            68&ndash;76.
        </div>
        """

    def _build_appendix(self) -> str:
        """Build a lightweight appendix with the scoring weights."""
        details = self.profile.get('dimension_details', {})

        weight_rows = ""
        for dim_key in ['open', 'autonomous', 'symbiotic', 'intelligent', 'sustainable']:
            dim_detail = details.get(dim_key, {})
            weights = dim_detail.get('weights', {})
            for metric, w in weights.items():
                weight_rows += f"""
                <tr>
                    <td>{dim_key.upper()}</td>
                    <td>{metric.replace('_', ' ').title()}</td>
                    <td class="numeric">{w*100:.0f}%</td>
                </tr>"""

        return f"""
        <div class="page-break"></div>

        <h1 class="appendix">Appendix A: Scoring Weights</h1>

        <table>
            <thead>
                <tr><th>Dimension</th><th>Metric</th><th style="text-align:right;">Weight</th></tr>
            </thead>
            <tbody>{weight_rows}</tbody>
            <caption>Metric weights used in OASIS dimension scoring.</caption>
        </table>

        <h2 class="appendix">Dimension Weight in Overall Score</h2>
        <table>
            <thead>
                <tr><th>Dimension</th><th style="text-align:right;">Weight</th></tr>
            </thead>
            <tbody>
                <tr><td>OPEN</td><td class="numeric">{self.profile['weights'].get('open', 0.20)*100:.0f}%</td></tr>
                <tr><td>AUTONOMOUS</td><td class="numeric">{self.profile['weights'].get('autonomous', 0.20)*100:.0f}%</td></tr>
                <tr><td>SYMBIOTIC</td><td class="numeric">{self.profile['weights'].get('symbiotic', 0.20)*100:.0f}%</td></tr>
                <tr><td>INTELLIGENT</td><td class="numeric">{self.profile['weights'].get('intelligent', 0.20)*100:.0f}%</td></tr>
                <tr><td>SUSTAINABLE</td><td class="numeric">{self.profile['weights'].get('sustainable', 0.20)*100:.0f}%</td></tr>
            </tbody>
            <caption>Dimension weights for overall OASIS score (default: equal weighting).</caption>
        </table>
        {self._build_glossary()}
        """

    def _build_glossary(self) -> str:
        """Appendix B: glossary of core metrics (analyst reference)."""
        glossary_terms = [
            ('Total System Throughput (TST)', 'Sum of all flows; overall activity scale.'),
            ('Average Mutual Information (AMI)', 'Average constraint/organization per unit flow.'),
            ('Ascendency (A)', 'Organized power: TST &times; AMI.'),
            ('Development Capacity (C)', 'Upper bound on ascendency: TST &times; flow diversity.'),
            ('Overhead (&Phi;)', 'Reserve capacity C &minus; A; supports resilience.'),
            ('Relative Ascendency (&alpha;)', 'A / C; efficiency-vs-resilience balance.'),
            ('Robustness (R)', '&minus;&alpha;&middot;ln(&alpha;); maximized near &alpha; &asymp; 0.37.'),
            ('Window of Viability', 'Empirical sustainable band &alpha; &isin; [0.2, 0.6].'),
        ]
        glossary_rows = "".join(
            f"<tr><td>{t}</td><td>{d}</td></tr>"
            for t, d in glossary_terms
        )
        return f"""
        <div class="page-break"></div>
        <h1 class="appendix">Appendix B: Metric Glossary</h1>
        <table>
            <thead><tr><th>Metric</th><th>Definition</th></tr></thead>
            <tbody>{glossary_rows}</tbody>
            <caption>Glossary of core metrics.</caption>
        </table>
        """

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def generate_html(self) -> str:
        """
        Generate the complete HTML document for the OASIS PDF report.

        Returns:
            Complete HTML string ready for PDF rendering.
        """
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>OASIS Assessment &mdash; {_escape(self.org_name)}</title>
    <style>{self._build_css()}</style>
</head>
<body>

{self._build_cover_page()}
{self._build_executive_summary()}
{self._build_benchmarking() if self.detailed else ""}
{self._build_risk_resilience() if self.detailed else ""}
{self._build_action_roadmap() if self.detailed else ""}
{self._build_methodology()}
{self._build_results()}
{self._build_esg_mapping() if self.detailed else ""}
{self._build_discussion()}
{self._build_references()}
{self._build_appendix()}

</body>
</html>"""

    def generate_pdf(self) -> Optional[bytes]:
        """
        Generate PDF bytes from the HTML report.

        Tries WeasyPrint first, then pdfkit, then returns None.

        Returns:
            PDF file as bytes, or None if no PDF engine is available.
        """
        html = self.generate_html()

        try:
            from weasyprint import HTML
            return HTML(string=html).write_pdf()
        except ImportError:
            pass

        try:
            import pdfkit
            return pdfkit.from_string(html, False, options={
                'page-size': 'A4',
                'margin-top': f'{PAGE["margin_top_mm"]}mm',
                'margin-bottom': f'{PAGE["margin_bottom_mm"]}mm',
                'margin-left': f'{PAGE["margin_left_mm"]}mm',
                'margin-right': f'{PAGE["margin_right_mm"]}mm',
                'encoding': 'UTF-8',
                'enable-local-file-access': '',
            })
        except Exception:
            pass

        return None

    def save_html(self, filepath: str) -> str:
        """
        Save the HTML report to a file.

        Args:
            filepath: Output file path

        Returns:
            The filepath written to.
        """
        html = self.generate_html()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        return filepath


# ---------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------------

def _escape(text: str) -> str:
    """Escape HTML special characters."""
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


def generate_oasis_pdf_report(
    oasis_calculator,
    ulanowicz_calculator,
    org_name: str = "Organization",
    chart_images: Optional[Dict[str, bytes]] = None,
    logo_path: Optional[str] = None,
    output_path: Optional[str] = None,
    detailed: bool = True,
) -> Optional[bytes]:
    """
    Convenience function to generate a complete OASIS PDF report.

    Args:
        oasis_calculator: OASISCalculator instance
        ulanowicz_calculator: UlanowiczCalculator instance
        org_name: Organization name
        chart_images: Optional dict of chart_name -> PNG bytes
        logo_path: Optional path to logo image
        output_path: If provided, also saves HTML to this path

    Returns:
        PDF bytes if a PDF engine is available, else None.
        The HTML is always saved if output_path is given.
    """
    profile = oasis_calculator.get_oasis_profile()
    interpretations = oasis_calculator.get_oasis_interpretation()
    recommendations = oasis_calculator.get_recommendations()
    metrics = ulanowicz_calculator.get_extended_metrics()

    report = OASISPDFReport(
        org_name=org_name,
        oasis_profile=profile,
        ulanowicz_metrics=metrics,
        interpretations=interpretations,
        recommendations=recommendations,
        chart_images=chart_images,
        logo_path=logo_path,
        detailed=detailed,
    )

    if output_path:
        html_path = output_path.replace('.pdf', '.html')
        report.save_html(html_path)

    return report.generate_pdf()
