"""
Contextual Documentation UI Helpers for Streamlit.

Provides:
  - info_button(key)  — renders an ⓘ icon with tooltip + deep-link
  - metric_with_info(label, value, key, ...) — st.metric wrapper with ⓘ
  - render_documentation_page() — full searchable documentation panel
"""

import streamlit as st
from typing import Optional, Union

from src.docs_registry import DOCS, get_anchor, get_entries_by_category, CATEGORY_ORDER


# ---------------------------------------------------------------------------
# CSS injection (call once at app startup)
# ---------------------------------------------------------------------------
_CSS_INJECTED = False


def inject_docs_css():
    """Inject CSS for info buttons. Call once in your app entry point."""
    global _CSS_INJECTED
    if _CSS_INJECTED:
        return
    st.markdown("""
    <style>
    /* Info button — small circle with ⓘ */
    .info-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: rgba(46, 204, 113, 0.15);
        color: #2ecc71;
        font-size: 12px;
        font-weight: 600;
        cursor: help;
        text-decoration: none;
        margin-left: 4px;
        vertical-align: middle;
        position: relative;
        border: 1px solid rgba(46, 204, 113, 0.3);
        transition: all 0.2s ease;
        line-height: 1;
    }
    .info-btn:hover {
        background: rgba(46, 204, 113, 0.3);
        color: #58d68d;
        transform: scale(1.1);
    }

    /* Tooltip container */
    .info-tooltip-wrap {
        display: inline-block;
        position: relative;
        vertical-align: middle;
    }

    /* Tooltip text */
    .info-tooltip-wrap .info-tip-text {
        visibility: hidden;
        opacity: 0;
        width: 280px;
        background-color: #1e1e2e;
        color: #e0e0e0;
        text-align: left;
        border-radius: 8px;
        padding: 10px 12px;
        position: absolute;
        z-index: 9999;
        bottom: 125%;
        left: 50%;
        margin-left: -140px;
        font-size: 12.5px;
        line-height: 1.45;
        font-weight: 400;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
        transition: opacity 0.15s ease, visibility 0.15s ease;
        pointer-events: none;
    }
    .info-tooltip-wrap .info-tip-text::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -6px;
        border-width: 6px;
        border-style: solid;
        border-color: #1e1e2e transparent transparent transparent;
    }
    .info-tooltip-wrap:hover .info-tip-text {
        visibility: visible;
        opacity: 1;
    }

    /* Documentation page styles */
    .doc-entry {
        padding: 16px 20px;
        margin-bottom: 12px;
        border-radius: 8px;
        background: rgba(255,255,255,0.02);
        border-left: 3px solid #2ecc71;
    }
    .doc-entry h4 {
        margin: 0 0 8px 0;
        color: #2ecc71;
    }
    .doc-formula {
        background: rgba(0,0,0,0.2);
        padding: 8px 12px;
        border-radius: 4px;
        font-family: 'Computer Modern', serif;
        margin: 8px 0;
    }
    .doc-citation {
        font-size: 12px;
        color: #888;
        margin-top: 8px;
        padding-top: 8px;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
    .doc-oasis-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        margin-left: 6px;
    }
    </style>
    """, unsafe_allow_html=True)
    _CSS_INJECTED = True


# ---------------------------------------------------------------------------
# Tier 1: info_button — inline ⓘ with tooltip
# ---------------------------------------------------------------------------
def info_button(key: str, position: str = "inline") -> str:
    """
    Return HTML for an ⓘ info button with hover tooltip and deep-link.

    Args:
        key: Registry key from DOCS dict
        position: "inline" (default) or "after" for placement hint

    Returns:
        HTML string to embed via st.markdown(..., unsafe_allow_html=True)
    """
    entry = DOCS.get(key)
    if not entry:
        return ""

    tooltip = entry.get("tooltip", "")
    anchor = get_anchor(key)
    # Escape quotes for HTML attributes
    tooltip_safe = tooltip.replace('"', '&quot;').replace("'", "&#39;")

    html = (
        f'<span class="info-tooltip-wrap">'
        f'<a class="info-btn" href="#{anchor}" title="{tooltip_safe}">i</a>'
        f'<span class="info-tip-text">{tooltip_safe}</span>'
        f'</span>'
    )
    return html


def info_icon(key: str) -> None:
    """
    Render an ⓘ info button inline via st.markdown.

    Args:
        key: Registry key from DOCS dict
    """
    html = info_button(key)
    if html:
        st.markdown(html, unsafe_allow_html=True)


def label_with_info(label: str, key: str) -> str:
    """
    Return a label string with an appended ⓘ button for use in headers.

    Args:
        label: Display label text
        key: Registry key from DOCS dict

    Returns:
        HTML string: "Label ⓘ"
    """
    btn = info_button(key)
    return f"{label} {btn}"


# ---------------------------------------------------------------------------
# Tier 1+: metric_with_info — st.metric wrapper
# ---------------------------------------------------------------------------
def metric_with_info(
    key: str,
    value: Union[str, int, float],
    delta: Optional[Union[str, int, float]] = None,
    delta_color: str = "normal",
    label_override: Optional[str] = None,
    help_text: Optional[str] = None,
):
    """
    Display a Streamlit metric with an ⓘ info button underneath.

    Args:
        key: Registry key from DOCS dict
        value: Metric value to display
        delta: Optional delta value
        delta_color: Delta color mode
        label_override: Override the label from registry
        help_text: Override Streamlit's built-in help tooltip
    """
    entry = DOCS.get(key, {})
    label = label_override or entry.get("label", key)
    tooltip = help_text or entry.get("tooltip", "")

    st.metric(
        label=label,
        value=value,
        delta=delta,
        delta_color=delta_color,
        help=tooltip,
    )


# ---------------------------------------------------------------------------
# Tier 2: render_documentation_page — full in-app docs
# ---------------------------------------------------------------------------
def render_documentation_page():
    """
    Render the complete searchable documentation panel.

    This generates the in-app documentation section with anchors for every
    registered metric. Called from the main app as a tab or page.
    """
    inject_docs_css()

    st.markdown("## 📖 Metric & Concept Reference")
    st.markdown(
        "Every metric, score, and visualization in this application is "
        "explained below with its scientific source. Click any **ⓘ** button "
        "throughout the app to jump directly to the relevant entry."
    )

    # Search filter
    search = st.text_input(
        "🔍 Search documentation",
        placeholder="Type a metric name, keyword, or concept...",
        key="docs_search_input",
    )

    grouped = get_entries_by_category()

    for cat in CATEGORY_ORDER:
        entries = grouped.get(cat, [])
        if not entries:
            continue

        # Filter by search
        if search:
            search_lower = search.lower()
            filtered = [
                (k, e) for k, e in entries
                if search_lower in e.get("label", "").lower()
                or search_lower in e.get("tooltip", "").lower()
                or search_lower in e.get("definition", "").lower()
                or search_lower in k.lower()
            ]
            if not filtered:
                continue
            entries = filtered

        st.markdown(f"### {cat}")

        for key, entry in entries:
            anchor = get_anchor(key)
            label = entry.get("label", key)
            definition = entry.get("definition", "")
            interpret = entry.get("interpret", "")
            formula = entry.get("formula", "")
            citation = entry.get("citation", "")
            doi = entry.get("doi", "")
            oasis_map = entry.get("oasis_map", "")

            # Anchor point
            st.markdown(f'<div id="{anchor}"></div>', unsafe_allow_html=True)

            with st.expander(f"**{label}**", expanded=bool(search)):
                if definition:
                    st.markdown(definition)

                if formula:
                    st.latex(formula)

                if interpret:
                    st.markdown(f"**Interpretation:**\n\n{interpret}")

                if oasis_map:
                    st.markdown(f"**OASIS Dimension:** {oasis_map}")

                if citation:
                    cite_md = f"📚 *{citation}*"
                    if doi:
                        cite_md += f" — [DOI]({doi})"
                    st.caption(cite_md)

    st.markdown("---")
    st.caption(
        "This reference is auto-generated from the documentation registry. "
        "Every ⓘ button in the application links to the corresponding entry above."
    )
