# Design Spec — Detailed & Thorough Ecosystemic Sustainability Report

**Date:** 2026-06-12
**Status:** Draft for review
**Milestone:** 1 of N (Report depth/quality)
**Author:** OASIS development session

---

## 1. Context & Goal

The OASIS system (Streamlit app at `app.py`) analyzes organizations as directed weighted
flow networks and computes Ulanowicz information-theoretic metrics and OASIS health
dimensions. It already produces a professional PDF report
(`src/oasis_pdf_report.py`, WeasyPrint HTML→PDF) with an academic IMRaD structure.

**Overall product goal** (set by user): enable end users to self-provision a detailed,
thorough report on the ecosystemic sustainability of their organization, providing
organizational network data via CSV upload or OAuth connectors (Microsoft 365, Google
Workspace, Atlassian, Slack), deployed **self-hosted / single-org** with bring-your-own
credentials.

The overall goal spans several independent subsystems and has been **decomposed** into
milestones, each with its own spec → plan → implementation cycle:

| # | Milestone | Status |
|---|-----------|--------|
| **1** | **Detailed & thorough report (data-source independent)** | **This spec** |
| 2 | Self-service report wizard (guided CSV upload → validate → map → configure → generate) | Future |
| 3 | OAuth connectors (Microsoft 365, Google, Atlassian, Slack — self-hosted BYO credentials) | Future |

This spec covers **Milestone 1 only**: making the generated report substantially more
detailed and thorough, independent of how the data arrived.

### Scope decisions (confirmed with user)

- **Audience / depth:** Layered — a tight executive layer (verdict, scorecard,
  prioritized actions) plus deep analytical detail (full metric tables, methodology,
  benchmarking, citations) in later sections/appendices.
- **New content:** Benchmarking, Prioritized Action Roadmap, Risk & Resilience analysis.
  (Per-department/node diagnostics intentionally deferred — not in this milestone.)
- **Framework alignment:** Both — keep primary grounding in the science (Ulanowicz;
  Fath et al. 2019) **and** add an explicit mapping to recognized ESG reporting
  frameworks (GRI, ESRS/CSRD, TCFD).

### Hard constraint (project rule)

Per `CLAUDE.md`: **no scientific formula may be added or changed without peer-reviewed
support.** This milestone adds **zero new scientific formulas.** All new sections are
built from metrics that are *already computed* by `UlanowiczCalculator` /
`OASISCalculator`, plus qualitative narrative synthesis, threshold lookups already
defined in the codebase, and qualitative framework crosswalks.

---

## 2. Current State (what exists)

**Report generator:** `src/oasis_pdf_report.py`
- Class `OASISPDFReport(org_name, oasis_profile, ulanowicz_metrics, interpretations,
  recommendations, chart_images, logo_path, analyst_name)`.
- `generate_html()` assembles: Cover → 1. Executive Summary → 2. Methodology →
  3. Results (core metrics table, network flow analysis, OASIS cards, charts) →
  4. Discussion & Recommendations → 5. References → Appendix A (scoring weights).
- `generate_pdf()` renders via WeasyPrint (fallback pdfkit).
- Convenience entry point: `generate_oasis_pdf_report(oasis_calculator,
  ulanowicz_calculator, org_name, chart_images, logo_path, output_path)`.

**Data contract (already available — the inputs the new sections consume):**
- `OASISCalculator.get_oasis_profile()` → `dimension_scores`, `overall_score`,
  `overall_status`, `dimension_status`, `dimension_details` (per-dim `metrics`,
  `weights`), `weights`.
- `OASISCalculator.get_oasis_interpretation()` → per-dimension narrative strings.
- `OASISCalculator.get_recommendations()` → list of
  `{dimension, priority(CRITICAL/HIGH/MEDIUM/LOW), issue, action, metrics_to_improve}`,
  already priority-sorted.
- `UlanowiczCalculator.get_extended_metrics()` → 40+ metrics incl.
  `total_system_throughput`, `average_mutual_information`, `ascendency`,
  `development_capacity`, `overhead`, `ascendency_ratio` (α), `overhead_ratio`,
  `robustness`, `redundancy`, `is_viable`, `connectance`, `effective_link_density`,
  `flow_diversity`, `trophic_depth`, etc.

**Reference data for benchmarking:** `src/services/published_metrics_db.py` —
peer-reviewed reference networks (Cone Spring original/eutrophicated, Crystal River,
etc.) with published metric values, accessor functions `list_networks()`,
`get_network_info()`, `get_published_metric()`.

**Window of Viability (scientifically grounded benchmark, already in engine):**
relative ascendency α = A/C; viable band α ∈ [0.2, 0.6]; robustness
R = −α·ln(α) maximized near α ≈ 0.37. These are existing constants/derivations in the
codebase, not new formulas.

App wiring: the analysis page renders a `📕 PDF Report` `st.download_button` that calls
the convenience function (app.py ~4835).

---

## 3. Design

### 3.1 New module: `src/report_intelligence.py`

A pure-Python module of deterministic functions that transform the existing
profile + metrics into structured content for the new sections. **No scientific
formulas** — only narrative synthesis, classification against existing thresholds,
and reference lookups. Each function returns plain dicts/lists (no HTML), so it is
unit-testable in isolation and reusable by future in-app (non-PDF) views.

```
build_benchmark_view(metrics, profile) -> dict
    # alpha, robustness, distance-to-optimum (|alpha - 0.37|),
    # position vs viability band [0.2, 0.6], in/out flag,
    # ecological reference anchors pulled from published_metrics_db
    #   (each labelled "scientific reference point, not a target")

build_risk_view(metrics, profile) -> dict
    # fragility classification from alpha position:
    #   alpha < 0.2  -> "under-organized / chaotic"  (too much redundancy)
    #   alpha > 0.6  -> "over-organized / brittle"    (too much efficiency)
    #   else          -> "within viable balance"
    # buffer indicators: overhead_ratio, redundancy (adaptive reserve)
    # distance from each viability bound
    # dimension-level critical/warning flags from dimension_status
    # returns ordered list of risk items {severity, title, evidence, implication}

build_action_roadmap(recommendations, profile) -> dict
    # sequences existing get_recommendations() into horizons:
    #   Immediate  <- CRITICAL
    #   Short-term <- HIGH
    #   Medium-term<- MEDIUM/LOW
    # each item carries dimension, issue, action, metrics_to_improve,
    # and a qualitative expected-impact note derived from which dimension/
    # metric it targets (lookup table, no scoring math)

build_esg_crosswalk(profile, metrics) -> list
    # qualitative mapping of OASIS findings to disclosure areas:
    #   GRI (e.g. 2-x governance, 3-x material topics),
    #   ESRS/CSRD (e.g. ESRS 2 governance, resilience of strategy),
    #   TCFD (governance, risk management, resilience/scenario)
    # each row: {oasis_dimension, finding_summary, gri_ref, esrs_ref, tcfd_ref}
    # NOTE: this is an interpretive crosswalk for navigation/credibility,
    # explicitly captioned as indicative, not a compliance attestation.

executive_verdict(profile) -> str
    # one-sentence plain-language overall verdict for the exec layer
```

All thresholds (0.2, 0.6, 0.37, status bands) are sourced from existing definitions in
`ulanowicz_calculator.py` / `oasis_calculator.py` — imported or referenced, not
re-invented.

### 3.2 Report structure (layered)

`OASISPDFReport` gains new `_build_*` methods, wired into `generate_html()` in this order:

| § | Section | Source | New? |
|---|---------|--------|------|
| — | Cover | existing | |
| 1 | Executive Summary (+ one-line verdict, top-3 actions) | existing + `executive_verdict`, `build_action_roadmap` | enhanced |
| 2 | Benchmarking & Position | `build_benchmark_view` + benchmark chart | **new** |
| 3 | Risk & Resilience Analysis | `build_risk_view` | **new** |
| 4 | Prioritized Action Roadmap | `build_action_roadmap` | **new** |
| 5 | Methodology | existing | |
| 6 | Detailed Results (metric tables, OASIS cards, charts) | existing | |
| 7 | ESG Framework Mapping | `build_esg_crosswalk` | **new** |
| 8 | Discussion & Limitations | existing (renumbered) | |
| 9 | References (+ GRI/ESRS/TCFD citations) | existing + additions | enhanced |
| A | Appendix A: Scoring Weights | existing | |
| B | Appendix B: Full Metric Glossary | `docs_registry` definitions | **new** |

The executive layer (Cover + §1) is self-contained for a board reader; §2–4 give the
"what does it mean / what do we do" depth; §5–7 + appendices give the analyst the full
rigor.

### 3.3 New visualization

One new chart for the Benchmarking section: the **Window of Viability curve** with the
organization's (α, robustness) point marked relative to the viable band and the
robustness-optimum. A window-of-viability plotting routine already exists in the
codebase (`validation/` and the app's `window_viability` chart) and will be reused /
adapted to emit PNG bytes into the existing `chart_images` dict — no new plotting math.

### 3.4 Integration & backward compatibility

- `generate_oasis_pdf_report(...)` gains an optional `detailed: bool = True` parameter.
  When `True` (default) the report includes the new sections; `False` reproduces the
  current lean report. This keeps the app's existing download button working and lets us
  ship the richer report as the default with a safe fallback.
- `OASISPDFReport.__init__` gains optional precomputed-intelligence params (or computes
  them lazily from the already-passed profile/metrics) so no extra data must be threaded
  through the app.
- The app's PDF button label/flow is unchanged in this milestone; richer content appears
  automatically.

---

## 4. Data Flow

```
OASISCalculator ──get_oasis_profile()──────┐
                ──get_oasis_interpretation()┤
                ──get_recommendations()─────┤
UlanowiczCalc   ──get_extended_metrics()────┤
                                            ▼
                              src/report_intelligence.py
                        (benchmark / risk / roadmap / esg / verdict)
                                            ▼
                              OASISPDFReport._build_* sections
                                            ▼
                       generate_html() → generate_pdf() (WeasyPrint)
                                            ▼
                          st.download_button("📕 PDF Report")
```

---

## 5. Error Handling

- Every `report_intelligence` function is total: missing metric keys default via
  `.get(key, default)` exactly as the current report does, so a sparse metric dict never
  raises.
- `build_benchmark_view` degrades gracefully if `published_metrics_db` lookups return
  `None` (omit the anchor row rather than fail).
- ESG crosswalk is static/qualitative and cannot fail on data.
- PDF engine fallback (WeasyPrint → pdfkit → None) is unchanged.

---

## 6. Testing

- `tests/test_report_intelligence.py` — unit tests with fixed metric/profile fixtures
  asserting deterministic structure and correct classification at boundary α values
  (0.19, 0.2, 0.37, 0.6, 0.61), horizon bucketing of recommendations, and graceful
  handling of missing keys.
- `tests/test_report_sections.py` — smoke test: build `OASISPDFReport` from a bundled
  sample dataset (e.g. an existing `data/` network), call `generate_html()`, assert all
  new section headings are present and HTML is well-formed.
- Reuse an existing sample network so the test needs no fixtures of its own.

---

## 7. Out of Scope (this milestone)

- CSV upload wizard (Milestone 2).
- OAuth connectors / live data pull (Milestone 3).
- Per-department / node-level diagnostics.
- Multi-period / longitudinal trend analysis.
- Any change to scientific formulas or scoring weights.
- ESG compliance attestation (the crosswalk is indicative navigation only).

---

## 8. Open Questions

None blocking. The ESG crosswalk reference codes (specific GRI/ESRS/TCFD clause numbers)
will be drafted conservatively and clearly captioned as indicative; they can be refined
later with a sustainability-reporting specialist review.
