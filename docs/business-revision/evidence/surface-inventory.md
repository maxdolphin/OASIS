# Surface Inventory

Master inventory of every OASIS surface that the business-revision audit must
score. References were captured directly from source via grep/read on branch
`feat/detailed-ecosystemic-report`. Two later tasks capture screenshots/PDFs of
these surfaces; the audit task scores each one for TechFlow and Balanced sample
organizations.

The in-app analysis results are rendered from the "🎯 Core Metrics" analysis
section (dispatched at `app.py:2263`). The exported PDF is produced by
`generate_pdf_report` in `src/pdf_generator.py` (wired into the app at
`app.py:4887`).

## Dashboard surfaces (in-app)

| ID | Surface | app.py ref | Audited (TechFlow) | Audited (Balanced) |
|----|---------|-----------|--------------------|--------------------|
| D1 | Core Metrics (header + KPIs) | app.py:2898 | ☐ | ☐ |
| D2 | Key Performance Indicators | app.py:2916 | ☐ | ☐ |
| D3 | Ulanowicz Core Metrics (computation flow expander) | app.py:2992 | ☐ | ☐ |
| D4 | Sustainability Assessment (Window of Viability + system health) | app.py:3110 | ☐ | ☐ |
| D5 | Window of Viability Bounds | app.py:3139 | ☐ | ☐ |
| D6 | Extended Network Metrics | app.py:3165 | ☐ | ☐ |
| D7 | Balance Indicators | app.py:3185 | ☐ | ☐ |
| D8 | Health Assessments | app.py:3219 | ☐ | ☐ |
| D9 | Network Roles & Functional Specialization | app.py:3236 | ☐ | ☐ |
| D10 | Overall System Health (visualizations tab) | app.py:2655 | ☐ | ☐ |
| D11 | Network Diagram | app.py:2682 | ☐ | ☐ |
| D12 | Interactive Sankey diagram | app.py:2801 (chart app.py:2837) | ☐ | ☐ |
| D13 | Window of Viability robustness curve | app.py:2843 (chart app.py:2845) | ☐ | ☐ |
| D14 | Multi-Metric Comparison radar chart | app.py:2848 (chart app.py:2851) | ☐ | ☐ |
| D15 | Network Analysis (topology / centrality / community / robustness) | app.py:4020 | ☐ | ☐ |
| D16 | System Health Dashboard (health radar) | app.py:4361 | ☐ | ☐ |
| D17 | OASIS Organizational Health Assessment (overall) | app.py:4463 | ☐ | ☐ |
| D18 | OASIS Dimension Status (radar) | app.py:4510 (radar app.py:4536) | ☐ | ☐ |
| D19 | OASIS Dimension Details (per-dimension gauges) | app.py:4587 | ☐ | ☐ |
| D20 | OASIS Recommendations | app.py:4782 | ☐ | ☐ |
| D21 | Analysis Report tab (in-app export/preview) | app.py:4853 | ☐ | ☐ |

Notes:
- The "System Health Radar" figure is created and titled at `app.py:4348`; it is
  surfaced under the "System Health Dashboard" subheader (D16, app.py:4361).
- OASIS radar uses `create_oasis_radar_chart` (imported at app.py:74) rendered at
  app.py:4536; per-dimension gauges use `create_all_dimension_gauges`/
  `create_dimension_gauge` (imported app.py:75-76) under D19.
- An alternate "Core Metrics" / "Sustainability Assessment" / "Balance
  Indicators" block also exists at app.py:3364 / 3388 / 3408 and a detailed
  Ulanowicz breakdown at app.py:3444-3573; these are secondary render paths for
  the same metrics and are covered by auditing D1-D8.

## Report surfaces (PDF)

All references are in `src/pdf_generator.py` (the report path wired into the app
at `app.py:4887`).

| ID | Surface | source ref | Audited (TechFlow) | Audited (Balanced) |
|----|---------|-----------|--------------------|--------------------|
| R1 | Cover page (title / org / branding) | src/pdf_generator.py:329 | ☐ | ☐ |
| R2 | Executive Summary (KPI table + narrative) | src/pdf_generator.py:380 | ☐ | ☐ |
| R3 | Table of Contents | src/pdf_generator.py:451 | ☐ | ☐ |
| R4 | 1. Introduction | src/pdf_generator.py:640 | ☐ | ☐ |
| R5 | 2. Methodology | src/pdf_generator.py:651 | ☐ | ☐ |
| R6 | 3. Results | src/pdf_generator.py:662 | ☐ | ☐ |
| R7 | 3.1 Core Network Metrics (table) | src/pdf_generator.py:669 | ☐ | ☐ |
| R8 | 3.2 Sustainability Assessment (table) | src/pdf_generator.py:746 | ☐ | ☐ |
| R9 | 3.3 Visualizations | src/pdf_generator.py:809 | ☐ | ☐ |
| R10 | 3.4 Flow Distribution Analysis | src/pdf_generator.py:815 | ☐ | ☐ |
| R11 | 4. OASIS Organizational Health Assessment | src/pdf_generator.py:864 | ☐ | ☐ |
| R12 | 4.1 Dimension Interpretations | src/pdf_generator.py:933 | ☐ | ☐ |
| R13 | 4.2 OASIS-Based Recommendations | src/pdf_generator.py:944 | ☐ | ☐ |
| R14 | 5. Benchmarking & Position | src/pdf_generator.py:1005 | ☐ | ☐ |
| R15 | 6. Risk & Resilience Analysis | src/pdf_generator.py:1049 | ☐ | ☐ |
| R16 | 7. Prioritized Action Roadmap | src/pdf_generator.py:1067 | ☐ | ☐ |
| R17 | 8. ESG Framework Mapping (GRI / ESRS-CSRD / TCFD crosswalk) | src/pdf_generator.py:1095 | ☐ | ☐ |
| R18 | 9. Discussion | src/pdf_generator.py:1135 | ☐ | ☐ |
| R19 | 10. Conclusions & Recommendations | src/pdf_generator.py:1146 | ☐ | ☐ |
| R20 | References | src/pdf_generator.py:1157 | ☐ | ☐ |
| R21 | Appendix: Detailed Data | src/pdf_generator.py:1182 | ☐ | ☐ |

Notes:
- Sections 5-8 (R14-R17) are generated from `src/report_intelligence.py`
  (`build_benchmark_view`, `build_risk_view`, `build_action_roadmap`,
  `build_esg_crosswalk`) and appended at src/pdf_generator.py:975-1125.
- Hypothesized "glossary appendix" (commit c6a0b88, CSS auto-numbering + glossary)
  belongs to the HTML/CSS detailed-report path, not the ReportLab `generate_pdf_report`
  path. The ReportLab PDF's terminal sections are R20 (References) and R21
  (Appendix: Detailed Data); no separate glossary appendix is emitted by
  `src/pdf_generator.py`. Flagged for the audit task to confirm whether a glossary
  appendix should be added to the printed PDF.
- Other report generators exist but are NOT the app's active PDF path:
  `src/publication_report.py` (imported app.py:69, HTML/markdown),
  `src/latex_report_generator.py` (imported app.py:70),
  `src/oasis_report.py`, `src/oasis_pdf_report.py`. Audit scope is the wired
  `generate_pdf_report` output above.
