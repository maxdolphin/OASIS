# OASIS Business Revision — Impact × Effort Redesign Roadmap

The prescription half of the review. Converts the ten ranked gaps in
`scored-matrix.md` Section C into prioritized, **presentation-only** recommendations
across three horizons. Every recommendation traces to at least one gap; no formula,
threshold, or coefficient is touched anywhere in this document.

**Sorting logic.**
- **Effort** = *presentation-layer tweak* (copy, colour, label, band, caption, table
  cell, section title) vs. *structural information-architecture change* (embedding a
  render pipeline, re-sequencing the document, building a new data pipeline). Formula
  work is explicitly out of scope and is never counted as effort here.
- **Impact** = weighted by **Decision relevance** (dim 1, the audit tiebreaker) and
  **Credibility/defensibility** (dim 5) — the two dimensions on which the
  operator→exec handoff succeeds or fails. A board-facing trust-killer outranks an
  analyst-only legibility miss.

**Horizon definitions.**
- **Immediate** — high-impact, low-effort. Copy/colour/label/band fixes shippable
  "this week" with no IA change.
- **Short-term** — high-impact, moderate-effort. Structural IA / render-pipeline /
  re-sequencing work; no new data.
- **Medium-term** — high-impact, higher-effort. New reference data, finding-specific
  crosswalk logic, or a peer-cohort data pipeline.

Recommendations are numbered **R1…R17 continuously** across all three horizons.
Gap IDs (Gap #1…#10) refer to `scored-matrix.md` Section C.

---

## Part 1 — Recommendations by horizon

### Horizon 1 — Immediate (high-impact, low-effort — "this week")

| # | Recommendation | Traces to gap(s) | Surface(s) | Business impact (H/M/L) | Effort | Notes |
|---|----------------|------------------|------------|:----------------------:|--------|-------|
| **R1** | **Reconcile the two headline verdicts into ONE.** Demote OASIS "Overall Health __/100 HEALTHY" from a co-equal headline to a *named sub-component*, and let the viability/SUSTAINABLE-pillar verdict lead. Relabel the OASIS banding text so a system that is Non-Viable cannot simultaneously read "HEALTHY" as its top line. | Gap #1 | D17, D18, R11, R12; echoed D21, R21/A2 | **H** | Presentation (copy + label + layout order) | The #1/#2 gap in all three audits; a 30-second, deal-killing self-contradiction. **The roll-up *weighting* that lets 3×100 outvote a CRITICAL pillar is a formula-validator hand-off (see Part 2); here we only reconcile the two verdicts on-screen.** |
| **R2** | **Fix the green "Non-Viable" → red.** Correct the traffic-light colour on the exec-summary verdict and any mirroring in-app chips/up-arrows so a *failure* verdict never renders in a success colour. | Gap #9 | R2; D21 (green ▲ up-arrows) | **H** | Presentation (colour token) | Pure layout; a failure verdict in green is an immediate trust tell on the one page the board reads. |
| **R3** | **Fix the "Non-Viabl/e" line-split, the mis-numbered §9/§10 headings, and leaked variable names.** Un-split the hyphenated word; renumber §9 sub-headers (currently "4.1/4.2/4.3") and §10 sub-headers (currently "5.1/5.2/5.3") to match their parent sections; replace leaked identifiers (`relative_ascendency`, `number_of_roles`) with human labels. | Gap #8, Gap #9 | R2, R3, R18, R19; D20/R13 (leaked names) | **H** | Presentation (text/label) | "Draft, unproofed" tells that undercut everything downstream before the content is read. |
| **R4** | **Add the α reference band + a one-line "so-what" under each headline metric.** Print each headline metric against its implemented band (α viability 0.2–0.6, robustness optimum ≈0.37 = 1/e, org anchor 0.30–0.45 Fath 2019) with one plain-language consequence sentence, per the per-metric table in `benchmarking-model.md` Part 2. | Gap #7, Gap #3 | D3, D6, D7, D9, D15, R7, R8, R10–R14; D1, D2 | **H** | Presentation (band overlay + caption copy) | Directly attacks the most *structurally* pervasive gap (dim 4 Benchmark/context, 🟥 across both families). Bands are read from code, not invented; no threshold change. |
| **R5** | **Stop printing α and ascendency-unit bounds in the same table.** Separate the 0–1 α ratio from the raw ascendency-unit Window bounds (2756.558 / 8269.674) so the central viability exhibit never shows "0.066 vs 2756" side by side. Render α against the α-scale band; show the unit bounds (if kept) in their own clearly-labelled scale panel. | Gap #5 | R8; recurs R2, R18 | **H** | Presentation (table split / relabel) | A CFO spots "0.066 cannot be below 2756" in five seconds. **Units/scale correctness and the "Lower FAIL / Upper PASS" coherence are formula-validator (Part 2); here we only stop mixing two scales in one exhibit.** |
| **R6** | **Promote the Fath 2019 org anchor into the §5 benchmark table; demote wetlands to a footnote.** Put "High-performing organizations: α ≈ 0.30–0.45 (Fath et al., 2019)" at the top of the §5 exhibit as the headline comparator (it already drives the on-screen Optimal/Warning verdict), and move Cone Spring / Crystal River / Florida Bay to a small "how the scale was validated in ecology" methodology note. | Gap #3, Gap #6 | R14; mirrored D5 | **H** | Presentation (table content re-order) | The org anchor already exists in code (`pdf_generator.py:408/750`, `latex_report_generator.py:275`); this only promotes an existing comparator and demotes the "compared to a swamp" ridicule risk. No data added. |

**Horizon 1 count: 6 recommendations (R1–R6).**

---

### Horizon 2 — Short-term (high-impact, moderate-effort)

| # | Recommendation | Traces to gap(s) | Surface(s) | Business impact (H/M/L) | Effort | Notes |
|---|----------------|------------------|------------|:----------------------:|--------|-------|
| **R7** | **Embed the visualizations into the PDF.** Render the network diagram, Window-of-Viability robustness curve, OASIS radar, Sankey, and gauge charts into the ReportLab path so §3.3 stops rendering zero images. Each figure carries a finding caption (not a bare title). | Gap #4 | R9 (Avg 1.1, lowest surface); lifts Visual across R1–R21 | **H** | Structural IA (render pipeline into PDF) | The single biggest report miss — an ecological-flow diagnosis whose thesis *is* a picture is currently delivered as prose. `pdfimages -list` confirms zero embedded images today. |
| **R8** | **Restructure to an exec one-pager with analyst depth gated behind a divider.** Build the 5-element one-pager (PM Q4): (1) one reconciled headline verdict + business consequence; (2) 3–4 KPI cards with target anchors; (3) the "you are here" WoV/robustness curve, captioned; (4) top-3 risks in Evidence→Implication form; (5) the prioritized roadmap. Demote the 12-row metrics table, extended metrics, flow stats, the redundant radars, and appendix A2 behind a "for your analyst" divider. | Gap #7, Gap #1 | R7, D6, R10, D14/D16/D18 (radars), R21/A2; consumes R1's reconciled verdict | **H** | Structural IA (re-layout + gating) | Overload buries the ~5 things an exec needs; verdict is restated 5+ times, three near-identical radars. Depends on R1 for the single reconciled verdict. |
| **R9** | **Promote the "why ecosystem math applies to your org" justification to the cover / first exec page, and add an in-app equivalent.** Lift the §1.1/§1.2 analogy off page 4 into a one-paragraph "Why this applies to your organization" on the cover/first page, led by organizational (Fath 2019) — not wetland — validation, and add the same paragraph as an in-app panel where the ecological vocabulary first appears. | Gap #2 | R4 (only place argued today); app-wide absence | **H** | Structural IA (content promotion + new in-app panel) | The entire product's authority rests on this one leap; a skeptical CFO's first question currently has no answer they will reach. |
| **R10** | **Rebuild the TOC to match real headings, with page numbers.** Regenerate the Table of Contents from the actual body headings ("3.1 Core Network Metrics," "3.2 Sustainability Assessment," etc.) and add page numbers, so the TOC describes its own document. | Gap #8 | R3 (Avg 1.6, second-lowest); R6 body jump | **H** | Structural IA (generated-TOC wiring) | A TOC matching no real heading and carrying no page numbers is an immediate auto-assembled-and-unproofed tell. |
| **R11** | **Apply the gradient-not-pass/fail reframe to the viability verdict.** Render the α axis with three zones (← diffuse/chaotic <0.2 · viable 0.2–0.6, sweet spot ≈0.37 · rigid/brittle >0.6 →), plot the org's dot, and state the direction of travel ("left of the band — coordination diffuse; add structure to move toward it"). Replace FAIL/PASS words with position + move; anchor the destination on the Fath 2019 org band; carry the calibration caveat as one honesty line. Uses existing `position` / `distance_to_optimum` outputs. | Gap #6, Gap #1 | D4, D17/R11, D5, R8 | **H** | Structural IA (gradient rendering + copy) | Converts a near-guaranteed "you fail" into actionable guidance. **Whether the food-web-calibrated bounds are valid for org networks is formula-validator (Part 2); here we only render the existing position as a gradient and add an honesty caveat.** |

**Horizon 2 count: 5 recommendations (R7–R11).**

---

### Horizon 3 — Medium-term (high-impact, higher-effort)

| # | Recommendation | Traces to gap(s) | Surface(s) | Business impact (H/M/L) | Effort | Notes |
|---|----------------|------------------|------------|:----------------------:|--------|-------|
| **R12** | **Add Tier-2 reference anchors from the 22 shipped datasets as illustrative "you-are-here" positions — led by the human-system networks.** Plot cross-domain, non-wetland anchors (`us_airport_network`, `manufacturing_network`, `pharma_development_network`, `dblp_coauthorship_network`) on the α line as "same math, other domains" illustration, each labelled "illustrative reference point — not an organizational target." | Gap #3 | R14; D5 | **M** | Higher-effort (wire runtime lookups for more anchors + new exhibit) | An airport or supply-chain network is a more intuitive analog to an org than a marsh; the datasets exist (`data/ecosystem_samples/*.json`, published α via `published_metrics_db`) but are unused as anchors. Still illustrative, not targets — no peer claim. |
| **R13** | **Replace the one-to-one ESG code lookup with a finding-specific crosswalk.** For each *finding* (not each dimension), attach disclosure text, the relevant data-point / materiality logic, and the matching GRI/ESRS/TCFD reference, retiring the stretch mappings (e.g. Window-of-Viability → GRI 201-2). Keep the "indicative, not a compliance attestation" caveat. | Gap #10 | R17 | **M** | Higher-effort (finding-driven crosswalk content + logic) | Today it is box-ticking in the buyer's own language; it will not survive a sustainability lead's review and risks an ESG-washing charge. |
| **R14** | **Plan the Tier-3 anonymized peer-cohort benchmark (data pipeline + minimum-N gating).** Specify the pipeline to run real organizations through the identical OASIS pipeline, tagged by size band × sector, with honest N-gating: N ≥ 30 per (sector × size) cell before quoting quartiles/percentiles; N ≥ 8–10 before a coarse below/around/above-median band; below that, plot individual anonymized points, not a distribution. Until it ships, the section stays titled "Position relative to the theoretical viability range," never "Benchmarking." | Gap #3, Gap #6 | R14; D5 | **H** | Higher-effort (data pipeline, cohort ingestion, percentile logic) | Fake peer averages are rejected — a fabricated benchmark manufactures unearned authority (the product's #1 risk). Reserves the word "benchmark" for a real cohort with percentiles. |

**Horizon 3 count: 3 recommendations (R12–R14).**

> **Coverage note.** Gaps #1–#10 are all addressed within the three horizons above.
> R15–R17 were not required; numbering stops at R14. (No orphan recommendations
> exist; every row references a Section C gap.)

**Totals: Immediate 6 · Short-term 5 · Medium-term 3 · 14 recommendations.**

---

## Part 2 — Formula-guardrail check

The audit marked four findings whose **root cause is math / calibration**
("(root: formula-validator)" in `scored-matrix.md`). For each, the **presentation
fix recommended in Part 1** is stated, and the **separate math question is handed to
formula-validator — NOT actioned in this review.**

| Root-cause finding | Presentation fix in this roadmap | Math question handed to formula-validator (not actioned here) |
|--------------------|----------------------------------|----------------------------------------------------------------|
| **HEALTHY-vs-Non-Viable roll-up weighting** (Gap #1) — the weighted OASIS roll-up lets three 100/100 pillars outvote a CRITICAL SUSTAINABLE pillar, and 46/49 scores band as "HEALTHY." | **R1** (reconcile the two verdicts into one headline; demote OASIS overall to a named sub-component; relabel the banding text) and **R8** (one-pager carries the single reconciled verdict). | Whether the roll-up **weighting** should allow high pillars to mask a CRITICAL pillar, and whether the HEALTHY **banding thresholds** are calibrated correctly. → formula-validator. |
| **α-vs-bounds scale / units** (Gap #5) — a 0–1 α ratio (0.066) judged against Window bounds in ascendency units (2756.558 / 8269.674), with a "Lower FAIL / Upper PASS" status for a system declared below the window; §6 quotes the lower bound as 0.2. | **R5** (stop printing α and ascendency-unit bounds in the same table; render α against the α band, unit bounds in their own labelled panel). | Whether the bounds are computed in the right **units**, and whether "Lower FAIL / Upper PASS" is a **coherent** status for a below-window system. → formula-validator. |
| **Near-universal-fail threshold calibration** (Gap #6) — every sampled org lands below the 0.2 α floor and reads Non-Viable; only a literal wetland passes. | **R11** (gradient reframe: position + direction of travel, not FAIL/PASS; calibration caveat as an honesty line) and **R6/R14** (anchor the destination on the Fath 2019 org band). | Whether the α **Window-of-Viability bounds**, calibrated on ecological food webs, are **valid for organizational flow networks**, or need re-calibration. → formula-validator. |
| **Network-Efficiency-vs-α identity question** (Gap #9) — two exec-summary KPI cards print the same 0.066 under two labels ("Network Efficiency" and "Rel. Ascendency α"); Cred overclaim on "high resilience (R=0.223)" of a Non-Viable system. | **R2** (fix green "Non-Viable"), **R3** (un-split word, fix headings, de-leak variable names), **R1/R8** (reconcile the verdicts so the resilience/viability copy no longer contradicts). | Whether **"Network Efficiency" and α are intended to be the same quantity** (and if so, the duplicate-label presentation follows from the confirmed identity). → formula-validator. |

**Assertion: No recommendation in this roadmap alters a scientific formula.**
Verified against Part 1: every recommendation is a copy, colour, label, band-overlay,
caption, table-split, section-title, re-sequencing, render-pipeline, illustrative-anchor,
crosswalk-content, or data-pipeline change. Bands and anchors used (α 0.2–0.6,
robustness optimum ≈0.37, Fath 2019 α 0.30–0.45) are read from the existing code, not
modified. No threshold, coefficient, weighting, or equation is changed by R1–R14.

---

## Part 3 — Traceability check

Every top-10 gap from `scored-matrix.md` Section C mapped to the recommendation(s)
that address it.

| Gap (Section C) | Short name | Addressed by | Covered? |
|-----------------|------------|--------------|:--------:|
| **Gap #1** | Self-contradicting HEALTHY vs Non-Viable headline | **R1**, R8, R11 | ✅ |
| **Gap #2** | Credibility keystone (org = ecosystem) buried / app-absent | **R9** | ✅ |
| **Gap #3** | "Benchmarking" has no organizational peer basis (only wetlands) | R4, **R6**, R12, R14 | ✅ |
| **Gap #4** | Zero embedded visualizations in the PDF | **R7** | ✅ |
| **Gap #5** | Viability table compares two different scales (α vs ascendency bounds) | **R5** | ✅ |
| **Gap #6** | Near-universal "fail" / binary pass-fail framing | R6, **R11**, R14 | ✅ |
| **Gap #7** | Raw ecological telemetry, no reference band, untranslated jargon | **R4**, R8 | ✅ |
| **Gap #8** | TOC matches no section; numbering leaks | R3, **R10** | ✅ |
| **Gap #9** | Exec Summary inconsistent, un-anchored, mis-colored | **R2**, R3 | ✅ |
| **Gap #10** | ESG crosswalk is a superficial one-to-one code lookup | **R13** | ✅ |

**All 10 top gaps are covered by at least one recommendation.** None is deferred
without a recommendation: the deferred *math* questions (Part 2) are hand-offs, not
uncovered gaps — each of those gaps also has a presentation recommendation here.
(Bold = the primary recommendation for that gap; others provide reinforcing coverage.)

---

*Scope: presentation, framing, information architecture, and narrative only. No
formula, threshold, coefficient, or weighting is changed. Items with a math/calibration
root cause are handed to formula-validator (Part 2); their business framing and
presentation fixes are retained here. Traces to `scored-matrix.md` Section C,
`benchmarking-model.md` (Tier 1/2/3 model + gradient reframe), and `audit-pm.md`
(exec one-pager Q4, board-ready ranking Q5).*
