# OASIS PDF Report — Business-Utility Audit

**Auditor lens:** strategy consultant preparing to hand this to a client C-suite / board.
**Job to be done:** *diagnose & benchmark organizational health* for an executive audience, defensibly, mapped to GRI/ESRS/TCFD.
**Evidence:** three generated PDFs in `docs/business-revision/evidence/reports/` (`techflow-report.pdf`, `balanced-report.pdf`, `viable-cone-spring-report.pdf`), 17 pages each, extracted via `pdftotext -layout` and `pdfimages -list`. Surfaces R1–R21 per `surface-inventory.md`.

**Scale:** 1 = fails badly · 3 = mediocre · 5 = consultant-grade. Cells ≤2 are gaps.
**Dimensions:** 1 Decision relevance (tiebreaker) · 2 So-what clarity · 3 Interpretability · 4 Benchmark/context · 5 Credibility/defensibility (emphasized) · 6 Narrative flow · 7 Visual effectiveness.

> Scope note: this audit scores presentation/framing/IA/narrative/framework-alignment only. Anything that looks like a math/logic defect is flagged **[for formula-validator]** with no fix proposed.

---

## 1. Scoring table (one row per report surface)

| R-ID | Surface | 1 Dec.rel | 2 So-what | 3 Interp | 4 Bench | 5 Cred | 6 Narr | 7 Visual | Avg |
|------|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| R1 | Cover page + KPI strip | 4 | 3 | 3 | 2 | 3 | 4 | 2 | **3.0** |
| R2 | Executive Summary | 4 | 3 | 2 | 2 | 2 | 3 | 1 | **2.4** |
| R3 | Table of Contents | 2 | 1 | 2 | 1 | 2 | 2 | 1 | **1.6** |
| R4 | 1. Introduction | 3 | 4 | 4 | 3 | 3 | 4 | 1 | **3.1** |
| R5 | 2. Methodology | 3 | 3 | 3 | 4 | 4 | 4 | 1 | **3.1** |
| R6 | 3. Results (section shell) | 3 | 2 | 3 | 2 | 3 | 3 | 1 | **2.4** |
| R7 | 3.1 Core Network Metrics table | 4 | 3 | 2 | 2 | 3 | 3 | 2 | **2.7** |
| R8 | 3.2 Sustainability Assessment table | 5 | 3 | 2 | 1 | 1 | 3 | 2 | **2.4** |
| R9 | 3.3 Visualizations | 5 | 1 | 1 | 1 | 1 | 1 | 1 | **1.6** |
| R10 | 3.4 Flow Distribution Analysis | 3 | 3 | 3 | 2 | 3 | 3 | 1 | **2.6** |
| R11 | 4. OASIS Health Assessment | 5 | 3 | 3 | 2 | 2 | 3 | 2 | **2.9** |
| R12 | 4.1 Dimension Interpretations | 4 | 3 | 4 | 2 | 2 | 3 | 2 | **2.9** |
| R13 | 4.2 OASIS Recommendations | 4 | 4 | 4 | 2 | 3 | 3 | 2 | **3.1** |
| R14 | 5. Benchmarking & Position | 5 | 3 | 3 | 1 | 2 | 3 | 2 | **2.7** |
| R15 | 6. Risk & Resilience Analysis | 5 | 4 | 4 | 3 | 3 | 4 | 2 | **3.6** |
| R16 | 7. Prioritized Action Roadmap | 5 | 4 | 4 | 3 | 3 | 4 | 2 | **3.6** |
| R17 | 8. ESG Framework Mapping | 4 | 3 | 3 | 3 | 3 | 3 | 2 | **3.0** |
| R18 | 9. Discussion | 3 | 3 | 4 | 3 | 2 | 4 | 1 | **2.9** |
| R19 | 10. Conclusions & Recommendations | 4 | 4 | 4 | 3 | 3 | 4 | 1 | **3.3** |
| R20 | References | 2 | 2 | 3 | 3 | 5 | 3 | 1 | **2.7** |
| R21 | Appendix: Detailed Data | 3 | 2 | 3 | 2 | 3 | 3 | 2 | **2.6** |

**Report-wide average ≈ 2.8** (mediocre). No surface reaches consultant-grade (≥4.0). The strongest surfaces are the intelligence-layer sections R15/R16 (Risk, Roadmap). The weakest are R3 (TOC), R9 (Visualizations), R8/R2 (viability table + exec summary) — precisely the surfaces an exec reads first and trusts most.

---

## 2. Every score ≤3 — surface, failing dimension(s), specific evidence, business consequence

### R1 — Cover page + KPI strip (Bench 2, Visual 2)
- **Evidence:** techflow p.1 KPI strip shows `Robustness 0.179` with no scale, no "healthy range," no color. `viable-cone-spring` p.1 shows `Robustness 0.317` — a naked number an exec cannot rank. On techflow the exec-summary KPI card literally renders the word "Non-Viable" split across two lines as "Non-Viabl / e" (techflow p.2, lines 30–32), a typographic defect on the single most important verdict.
- **Business consequence:** the first impression of the deliverable carries a broken word on the headline verdict and un-benchmarked numbers — a partner would not let this leave the building. **[for formula-validator: not applicable — this is a layout defect]**

### R2 — Executive Summary (Interp 2, Bench 2, Cred 2, Visual 1)
- **Cred/Interp:** The four KPI cards read `Robustness (R) 0.179 = "Moderate"`, `Network Efficiency 0.066 = "Sub-optimal"`, `Rel. Ascendency (α) 0.066 = "Warning"` (techflow p.2). Two of the three cards are the *same number* (0.066) with two different labels ("Network Efficiency" and "Rel. Ascendency"), which looks like a copy error to a skeptical reader **[for formula-validator: confirm whether Network Efficiency and α are intended to be identical]**.
- **Cred (overclaim):** balanced p.2 says "Robustness of R = 0.223 suggests **high** resilience" while its own KPI card labels the same value the org is "Non-Viable" — the exec summary praises resilience of a system it declares non-viable, with no reconciling sentence.
- **Bench/Visual:** no window-of-viability chart, no traffic-light, no peer position. A 0–1 "Warning" on α means nothing to a CFO without the band drawn.
- **Business consequence:** the one page the board actually reads is internally inconsistent and gives no visual anchor for "how bad is bad." This is where credibility is won or lost, and it currently loses it.

### R3 — Table of Contents (Dec 2, So-what 1, Interp 2, Bench 1, Cred 2, Narr 2, Visual 1)
- **Evidence:** TOC (all three, p.3) lists `3.1 Network Structure / 3.2 Information-Theoretic Analysis / 3.3 System Organization / 3.4 Sustainability Assessment / 3.5 Resilience Metrics / 3.6 Flow Distribution`. The **actual body** has `3.1 Core Network Metrics`, `3.2 Sustainability Assessment`, then jumps to `3.4 Flow Distribution Analysis` — **there is no 3.3 in the body and none of the six TOC subsection titles match the real ones.** The TOC also has **no page numbers**.
- **Business consequence:** a TOC that does not describe the document is an immediate tell that the report was auto-assembled and not proofed. An exec who clicks/flips to "3.3 System Organization" finds nothing. Undermines trust in everything downstream.

### R6 — 3. Results shell (So-what 2, Bench 2)
- **Evidence:** section jumps 3.2 → 3.4 in the body (techflow lines 214/236); a "3.3" is referenced in TOC but never rendered. No framing sentence tells the exec what the Results section will conclude.
- **Business consequence:** the missing 3.3 reads as a production gap; the numbering discontinuity is visible to any careful reader.

### R7 — 3.1 Core Network Metrics table (Interp 2, Bench 2)
- **Evidence:** 12-row table of `AMI 0.283 bits`, `Flow Diversity 4.290 bits`, `Effective Link Density 0.055 ratio`, `Trophic Depth 1.000 levels` (techflow p.6). Interpretation column gives one-word labels ("bits", "ratio", "levels") that are units, not interpretations. No reference/target column, so the exec cannot tell whether 0.283 bits of AMI is good or bad.
- **Business consequence:** dense ecologist's table with no "good/bad" anchor; a non-technical reader skims past the actual diagnosis.

### R8 — 3.2 Sustainability Assessment table (Interp 2, Bench 1, Cred 1) — **flagged in brief, confirmed**
- **Evidence (techflow p.6):** the table prints `Current Position (α) = 0.066`, then `Lower Bound = 2756.558 FAIL`, `Upper Bound = 8269.674 PASS`. **The bounds are in raw ascendency units (thousands) while the "Current Position" they are compared against is a 0–1 ratio (0.066).** The same mismatch appears in every PDF (viable: α 0.577 vs bounds 5309.894 / 15929.681; balanced: α 0.095 vs 69.112 / 207.337). Worse, the status column reads `Lower Bound FAIL / Upper Bound PASS` — meaning the report says the org simultaneously fails the lower bound and passes the upper bound of a window it is supposedly *below*, and the exec summary elsewhere quotes those same thousands as "bounds" for a 0.066 quantity ("bounds: 2756.56–8269.67", techflow p.2/line 43).
- **Business consequence:** this is the report's central diagnostic exhibit and it compares two different scales in one table. A client's CFO will spot in five seconds that a 0.066 cannot be "below 2756." It reads as either a bug or sloppiness and torpedoes the credibility of the viability verdict — the single most important output. **[for formula-validator: confirm whether the α-vs-window comparison is a units/scale bug in the underlying computation, and whether "Lower Bound FAIL / Upper Bound PASS" is logically coherent for a system declared below the window.]**

### R9 — 3.3 Visualizations (So-what 1, Interp 1, Bench 1, Cred 1, Narr 1, Visual 1) — **major gap**
- **Evidence:** `pdfimages -list` returns **zero embedded images in all three PDFs.** There is no network diagram, no Sankey, no Window-of-Viability robustness curve, no OASIS radar, no dimension gauges — none of the inventory's dashboard visuals (D11–D14, D16, D18–D19) reach the PDF. The section exists in the IA (title "3.3 Visualizations" is in the TOC family) but renders nothing.
- **Business consequence:** an "ecological flow network" report with **no picture of the network** is a hard sell. The whole thesis (org-as-flow-network, position-in-a-window) is inherently visual, and the PDF delivers it as prose and number tables only. This is the biggest single miss versus a consultant deck, and it is why every surface scores Visual 1–2.

### R10 — 3.4 Flow Distribution Analysis (Bench 2, Visual 1)
- **Evidence:** Gini/CV/mean table (techflow p.7) followed by one interpretive sentence ("Gini 0.361 indicates moderate inequality"). No Lorenz curve, no distribution chart, no threshold for what Gini is concerning.
- **Note [for formula-validator]:** viable-cone-spring reports `Gini 0.650 = "high inequality"` and simultaneously `SYMBIOTIC 68/100 HEALTHY` and an overall Viable verdict — worth confirming the Gini→health mapping is intended.
- **Business consequence:** a data dump that most execs will skip; the one number that matters (concentration risk) is buried without a picture.

### R11 — 4. OASIS Health Assessment (Bench 2, Cred 2) — **optics contradiction confirmed**
- **Evidence (techflow p.7–8):** overall **"health score is 76/100 (HEALTHY)"** with `OPEN 100`, `AUTONOMOUS 100`, `SYMBIOTIC 100`, `INTELLIGENT 46`, `SUSTAINABLE 35 CRITICAL`. The **cover of the same report says the org is Non-Viable.** Balanced (p.7) is worse: **overall 79/100 HEALTHY**, three dimensions at 100/100, `SUSTAINABLE 46 WARNING`, cover Non-Viable. Nowhere does the report reconcile "76/100 HEALTHY overall" with "Non-Viable + SUSTAINABLE CRITICAL."
- **Cred:** three dimensions pinned at exactly 100/100 in two different synthetic orgs reads as saturated/uncalibrated to an exec **[for formula-validator: confirm the 100/100 ceiling and the 76/79 weighting are intended, and why a system with a CRITICAL sustainability pillar rolls up to "HEALTHY"]**.
- **Business consequence:** the headline optics say "healthy" while the verdict says "non-viable." An exec cannot act on a report that grades itself green and red at once; a partner would refuse to present it until the roll-up is reconciled.

### R12 — 4.1 Dimension Interpretations (Bench 2, Cred 2)
- **Evidence:** `INTELLIGENT — score 46/100 … "HEALTHY"` (techflow p.8, Table 4 status column) but the interpretation prose says "Moderate functional diversity … could be enhanced." A 46/100 labeled HEALTHY is a status/score mismatch repeated across all three reports (viable INTELLIGENT 63 HEALTHY, balanced INTELLIGENT 49 HEALTHY). **[for formula-validator: confirm the score→status thresholds; 46 and 49 rendering as HEALTHY looks like a banding error.]**
- **Business consequence:** the status chips are not trustworthy, so the reader cannot use the color-coding to triage — defeating the purpose of a dimension scorecard.

### R13 — 4.2 OASIS Recommendations (Bench 2)
- **Evidence:** recommendations are real and prioritized (techflow p.8: "CRITICAL · SUSTAINABLE · Increase structure, standardize processes"). But "Metrics to improve: relative_ascendency, robustness" prints raw variable names, and there is no baseline/target for what "improved" looks like.
- **Business consequence:** actionable in spirit but not measurable; an exec cannot set a target or track progress from "improve relative_ascendency."

### R14 — 5. Benchmarking & Position (Bench 1, Cred 2) — **peer-basis gap confirmed**
- **Evidence (techflow p.10):** the only "benchmark" table is four **published ecosystems** — Cone Spring (0.505), Cone Spring Eutrophicated (0.529), Crystal River Creek (0.552), Florida Bay (0.367). The report itself disclaims: "Published ecosystem values below are scientific reference points for the viability scale—**not organizational targets**." So the section explicitly tells the reader the only comparison set is *not* a peer basis.
- **Business consequence:** a "Benchmarking & Position" section for TechFlow Innovations that benchmarks it against a **swamp and a tidal bay** and then says "don't treat these as targets" gives the exec nothing to position against. There is no peer-organization percentile, no industry cohort — the section is credibility-neutral at best and faintly absurd at worst ("your software company scores below Florida Bay").

### R15 — 6. Risk & Resilience Analysis (Visual 2)
- **Evidence:** genuinely useful — severity-rated risks with evidence and implication (techflow p.11: "HIGH — System is chaotic … alpha 0.066 below the lower viability bound (0.2)"). Note it correctly uses **0.2** as the bound here, directly contradicting the R8 table that used **2756.558** as the lower bound for the same α. **[for formula-validator: the lower bound is quoted as 0.2 in §6 and as 2756.558 in §3.2 — same concept, two scales.]**
- **Business consequence:** strongest section, but its internal use of 0.2 exposes the R8 table error by contrast; a careful reader will notice the report can't keep its own bound consistent.

### R16 — 7. Prioritized Action Roadmap (Visual 2)
- **Evidence:** horizon-structured (0–3 / 3–9 / 9–18 months). On viable-cone-spring (p.12) two of three horizons read "No actions in this horizon" — honest, but a roadmap that is 2/3 empty looks thin, and there is no owner, cost, or effort column.
- **Business consequence:** good bones, but not yet a plan a COO could staff; "No actions in this horizon" x2 undersells the section's value.

### R17 — 8. ESG Framework Mapping (Cred 3, Visual 2) — **crosswalk depth confirmed superficial**
- **Evidence (techflow p.13):** each OASIS dimension maps to one GRI code, one ESRS code, one TCFD pillar (e.g., `OPEN → GRI 2-9/2-29 → ESRS 2 GOV/SBM → Governance`). The report labels it "**Indicative** crosswalk … for navigation and context only; **not a compliance attestation**." Mappings are plausible at the label level but there is no disclosure text, no data-point ID, no materiality logic — it is a one-to-one code lookup, not a substantive crosswalk. Some mappings are a stretch (`SUSTAINABLE (Window of Viability) → GRI 201-2 financial implications of climate change` conflates an information-theoretic balance metric with climate financial risk).
- **Business consequence:** for a CSRD-conscious board this is box-ticking; it will not survive a sustainability lead's review and could invite the charge of ESG-washing if presented as framework alignment. Defensible only because it is explicitly caveated as non-attestation — that caveat is doing all the credibility work.

### R18 — 9. Discussion (Dec 3, So-what 3, Cred 2, Visual 1)
- **Evidence:** subsections mis-numbered as **"4.1 Strategic Assessment / 4.2 Comparative Positioning / 4.3 Limitations"** inside Section **9** (techflow p.14) — a copy-paste numbering leak. Content restates the α = 0.066 outside "bounds of 2756.56 to 8269.67" (again the scale mismatch). Limitations section is genuinely good (single-point-in-time, flow-type ambiguity, boundary sensitivity, no causal claims) and is the report's most defensible passage.
- **Business consequence:** the strong limitations text is undercut by "4.x" headers sitting inside Section 9 and by re-quoting the mismatched bounds.

### R19 — 10. Conclusions (Visual 1) & numbering
- **Evidence:** subsections again mis-numbered "5.1 / 5.2 / 5.3" inside Section 10 (techflow p.15). Prose recommendations are solid and horizon-based.
- **Business consequence:** cosmetic but repeated numbering leaks reinforce the "auto-assembled, unproofed" impression.

### R20 — References (Dec 2, So-what 2, Visual 1)
- **Evidence:** clean, correctly formatted Ulanowicz/Fath/Holling citations (p.16). This is the single most credible surface (Cred 5) — the science is real and properly attributed.
- **Business consequence:** low decision-relevance for an exec but high defensibility; keep it, it is an asset.

### R21 — Appendix: Detailed Data (So-what 2, Bench 2)
- **Evidence:** node-level in/out flow table + "A2. Assessment Categories" (techflow p.17: "Sustainability UNSUSTAINABLE - Too chaotic"). Note A2 says **UNSUSTAINABLE** while the OASIS section (R11) rolled the same org up to **76/100 HEALTHY** — a third internal-consistency conflict.
- **Business consequence:** the appendix quietly contradicts the OASIS headline; anyone who reads to the back finds the report disagreeing with itself.

### Cross-cutting: Glossary absence (interpretability cost)
- **Confirmed:** no glossary appendix in any of the three PDFs. Terminal sections are R20 (References) and R21 (Appendix: Detailed Data). Terms like AMI, ascendency, overhead, trophic depth, α, "window of viability," effective link density appear with only inline first-use definitions in the methodology and are never collected. The inventory's note is correct — the glossary lives in the HTML/CSS path, not this ReportLab PDF.
- **Business consequence:** a non-ecologist exec meeting "AMI 0.283 bits" or "Trophic Depth 1.000 levels" in Table 1 has no back-of-book to consult; interpretability across R7/R10/R21 suffers.

### Cross-cutting: Domain framing / ecological→organizational analogy
- **Confirmed:** the ecosystem sample (`viable-cone-spring`) is framed identically to the orgs — cover says "ORGANIZATIONAL NETWORK ANALYSIS," Section 1.3 promises "recommendations for organizational leadership," Section 7 gives a roadmap in months, and node names are literally `Plants / Detritus / Bacteria / Detritivores / Carnivores` (p.17). The report never pauses to justify to a skeptical exec **why** a metric validated on food webs should govern a software company; Section 1.1 asserts the transfer in one sentence ("applies … principles originally developed for ecological network analysis to organizational systems") with no caveat about the strength or limits of that analogy.
- **Business consequence:** the core intellectual leap of the product is stated, never defended. A board member who asks "why should I trust a swamp metric to grade my org?" finds no answer in the report — the biggest unaddressed credibility risk after the scale-mismatch table.

---

## 3. Top 5 report gaps, ranked by business impact

1. **Viability table mixes two scales (R8) + the verdict is self-contradictory across the report.** α (0–1) is compared against bounds in raw ascendency thousands (2756–8269), with a "Lower FAIL / Upper PASS" table, while §6 quotes the bound as 0.2 and §3.2 as 2756. Combined with R11's "76/100 HEALTHY" cover-verdict "Non-Viable" and appendix "UNSUSTAINABLE," the report contradicts itself on its single most important output. A CFO spots this immediately; it is the top defensibility risk. *(Also flagged [for formula-validator] as a possible units bug.)*

2. **Zero visualizations in a report whose entire thesis is a picture (R9).** No network diagram, no Window-of-Viability curve, no OASIS radar, no gauges — `pdfimages` confirms zero images in all three PDFs. An ecological-flow-network diagnosis delivered with no network drawn is not consultant-grade and forces every "position in a window" claim to be taken on faith.

3. **OASIS roll-up optics contradict the verdict (R11/R12).** Three dimensions saturated at 100/100 and an overall "HEALTHY 76–79/100" sitting over a "Non-Viable / SUSTAINABLE CRITICAL" verdict, plus 46- and 49-point dimensions labeled "HEALTHY." The scorecard grades the org green and red simultaneously and never reconciles it. *(Score→status banding flagged [for formula-validator].)*

4. **"Benchmarking" has no organizational peer basis (R14).** The only comparators are four published ecosystems (incl. Florida Bay, a swamp), explicitly disclaimed as "not organizational targets." A benchmarking section that benchmarks a company against wetlands, then says don't use them as targets, gives the exec no position to act on.

5. **Front-matter/IA defects erode trust before the content starts (R3 + numbering leaks + broken cover word).** TOC subsection titles match none of the real body sections and carry no page numbers; body jumps 3.2→3.4; Sections 9 and 10 contain "4.x/5.x" sub-headers; the cover KPI card renders the verdict as "Non-Viabl/e." Individually cosmetic, collectively they signal an unproofed auto-assembly to exactly the audience most primed to distrust it.

---

## Items handed to formula-validator (not scored here; no fixes proposed)
- R2/R8/R18: α (0–1) compared against window bounds printed in ascendency units (thousands); §6 uses 0.2 for the same lower bound. Confirm units/scale correctness and the coherence of "Lower Bound FAIL / Upper Bound PASS" for a below-window system.
- R2: "Network Efficiency" and "Rel. Ascendency (α)" print the identical value (0.066 techflow / 0.095 balanced / 0.577 viable). Confirm they are intended to be the same quantity.
- R11/R12: dimensions scoring 46 and 49 rendered as status "HEALTHY"; three dimensions pinned at exactly 100/100 in two distinct synthetic orgs; overall "HEALTHY" roll-up despite a CRITICAL SUSTAINABLE pillar. Confirm banding thresholds and weighting.
- R21 vs R11: appendix "UNSUSTAINABLE" vs OASIS "HEALTHY 76/100" for the same org. Confirm which classifier is authoritative.
- R10: Gini 0.650 = "high inequality" coexisting with SYMBIOTIC 68 HEALTHY and an overall Viable verdict. Confirm Gini→health mapping.
