# PM Audit — Operator→Executive Value Chain (Business Revision)

**Scope:** Both OASIS surfaces (in-app dashboards + exported PDF). Job-to-be-done: **diagnose & benchmark org health** and hand the output from an operator (consultant / sustainability lead) to a C-suite exec who trusts and acts on it *without translation*. Intervention-planning and time-tracking are out of scope.
**Lens:** Decision relevance (tiebreaker) and So-what clarity, plus explicit **value-chain / board-ready** judgment.
**Evidence sampled:** dashboards for TechFlow (red), Balanced (red), Cone Spring (green) across core-metrics / network-analysis / visualizations / oasis-health / detailed-report; PDF reports for all three (TechFlow read in full, Cone Spring + Balanced exec summaries).
**Constraint honored:** Presentation / framing / IA / narrative only. Math and threshold-calibration concerns are flagged for **formula-validator**, not changed here.

---

## 1. Surface scoring table

Scores 1–5 (5 = high). **DR** = Decision relevance (would a decision change based on this surface?). **SW** = So-what clarity (does it state what to conclude, for a non-analyst?). **Board-ready?** = could an operator paste this in front of a C-suite exec as-is.

### Dashboards (in-app)

| ID | Surface | DR | SW | Board-ready? | Note |
|----|---------|----|----|--------------|------|
| D1 | Core Metrics header + KPIs | 3 | 3 | Partial | 4 KPI cards (Efficiency/Robustness/Viability/Roles) are the strongest exec artifact in the app; but raw values (0.07) with no target need a legend. |
| D2 | Key Performance Indicators | 3 | 2 | No | Numbers without a "good/bad vs. what" anchor. |
| D3 | Ulanowicz Core Metrics (computation expander) | 2 | 1 | No | Analyst-only; formula trace. Correctly hidden in an expander. |
| D4 | Sustainability Assessment (WoV + health) | 4 | 3 | Partial | The single most decision-relevant verdict ("UNSUSTAINABLE — Too chaotic"); but one-word verdict lacks business consequence. |
| D5 | Window of Viability Bounds | 2 | 2 | No | Bounds in raw throughput units (2.76K–8.27K) mean nothing to an exec. |
| D6 | Extended Network Metrics | 2 | 1 | No | Structural Info, Trophic Depth, etc. Pure analyst payload. |
| D7 | Balance Indicators | 2 | 1 | No | Redundancy/Organization ratios; no so-what. |
| D8 | Health Assessments (5 chips) | 3 | 3 | Partial | Plain-language bands (Resilience HIGH, Efficiency LOW) — closer to exec-readable than most. |
| D9 | Network Roles & Functional Specialization | 2 | 2 | No | "Number of roles 1.33" is uninterpretable to a business reader. |
| D10 | Overall System Health (viz tab) | 2 | 2 | No | Duplicates D1 health framing. |
| D11 | Network Diagram | 2 | 2 | No | Pretty, not decision-bearing without labels/story. |
| D12 | Sankey diagram | 3 | 2 | Partial | Flow concentration is genuinely intuitive to execs *if* captioned with the finding. |
| D13 | WoV robustness curve | 3 | 2 | Partial | The "you are here on the hump" chart is the best single credibility visual; under-captioned. |
| D14 | Multi-Metric radar | 2 | 2 | No | Radar without reference shape = shape with no meaning. |
| D15 | Network Analysis (topology/centrality/community/robustness) | 2 | 1 | No | Deep analyst tab; 20+ metrics, "MODERATE" health footer is the only exec line. |
| D16 | System Health Dashboard (radar) | 2 | 2 | No | Another radar; overlaps D14/D18. |
| D17 | OASIS Org Health Assessment (overall score) | 4 | 4 | Partial | Best-designed exec artifact: big number + HEALTHY chip. **But see Q2 — the number contradicts the viability verdict.** |
| D18 | OASIS Dimension Status (radar) | 3 | 3 | Partial | Five named dimensions readable; radar redundant with D19 bars. |
| D19 | OASIS Dimension Details (gauges) | 3 | 3 | Partial | Per-dimension gauges + prose interpretations are the most translation-free content in the app. |
| D20 | OASIS Recommendations | 3 | 3 | Partial | Action-oriented, but generic ("increase structure, standardize processes") and metric-name-leaky. |
| D21 | Analysis Report tab (in-app preview/export) | 3 | 3 | Partial | Mirrors the PDF; the handoff funnel. |

### Report (PDF)

| ID | Surface | DR | SW | Board-ready? | Note |
|----|---------|----|----|--------------|------|
| R1 | Cover page | 3 | 3 | **Yes** | Clean, branded, includes headline verdict strip (Non-Viable / Robustness). Board-credible object. |
| R2 | Executive Summary | 4 | 3 | Partial | Right idea (verdict + 4 KPIs + 2 narrative lines). **Defect: "Non-Viable" is rendered in GREEN — traffic-light failure.** No business consequence stated. |
| R3 | Table of Contents | 2 | 2 | Yes | Fine; signals seriousness. |
| R4 | 1. Introduction | 3 | 4 | Yes | **The only place the ecosystem→org analogy is argued (1.1/1.2). This is the credibility keystone — see Q1.** |
| R5 | 2. Methodology | 2 | 3 | Partial | Well-written but analyst-facing; execs skip it. |
| R6 | 3. Results (header) | 2 | 2 | Partial | Container. |
| R7 | 3.1 Core Network Metrics table | 3 | 3 | Partial | 12-row table with an Interpretation column — good, but 12 rows overloads an exec (see Q4). |
| R8 | 3.2 Sustainability Assessment table | 4 | 3 | Partial | The verdict table. Bounds in raw units again (2756–8269). |
| R9 | 3.3 Visualizations | 2 | 2 | No | Referenced but thin in ReportLab path. |
| R10 | 3.4 Flow Distribution Analysis | 2 | 2 | No | Gini/CoV stats; analyst payload. |
| R11 | 4. OASIS Health Assessment | 4 | 4 | Partial | Strong table (5 dims, score, status, focus). **Same 76/100-HEALTHY-vs-Non-Viable contradiction as D17.** |
| R12 | 4.1 Dimension Interpretations | 4 | 4 | **Yes** | Best prose in the product — plain-language, per-dimension, actionable. This is the model for the whole report. |
| R13 | 4.2 OASIS Recommendations | 3 | 3 | Partial | Good structure; leaks metric variable names (`relative_ascendency`, `number_of_roles`). |
| R14 | 5. Benchmarking & Position | 3 | 3 | Partial | **Honest** ("reference points, not targets") but that honesty exposes there is NO peer benchmark — only 4 wetlands (see Q3). |
| R15 | 6. Risk & Resilience Analysis | 4 | 4 | **Yes** | Evidence→Implication structure is exactly what an exec needs. Best-formatted decision content. |
| R16 | 7. Prioritized Action Roadmap | 4 | 4 | **Yes** | Time-horizoned, impact-stated. (Note: roadmap borders on out-of-scope "intervention planning" but reads as diagnosis-to-next-step, acceptable.) |
| R17 | 8. ESG Framework Mapping | 4 | 3 | Partial | High business value (CSRD/ESRS/GRI/TCFD is the buyer's language). Labeled indicative/not-attestation — correct. Crosswalk is generic per-dimension, not finding-specific. |
| R18 | 9. Discussion | 2 | 3 | Partial | **Heading bug: numbered "4.1/4.2/4.3" under section 9.** Good limitations section, but long-form; execs won't read. |
| R19 | 10. Conclusions | 3 | 4 | Yes | Clear summary + prioritized recs. **Same heading bug (5.1/5.2/5.3 under section 10).** |
| R20 | References | 2 | 2 | Yes | Credibility signal; correctly terminal. |
| R21 | Appendix: Detailed Data | 3 | 3 | Partial | Node-level flow table is genuinely useful for the analyst; A2 duplicates verdicts already stated 3x. |

**Surfaces that FAIL the diagnose-&-benchmark job outright (present but useless or misleading):**
- **D17 / R11 (OASIS overall score)** — *actively misleading*: labels every sampled org "HEALTHY" (76/79/75) while the same org is simultaneously "Non-Viable / CRITICAL." This is the single biggest handoff hazard.
- **R2 exec summary color** — *actively misleading*: "Non-Viable" printed in green.
- **D5 / R8 bounds in raw throughput units** — present but useless to the target reader.
- **D6, D7, D9, D14, D16** — noise for the exec handoff (fine as analyst depth, but they dilute).

---

## 2. Strategic questions

### Q1 — The credibility keystone: is the "org = ecosystem" leap ever justified to a skeptical exec?

**Partially, and only in one place an exec won't reach.** The analogy is argued *only* in PDF §1.1 Engagement Context and §1.2 Theoretical Foundation ("Conventional performance metrics capture what an organization achieves…; they rarely illuminate whether the underlying network of flows… is configured for long-term viability"). That paragraph is genuinely good — it's the business rationale, stated once, competently. **But it is buried on page 4, after the cover, exec summary, and TOC, and it appears NOWHERE in the app.** A dashboard user gets ecological vocabulary (ascendency, trophic depth, "Window of Viability," a robustness hump) with zero justification for why a wetland's math governs their company.

**Business consequence:** This is the #1 business risk. The entire tool's output is only as trustworthy as this leap, and the leap is (a) invisible in the app, (b) un-signposted in the report, (c) never quantified ("decades of ecological validation" is asserted, never shown for *organizations*). A skeptical CFO's first question — "why does a swamp tell me my company is failing?" — has no answer they'll find. Everything downstream (verdicts, benchmarks, recommendations) inherits this unearned-authority problem. **Highest-leverage fix is not math; it's promoting and hardening this justification: a one-paragraph "Why this applies to your organization" on the cover/first exec page and an in-app equivalent, ideally citing organizational (not just ecological) validation.** Note §4.2 already claims "high-performing organizations analyzed using the same framework show alpha 0.30–0.45 (Fath et al., 2019)" — if that org-level evidence is real, it should be front-and-center, not on page 14.

### Q2 — The "both synthetic orgs are unsustainable" signal — does the tool tell almost everyone "you fail"?

**Yes, and worse: the tool contradicts itself about it.** Every sampled organization lands **Non-Viable / outside the Window of Viability**:
- TechFlow: α = 0.066, Non-Viable
- Balanced (literally named "Balanced"): α = 0.095, Non-Viable
- Cone Spring (a literal wetland): α = 0.577, **Viable** — the only pass.

Only the actual ecosystem passes. Two designed "organizations," including one built to be balanced, both fail. This strongly suggests the α thresholds calibrated on ecological webs don't translate to organizational flow networks — organizational matrices (dense, low-throughput-concentration) sit far below the ecological α band by construction. **[FLAGGED FOR FORMULA-VALIDATOR: calibration/validity of the Window-of-Viability α-bounds when applied to organizational networks. Do NOT change formulas here.]**

The compounding, PM-owned problem: **the two headline verdicts disagree.** The same TechFlow is "Non-Viable / UNSUSTAINABLE / SUSTAINABLE dimension CRITICAL (35/100)" *and* "OASIS Overall Health 76/100 — HEALTHY." Balanced: "Non-Viable" *and* "79/100 HEALTHY." An exec cannot act on a report that says both "you're failing" and "you're healthy" on adjacent screens.

**Business consequence:** A diagnostic that says "fail" to virtually every real company is commercially dead — either it's never wrong (so it's useless) or it's not credible. And the internal contradiction converts the second-biggest risk into an immediate trust-killer: the operator cannot hand this over, because the exec will spot the contradiction in 30 seconds and discount the whole tool. **PM actions (presentation only):** (1) reconcile the two verdicts into ONE headline number/verdict, with the other reframed as a sub-component, not a co-equal headline; (2) reframe the viability verdict away from binary pass/fail toward a *position on a gradient* ("your coordination is diffuse relative to sustainable systems; here's the direction to move") so it reads as diagnostic guidance, not a death sentence; (3) surface the calibration caveat honestly rather than shipping a near-guaranteed "fail."

### Q3 — Benchmark basis: what does an org actually get compared to?

**Only theoretical thresholds and four published wetlands. There is no peer basis.** §5 Benchmarking (R14) is admirably honest — it labels the ecosystem table "scientific reference points for the viability scale — not organizational targets." The only comparators shipped are Cone Spring (0.505), Cone Spring Eutrophicated (0.529), Crystal River Creek (0.552), Florida Bay (0.367). §9 mentions an org benchmark (α 0.30–0.45, Fath 2019) but no org appears in the benchmark *table*. So an exec asking "compared to whom?" gets: a theoretical band and some swamps.

**Business consequence:** "Benchmarking" is the word that sells this to a board, and right now it's a promise the product can't keep. Comparing a company to Florida Bay invites ridicule. **What would make it trustworthy to an exec:** (a) an anonymized peer cohort (same-sector, same-size orgs run through the same pipeline) with percentile placement — even a small seeded cohort beats zero; (b) if no peer set exists yet, *stop calling it benchmarking* in the exec framing and call it "position relative to the theoretical viability range," setting honest expectations; (c) promote the org-level α 0.30–0.45 reference into the benchmark table as the primary comparator and demote the wetlands to a methodology footnote.

### Q4 — Redundancy / overload across 21 + 21 surfaces: what's noise, what's the minimum?

**Heavy redundancy; the verdict is stated ~5 times and the health framing ~4 times.** Duplication map:
- **Viability verdict** appears in D1 (Viability card), D4 (Sustainability Assessment), R2 (exec), R8 (§3.2), R11/§4 (Sustainable dim), R15 (§6 Risk), R21/A2 (appendix). Five-plus restatements.
- **Radars:** D14 (multi-metric), D16 (system health), D18 (OASIS) — three radar charts, largely the same story.
- **Health chips/bands:** D1, D8, D10, D15 footer, D17 — overlapping "health" framings.
- **Core metrics** rendered twice in-app (primary path + secondary at app.py:3364/3388/3408) per the inventory notes.
- **Discussion (R18) + Conclusions (R19)** substantially restate each other.

**Minimum an exec actually needs (the "one-pager"):** (1) ONE reconciled headline verdict with business consequence; (2) 3–4 KPI cards with target anchors (the D1 four-card layout is the right pattern); (3) the "you are here" WoV/robustness curve (D13) with a caption; (4) top 3 risks in Evidence→Implication form (R15); (5) the prioritized roadmap (R16). Everything else is analyst depth that belongs behind a "for your analyst" divider. **The 12-row Core Metrics table (R7), extended metrics (D6), flow stats (R10), three radars, and appendix A2 are all demotable.**

### Q5 — Handoff readiness ranking (board-ready → needs an analyst)

**Board-ready as-is (paste in front of a C-suite):**
1. R12 — Dimension Interpretations (plain-language, per-dimension)
2. R15 — Risk & Resilience (Evidence→Implication)
3. R16 — Action Roadmap (time-horizoned)
4. R1 — Cover page
5. R19 — Conclusions (modulo heading bug)

**Board-ready after a small fix (caption / anchor / color):**
6. R2 Exec Summary (fix the green "Non-Viable"; add consequence)
7. D1 KPI cards (add target anchors)
8. R17 ESG crosswalk (buyer's language; make finding-specific)
9. D17/R11 OASIS score (**must reconcile with viability first — currently misleading**)
10. D13 WoV curve (add caption)

**Needs an analyst to interpret (keep, but gate):**
D3, D5, D6, D7, D9, D14, D15, D16, R7, R8 (bounds), R10, R21.

---

## 3. Top 5 highest-leverage value-chain gaps (ranked)

1. **The headline contradiction — "Non-Viable / CRITICAL" vs. "76–79/100 HEALTHY."** Two co-equal headline verdicts that disagree, on both surfaces, for every org. This is an immediate, 30-second trust-killer that blocks the handoff outright. **Fix (presentation): one reconciled headline; demote the other to a sub-component.** *(Underlying cause — every org scores "HEALTHY" regardless of viability — is also a calibration question for formula-validator.)*

2. **The credibility keystone is buried and app-absent.** The "why does ecosystem math apply to my company" justification exists only on PDF page 4 and nowhere in the app. Without it, every downstream number is unearned authority. **Fix: promote a "Why this applies to your organization" paragraph to the cover/first exec page and add an in-app equivalent; lead with org-level (not wetland) validation.**

3. **"Benchmarking" has no peer basis.** The product's most sellable word is backed only by theoretical bounds and four wetlands; comparing a company to Florida Bay is not board-credible. **Fix: introduce even a small anonymized peer cohort with percentile placement, promote the org-level α reference into the benchmark table, and otherwise stop calling it "benchmarking" in the exec framing.**

4. **Near-universal "fail" verdict + binary framing.** Both designed orgs (incl. "Balanced") read Non-Viable; only a literal wetland passes. A diagnostic that fails almost everyone is commercially non-viable and reads as miscalibrated. **Fix (presentation): reframe pass/fail as a position-on-a-gradient with a direction of travel; surface the calibration caveat honestly.** *(Threshold calibration flagged for formula-validator — no formula change here.)*

5. **Overload dilutes the decision; polish defects undercut credibility.** The verdict is restated 5+ times, three near-identical radars, a 12-row metrics table, and duplicate render paths bury the ~5 things an exec needs — while a green "Non-Viable," mis-numbered headings (§9 → "4.1", §10 → "5.1"), and leaked variable names (`relative_ascendency`) signal "draft," not "board deck." **Fix: build the 5-element exec one-pager, gate analyst depth behind a divider, and clear the presentation defects.**
