# OASIS Business Revision

**A consultant-grade diagnosis of whether OASIS's dashboards and PDF report are ready to be handed from an operator to a C-suite executive — and the prioritized plan to make them so.**

Date: 2026-07-02 · Branch: `feat/detailed-ecosystemic-report` · Scope: presentation, framing, information architecture, and narrative only. No scientific formula is changed by this review.

---

## 1. Executive Summary

**Verdict: OASIS is not consultant-ready today.** Across 42 inventoried surfaces (41 scored), the product's overall business-utility score is **2.85 / 5** — mediocre. **No single surface reaches consultant-grade (≥4.0)**, and only 8 of 41 clear 3.4. The failure is not in the science and not in *what* OASIS measures — the "Decision relevance" dimension is the healthiest column in the entire product (avg 3.4), meaning the tool is about the right things. **OASIS fails at explaining, benchmarking, and drawing those things** — the presentation layer, not the engine.

That distinction is the good news: the highest-value fixes are low-effort presentation changes, and the worst trust-killers are all shippable "this week."

**The five headline gaps:**

1. **A self-contradicting headline verdict.** The same organization reads "**Non-Viable / SUSTAINABLE CRITICAL 35/100**" *and* "**Overall Health 76/100 — HEALTHY**" (green), with three OASIS dimensions pinned at a perfect 100/100, on adjacent screens. See `evidence/dashboards/techflow-oasis-health.png`. An executive who skims the big green "HEALTHY 76" concludes the org is fine — the exact opposite of the diagnosis. A 30-second, deal-killing trust failure.
2. **The credibility keystone is buried and app-absent.** The single argument for *why* ecosystem math should grade a company is made once, competently, on PDF page 4 — and appears nowhere in the dashboards. Every downstream number inherits this unearned-authority risk.
3. **"Benchmarking" with no organizational peer basis.** The only comparators shipped are four published wetlands (Cone Spring, Florida Bay, et al.), self-disclaimed as "not organizational targets." The product's most sellable word is a promise it cannot keep.
4. **Zero visualizations in the PDF.** `pdfimages -list` confirms **zero embedded images in all three reports** — no network diagram, no Window-of-Viability curve, no radar. An ecological-flow diagnosis whose entire thesis is a picture is delivered as prose and number tables.
5. **A near-universal "fail" verdict.** Every sampled org — including one literally named "Balanced" (α = 0.095) — reads Non-Viable; only the literal wetland (Cone Spring, α = 0.577) passes. A diagnostic that fails almost everyone reads as miscalibrated and is commercially dead.

**The redesign thesis, in one sentence:** front-load interpretation, credibility, and visuals — reconcile the two verdicts into one, promote the "why this applies to you" justification to the cover, replace pass/fail with a position-on-a-gradient, and draw the pictures — because the fixes are almost entirely presentation, and the highest-value ones are low-effort.

**Three independent lenses converged.** This review ran three specialized agents in parallel — a UI/UX auditor (dashboards), a reporting auditor (PDF), and a product-management auditor (the operator→exec value chain, scoring both). Each surfaced the *same* top gaps independently: all three ranked the HEALTHY-vs-Non-Viable contradiction #1 or #2; all three flagged the missing peer benchmark and the buried credibility argument. Convergence across three methods is evidence the findings are real, not impressionistic.

The two weakest rubric dimensions tell the whole story: **Visual effectiveness 2.29** (the PDF has no pictures) and **Benchmark/context 2.49** (pervasive — red across *both* surface families). Fix those two and OASIS crosses from "academic toy" to "board deck."

---

## 2. Method & Rubric

### Scope

Two surface families are in scope: the **in-app Streamlit dashboards** and the **exported ReportLab PDF report**. The review covers presentation, framing, information architecture, and narrative only. **The scientific formulas are fixed** (per `CLAUDE.md`): where a finding's root cause is math or calibration, it is *flagged* for the `formula-validator` path and its business framing retained here — never actioned. Intervention-planning and longitudinal tracking are out of scope.

### The Business-Utility Rubric (7 dimensions)

Every surface was scored 1–5 (1 = fails badly, 3 = mediocre, 5 = consultant-grade; cells ≤2 are gaps) against seven dimensions:

| # | Dimension | The question it asks |
|---|-----------|----------------------|
| 1 | **Decision relevance** *(TIEBREAKER)* | Does this drive the "diagnose & benchmark" job, or is it data for data's sake? |
| 2 | **So-what clarity** | Is the business implication explicit, or must the user infer it? |
| 3 | **Interpretability** | Can a non-ecologist executive read it without a glossary? |
| 4 | **Benchmark / context** | Is the number shown against a reference so "good vs bad" is obvious? |
| 5 | **Credibility / defensibility** | Would a consultant stake their reputation on it? |
| 6 | **Narrative flow** | Does the story build headline → evidence → detail? |
| 7 | **Visual effectiveness** | Right chart for the message; signal over decoration. |

**Decision relevance is the tiebreaker:** where a board-facing surface fails dim 1 or dim 5, it outranks an analyst-only surface failing dim 7.

### Three-lens agent audit

Three specialized agents ran in parallel, each scoring its domain against all seven dimensions, grounded in real captured artifacts rather than memory:

| Lens | Owns | Primary focus |
|------|------|---------------|
| **UI/UX auditor** | Dashboards (D1–D21) | Visual effectiveness, interpretability, on-screen narrative |
| **Reporting auditor** | PDF report (R1–R21) | Credibility, framework alignment, executive narrative |
| **Product-management auditor** | Both | Decision relevance, so-what clarity, the operator→exec value chain |

The three audits were then reconciled into one scored matrix. **Reconciliation rule:** where the PM and domain lenses disagreed by ≥2 points on dim 1 or dim 2, the **lower** score was taken and both values footnoted — conservative, because a gap flagged by either lens is a real handoff risk.

### Three contrasting organizations

The surfaces were tested across three organizations spanning the full outcome range, so the review measures whether they communicate *across* outcomes, not just for one case:

- **TechFlow Innovations** — the unsustainable exemplar (α = 0.066, Non-Viable, "too chaotic").
- **Balanced Test Org** — designed to be balanced, yet also **Non-Viable** (α = 0.095). That a system built to be balanced still fails is itself a finding: it points to the near-universal-fail calibration issue.
- **Cone Spring Ecosystem** — the viable/green reference (α = 0.577), a literal wetland and the *only* sampled system that passes.

### Coverage

**42 surfaces inventoried** (21 dashboard + 21 report). 41 were scored; **D14** (Multi-Metric Comparison radar) was *not captured* — it sits below the fold in all three visualization screenshots — and is marked `n/c`, excluded from every average.

---

## 3. Findings

**The overall product average is 2.85 / 5.** No surface reaches consultant-grade (≥4.0); only 8 of 41 clear 3.4. The healthiest column is Decision relevance (3.4) — OASIS is *about* the right things. The two weakest dimensions are where it fails:

- **Visual effectiveness — 2.29 (weakest overall).** Driven almost entirely by the PDF: 18 of 21 report rows score Visual ≤2 because the ReportLab path embeds **zero images** (confirmed via `pdfimages -list`). "Visual is weakest" really means "the PDF has no pictures." The dashboards fare far better (only D3/D6/D7/D21 are red on Visual).
- **Benchmark/context — 2.49 (most structurally pervasive).** Unlike Visual, this is red across *both* surface families — 25 of 41 scored cells ≤2. Raw numbers appear with no good/bad band, and the one section literally named "Benchmarking" scores Bench = 1 because its only comparators are wetlands. This is the single clearest pattern in the heatmap.

### Reconciled scored matrix — column averages

| Dimension | 1 DecRel | 2 So-what | 3 Interp | 4 Bench | 5 Cred | 6 Narr | 7 Visual | **Overall** |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Column avg (41 scored)** | **3.4** | **2.8** | **2.9** | **2.5** | **2.9** | **3.1** | **2.3** | **2.85** |

Weakest surfaces overall: **R9** (1.1, the empty Visualizations section) · **R3** (1.6, a Table of Contents matching no real heading) · **D6** (2.0) · **D7** (2.1) · **D3 / D9** (tie at 2.3). Four of the five worst are the raw-ecological-telemetry blocks plus the two broken PDF front-/mid-matter surfaces.

### Gap heatmap

🟥 = cell ≤2 (gap) · 🟨 = cell = 3 · 🟩 = cell ≥4 · ⬜ = not captured.

| ID | 1 DecRel | 2 So-what | 3 Interp | 4 Bench | 5 Cred | 6 Narr | 7 Visual |
|----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| D1  | 🟩 | 🟨 | 🟨 | 🟩 | 🟩 | 🟨 | 🟩 |
| D2  | 🟩 | 🟨 | 🟨 | 🟨 | 🟨 | 🟨 | 🟩 |
| D3  | 🟨 | 🟥 | 🟥 | 🟥 | 🟩 | 🟥 | 🟥 |
| D4  | 🟩 | 🟩 | 🟨 | 🟨 | 🟨 | 🟩 | 🟨 |
| D5  | 🟥 | 🟨 | 🟥 | 🟩 | 🟩 | 🟨 | 🟨 |
| D6  | 🟥 | 🟥 | 🟥 | 🟥 | 🟨 | 🟥 | 🟥 |
| D7  | 🟥 | 🟥 | 🟥 | 🟥 | 🟨 | 🟥 | 🟥 |
| D8  | 🟨 | 🟨 | 🟨 | 🟥 | 🟨 | 🟨 | 🟨 |
| D9  | 🟥 | 🟥 | 🟥 | 🟥 | 🟨 | 🟥 | 🟨 |
| D10 | 🟨 | 🟥 | 🟨 | 🟥 | 🟨 | 🟨 | 🟨 |
| D11 | 🟨 | 🟥 | 🟨 | 🟥 | 🟨 | 🟨 | 🟨 |
| D12 | 🟨 | 🟥 | 🟨 | 🟥 | 🟨 | 🟨 | 🟩 |
| D13 | 🟩 | 🟨 | 🟨 | 🟩 | 🟩 | 🟨 | 🟩 |
| D14 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| D15 | 🟨 | 🟥 | 🟥 | 🟥 | 🟨 | 🟨 | 🟨 |
| D16 | 🟨 | 🟨 | 🟨 | 🟨 | 🟨 | 🟨 | 🟨 |
| D17 | 🟩 | 🟨 | 🟨 | 🟨 | 🟥 | 🟨 | 🟨 |
| D18 | 🟩 | 🟨 | 🟩 | 🟩 | 🟥 | 🟨 | 🟩 |
| D19 | 🟩 | 🟩 | 🟨 | 🟩 | 🟨 | 🟩 | 🟨 |
| D20 | 🟨 | 🟩 | 🟩 | 🟨 | 🟨 | 🟩 | 🟩 |
| D21 | 🟩 | 🟨 | 🟨 | 🟨 | 🟩 | 🟩 | 🟥 |
| R1  | 🟩 | 🟨 | 🟨 | 🟥 | 🟨 | 🟩 | 🟥 |
| R2  | 🟩 | 🟨 | 🟥 | 🟥 | 🟥 | 🟨 | 🟥 |
| R3  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| R4  | 🟨 | 🟩 | 🟩 | 🟨 | 🟨 | 🟩 | 🟥 |
| R5  | 🟨 | 🟨 | 🟨 | 🟩 | 🟩 | 🟩 | 🟥 |
| R6  | 🟨 | 🟥 | 🟨 | 🟥 | 🟨 | 🟨 | 🟥 |
| R7  | 🟩 | 🟨 | 🟥 | 🟥 | 🟨 | 🟨 | 🟥 |
| R8  | 🟩 | 🟨 | 🟥 | 🟥 | 🟥 | 🟨 | 🟥 |
| R9  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| R10 | 🟨 | 🟨 | 🟨 | 🟥 | 🟨 | 🟨 | 🟥 |
| R11 | 🟩 | 🟨 | 🟨 | 🟥 | 🟥 | 🟨 | 🟥 |
| R12 | 🟩 | 🟨 | 🟩 | 🟥 | 🟥 | 🟨 | 🟥 |
| R13 | 🟩 | 🟩 | 🟩 | 🟥 | 🟨 | 🟨 | 🟥 |
| R14 | 🟨 | 🟨 | 🟨 | 🟥 | 🟥 | 🟨 | 🟥 |
| R15 | 🟩 | 🟩 | 🟩 | 🟨 | 🟨 | 🟩 | 🟥 |
| R16 | 🟩 | 🟩 | 🟩 | 🟨 | 🟨 | 🟩 | 🟥 |
| R17 | 🟩 | 🟨 | 🟨 | 🟨 | 🟨 | 🟨 | 🟥 |
| R18 | 🟨 | 🟨 | 🟩 | 🟨 | 🟥 | 🟩 | 🟥 |
| R19 | 🟩 | 🟩 | 🟩 | 🟨 | 🟨 | 🟩 | 🟥 |
| R20 | 🟥 | 🟥 | 🟨 | 🟨 | 🟩 | 🟨 | 🟥 |
| R21 | 🟨 | 🟥 | 🟨 | 🟥 | 🟨 | 🟨 | 🟥 |

Two cross-cutting patterns anchor the heatmap. **Credibility (dim 5) collapses precisely on the OASIS roll-up and viability surfaces** — red on D17, D18, R2, R8, R9, R11, R12, R14, R18 — the surfaces where the self-contradiction and the α-vs-bounds scale mismatch live; it is fine where the science is quoted straight (R20 = 5, D1/D13 = 4). And the **PDF's entire Visual column is a near-solid red wall** because there are no images.

### Top 10 ranked gaps

Ranked by lowest score × surface prominence. Items whose *root* cause is math/calibration are marked **(root: formula-validator)** — the business framing stays here; the fix in this review is presentation-only.

**1. Self-contradicting headline: "Non-Viable / CRITICAL" vs "76–79/100 HEALTHY"**
Surfaces/dims: D17 (Cred 2), D18 (Cred 2), R11 (Bench 2, Cred 2), R12 (Cred 2); echoed on D21, R21/A2.
Evidence: `evidence/dashboards/techflow-oasis-health.png` and `evidence/reports/techflow-report.pdf` p.7–8 — overall "76/100 HEALTHY" (green) with OPEN/AUTONOMOUS/SYMBIOTIC pinned at 100/100, while the same org's cover, D4 banner, and appendix A2 read "Non-Viable / UNSUSTAINABLE / SUSTAINABLE 35 CRITICAL." Balanced is identical at 79/100. All three audits flagged this #1 or #2.
Consequence: An exec reads the big green "HEALTHY 76" and three perfect 100s and concludes the org is fine — the opposite of the diagnosis. Blocks the operator→exec handoff outright. **(root: formula-validator** — the roll-up weighting that lets 3×100 outvote a CRITICAL pillar, and the 46/49-labeled-HEALTHY banding, are calibration questions; the on-screen *reconciliation* is the presentation fix.**)**

**2. Credibility keystone (org = ecosystem analogy) buried on PDF p.4, absent in-app**
Surfaces/dims: R4 (the only place the analogy is argued); app-wide absence.
Evidence: PM Q1 — §1.1/§1.2 argue the ecosystem→org transfer once, competently, on page 4, after the cover, exec summary, and TOC; it appears **nowhere in the dashboards**. §4.2's org-level reference (α 0.30–0.45, Fath 2019) is stranded on page 14.
Consequence: The product's authority rests on this one leap, and a skeptical CFO's first question — "why does a swamp metric grade my company?" — has no answer they will reach. Every downstream verdict inherits the unearned-authority risk.

**3. "Benchmarking" has no organizational peer basis — only wetlands**
Surfaces/dims: R14 (Bench 1, Cred 2); mirrored on D5 (bench-in-raw-units).
Evidence: `evidence/reports/techflow-report.pdf` p.10 — the sole benchmark table is four published ecosystems (Cone Spring 0.505, Cone Spring Eutrophicated 0.529, Crystal River Creek 0.552, Florida Bay 0.367), self-disclaimed as "reference points … not organizational targets." No peer cohort, no percentile.
Consequence: "Benchmarking" is the word that sells this to a board, and the product can't keep the promise — positioning a software company against a tidal bay and then saying don't use it as a target gives the exec nothing to act on and invites ridicule.

**4. Zero embedded visualizations in the PDF**
Surfaces/dims: R9 (all seven dims = 1, Avg 1.1 — the single lowest-scoring surface); drags Visual ≤2 across R1–R21.
Evidence: `pdfimages -list` returns **zero embedded images in all three PDFs** — no network diagram, no Sankey, no Window-of-Viability curve, no OASIS radar, no gauges. The section title exists in the IA; it renders nothing.
Consequence: An ecological-flow diagnosis whose thesis *is* a picture ("your position in a window," "the shape of your flows") is delivered as prose and tables. Every "position in a window" claim must be taken on faith — the biggest single miss versus a consultant deck.

**5. The viability table compares two different scales (α vs. ascendency-unit bounds)**
Surfaces/dims: R8 (Bench 1, Cred 1 — the report's central diagnostic exhibit); recurs in R2, R18.
Evidence: `evidence/reports/techflow-report.pdf` p.6 — "Current Position (α) = 0.066" compared against "Lower Bound = 2756.558 FAIL / Upper Bound = 8269.674 PASS," i.e. a 0–1 ratio judged against bounds in the thousands, with a "FAIL lower / PASS upper" status for a system declared *below* the window. §6 (R15) quotes the same lower bound as **0.2**.
Consequence: A CFO spots in five seconds that "0.066 cannot be below 2756," and the report's most important exhibit reads as a bug — torpedoing the viability verdict. **(root: formula-validator** — units/scale correctness and the coherence of "Lower FAIL / Upper PASS" are computation questions; the fix here is not printing two scales in one table.**)**

**6. Near-universal "fail" verdict / binary pass-fail framing**
Surfaces/dims: D4, D17/R11 (verdict framing); product-wide.
Evidence: PM Q2 — every sampled org is Non-Viable/outside the window (TechFlow α 0.066, "Balanced" α 0.095), and only the literal wetland (Cone Spring, α 0.577) passes. Two designed orgs — including one built to be balanced — both fail.
Consequence: A diagnostic that tells virtually every real company "you fail" is commercially dead and reads as miscalibrated. The presentation fix is to reframe pass/fail as a *position on a gradient with a direction of travel* and surface the calibration caveat honestly. **(root: formula-validator** — whether the α bounds, calibrated on food webs, are valid for organizational networks is a calibration question; no formula change proposed here.**)**

**7. Raw ecological telemetry with no reference band and untranslated jargon**
Surfaces/dims: D6 (Avg 2.0), D7 (2.1), D3 (2.3), D9 (2.3) — all failing Bench (1–2) and So-what (1–2); R7 (Interp 2, Bench 2).
Evidence: `evidence/dashboards/techflow-core-metrics.png` — Ascendency 4.29, Overhead 0.45, AMI, α = A/C, "Effective Roles 73.00," Structural Info 0.31, Effective Link 0.06 as bare numbers with only unit micro-captions, no good/bad band. The blocks that actually explain *why* the org is unsustainable are the least legible on the page. Only the OASIS dimension expanders (D19) translate anything.
Consequence: The causal story is present in the math but invisible to the reader; an exec cannot act on "Ascendency = 4.29" and a consultant must hand-annotate every figure, violating the core "no ecology PhD" constraint.

**8. Table of Contents matches no real section; front-/mid-matter numbering leaks**
Surfaces/dims: R3 (Avg 1.6 — second-lowest; DecRel 2, So-what 1, Bench 1, Visual 1); R18/R19 heading leaks.
Evidence: `evidence/reports/techflow-report.pdf` p.3 — the TOC lists "3.1 Network Structure / 3.3 System Organization / …," none of which match the real body ("3.1 Core Network Metrics," "3.2 Sustainability Assessment," then a jump to "3.4"), and carries **no page numbers**. Section 9 contains sub-headers numbered "4.1/4.2/4.3"; Section 10 contains "5.1/5.2/5.3."
Consequence: A TOC that doesn't describe its own document, plus mis-numbered headings, are an immediate tell that the report was auto-assembled and unproofed — undermining trust before the content is read.

**9. Exec Summary is internally inconsistent, un-anchored, and mis-colored**
Surfaces/dims: R2 (Interp 2, Bench 2, Cred 2, Visual 1); D21 mirror (green up-arrows).
Evidence: `evidence/reports/techflow-report.pdf` p.2 — "Non-Viable" rendered in **green** (traffic-light failure); the word split as "Non-Viabl/e"; two KPI cards print the *same* 0.066 under two labels ("Network Efficiency" and "Rel. Ascendency α"); Balanced's summary praises "high resilience (R=0.223)" of a system it labels Non-Viable, with no reconciling sentence. In-app, D21 puts green ▲ up-arrows next to "Sub-optimal / Non-Viable."
Consequence: The one page the board reads contradicts itself and gives no visual anchor for "how bad is bad." **(Cred overclaim and the identical-0.066 labels are partly root: formula-validator** — confirm whether Network Efficiency and α are intended to be the same quantity; the green "Non-Viable" and split word are pure layout.**)**

**10. ESG crosswalk is a superficial one-to-one code lookup**
Surfaces/dims: R17 (Cred 3, Visual 2).
Evidence: `evidence/reports/techflow-report.pdf` p.13 — each OASIS dimension maps to one GRI code, one ESRS code, one TCFD pillar, with no disclosure text, data-point ID, or materiality logic; some mappings stretch (SUSTAINABLE / Window-of-Viability → GRI 201-2 climate financial implications). Caveated as "indicative … not a compliance attestation."
Consequence: For a CSRD-conscious board this is box-ticking in the buyer's own language; it will not survive a sustainability lead's review and risks an ESG-washing charge. The non-attestation caveat is doing all the credibility work.

---

## 4. Benchmarking Strategy

Benchmark/context (dim 4) is the most *structurally* pervasive gap in the product — red across both surface families (avg 2.49; 25 of 41 cells ≤2). The recommended fix is a **layered, three-tier model**, each tier honest about what it can and cannot claim. All reference values below were verified against source code, not the narrative.

### The three tiers

**Tier 1 — Theoretical norms (SHIP NOW).** Frame every headline metric against its own implemented band. The Window-of-Viability band is **α ∈ [0.2, 0.6]** (`report_intelligence.py:13–14`; `ulanowicz_calculator.py:379–380`) and the robustness optimum is **α ≈ 0.37 = 1/e** (`report_intelligence.py:15`, the exact constant `0.367879441`). This is mathematically defensible from first principles — the robustness curve R = −α·ln(α) has a single analytic maximum at 1/e — with zero data cost. Its limit: it answers "viable vs. not," never "better vs. peer." Framing rule: call this "position relative to the theoretical viability range," **never** "benchmarking."

> **Code-hygiene note (present the reconciled value on-screen):** the codebase carries the optimum as both `0.367879441` (1/e, used by the report layer) and a rounded `0.37` (inside `calculate_regenerative_capacity`); present it as **α ≈ 0.37 (= 1/e)** so both agree. Separately, the engine viability band (0.2–0.6) differs from the food-web literature band cited in prose (`latex_report_generator.py:274`: α ∈ [0.20, 0.50], Ulanowicz 2009) — the band the tool *enforces* is 0.2–0.6; reconcile the copy. These are presentation defects, not benchmarking blockers.

**Tier 2 — Reference anchors (NEAR-TERM).** Use the 22 shipped datasets (`data/ecosystem_samples/*.json`) as illustrative "you-are-here" anchors on the α line, clearly labeled cross-domain and *not* organizational targets. The critical correction:

1. **Promote the org-level α reference to the PRIMARY anchor.** "High-performing organizations: α ≈ 0.30–0.45 (Fath et al., 2019)" is **already wired into three live surfaces** — `latex_report_generator.py:275`, and the "Optimal/Warning" verdict at `pdf_generator.py:408` (exec-summary KPI card) and `pdf_generator.py:750` (core-metrics table). It already drives the on-screen verdict; it is simply never surfaced as a *named comparator* in §5, which shows only wetlands. This is the board-credible, organizational anchor the audit asks for — and it already exists in code. **This is the single most important missing element: the anchor is present in the engine but absent from the §5 benchmark table.**
2. **Demote the wetlands to a methodology footnote** — provenance for how the scale was validated in ecology, not the exec's headline comparator.
3. **Optionally add cross-domain human-system anchors** (`us_airport_network`, `manufacturing_network`, `pharma_development_network`, `dblp_coauthorship_network`) as "same math, other domains" illustration — an airport network is a more intuitive analog to an org than a marsh.

**Tier 3 — Peer cohort (DEFERRED, flagged).** This does not exist yet, and until it does the exec framing must not say "benchmarking." It would require an anonymized cohort of real orgs run through the identical OASIS pipeline, tagged by size band × sector, with honest N-gating: **N ≥ 30 per (sector × size) cell** before quoting quartiles/percentiles; **N ≥ 8–10** before even a coarse below/around/above-median band; below that, plot individual anonymized points, not a distribution. **Fake peer averages are rejected** — a fabricated benchmark manufactures unearned authority, the product's #1 risk. Better an honest "no peer basis yet" than a fake one.

### Per-metric contextualization table

Reference bands are the ones implemented in code. On-screen labels and "so-what" sentences are the recommended presentation. α = relative ascendency = A/C.

| Metric | Reference band (from code) | On-screen label | "So-what" |
|--------|----------------------------|-----------------|-----------|
| **Relative Ascendency (α = A/C)** | Viability **0.2–0.6**; robustness optimum **≈0.37 = 1/e**; **org anchor 0.30–0.45 (Fath 2019)** | "Coordination balance — α = {v} (viability 0.2–0.6; high-performing orgs 0.30–0.45; sweet spot ≈0.37)" | How much capacity is locked into fixed structure vs. kept as flexible reserve; too low = diffuse/chaotic, too high = rigid/brittle. *Honesty caveat: the 0.2–0.6 band is calibrated on food webs; validity for org networks is an open calibration question (formula-validator) — every sampled org lands below 0.2, which may be a calibration artifact.* |
| **Robustness (R)** | Peaks at α = 1/e ≈ 0.368, R_max ≈ 0.368 | "Resilience — R = {v} of a theoretical max ≈ 0.37 ({High/Moderate/Low})" | Capacity to absorb shocks without collapsing; highest when order and flexibility balance (α ≈ 0.37), so read *together with* α. *(Two R-band thresholds exist in code — reconcile to one on-screen band; presentation, not formula.)* |
| **Total System Throughput (TST)** | No theoretical band (scale quantity) | "Total activity — {v} units (scale indicator, no good/bad band)" | Gross volume of flow — a size measure, not a health verdict; it contextualizes the ratios, never itself pass/fail. |
| **AMI** | No standalone band; feeds α via A = TST·AMI | "Flow organization — {v} bits (feeds α; not judged alone)" | How constrained/organized the flow pattern is; meaningful only relative to capacity, which is what α captures — judge α, not AMI alone. |
| **Ascendency (A)** | No standalone band; judged only as A/C = α | "Organized activity — {v} (numerator of α; judge as α)" | The organized portion of activity; a raw magnitude whose health meaning comes entirely from A/C = α. *Never print A on a 0–1 α scale beside bounds in raw ascendency units (gap #5).* |
| **Development Capacity (C)** | No theoretical band; C = A + Φ | "Total capacity — {v} (the 100% that α is a fraction of)" | Total organizational potential (organized + reserve); the denominator of α — contextualize α, never pass/fail alone. |
| **OASIS Overall Score** | 0–100 composite, weighted across 5 dimensions | "Overall health — {score}/100 ({status})" | A roll-up of the five OASIS dimensions; **must be reconciled on-screen with the viability verdict** — an org reading "76/100 HEALTHY" while "Non-Viable" is a 30-second trust-killer (gap #1). Present as one headline with viability as a named sub-component. *(The masking weighting is formula-validator; the reconciliation is presentation.)* |
| **SUSTAINABLE dimension** | 0–100; `SUS = 0.30·R_norm + 0.20·W + 0.20·RC_norm + 0.30·α_opt` | "Sustainability pillar — {score}/100 (robustness + viability + α-optimality)" | Carries the viability verdict into the roll-up; 60% driven by robustness and α-optimality, so a low α pulls it down hard — this pillar should *lead* the reconciled headline, not be masked by the average. |

### The "gradient, not pass/fail" reframe

Because the α band is food-web-calibrated, essentially every real organization lands *below* it and reads "Non-Viable / FAIL" — commercially dead, and the more absurd because a literal wetland is the only "pass." The reframe presents position as a **direction of travel on a gradient**, not a binary:

- **Show the α line, mark the org's dot, name which way to move.** Render three zones — **← diffuse/chaotic (α < 0.2) · viable (0.2–0.6, sweet spot ≈0.37) · rigid/brittle (α > 0.6) →** — plot the dot, and state the vector: *"Your α is left of the viability band — coordination is diffuse. Direction of travel: add structure to move toward it."*
- **Replace FAIL/PASS words with position + move.** Same underlying number, opposite reception. `build_benchmark_view` already computes `position` ∈ {below, within, above} and `distance_to_optimum` (`report_intelligence.py:53–70`) — the data for a gradient exists; only the *rendering* is binary.
- **Anchor the destination on the org comparator** (0.30–0.45, Fath 2019), not the wetland.
- **Carry the calibration caveat as one honesty line:** *"Viability bounds are calibrated on ecological networks; treat your position as a direction of travel rather than an absolute grade (calibration for organizational networks is under review)."* This defuses the "the swamp passed and I failed" objection without touching the math.

Net effect: the section reads as "here is where you sit and which way to move," not "you fail."

---

## 5. Redesign Roadmap

Fourteen recommendations (R1–R14) across three horizons, every one traced to a top-10 gap. **Effort** = presentation-layer tweak vs. structural IA change; formula work is out of scope and never counted. **Impact** is weighted by Decision relevance (dim 1) and Credibility (dim 5) — the two dimensions on which the handoff succeeds or fails.

**Headline: the top trust-killers are all in the Immediate (low-effort) tier.** The self-contradicting verdict, the green "Non-Viable," the missing reference bands, and the promotion of the real org anchor are all copy/colour/label/table-order changes shippable "this week."

### Horizon 1 — Immediate (high-impact, low-effort)

| # | Recommendation | Traces to | Impact | Effort |
|---|----------------|-----------|:------:|--------|
| **R1** | **Reconcile the two headline verdicts into ONE.** Demote OASIS "Overall Health __/100 HEALTHY" from a co-equal headline to a *named sub-component*; let the viability/SUSTAINABLE verdict lead; relabel the banding so a Non-Viable system cannot read "HEALTHY" as its top line. | Gap #1 | **H** | Presentation (copy + label + layout order) |
| **R2** | **Fix the green "Non-Viable" → red.** Correct the traffic-light colour on the exec-summary verdict and any mirroring in-app chips/up-arrows so a failure verdict never renders in a success colour. | Gap #9 | **H** | Presentation (colour token) |
| **R3** | **Fix the "Non-Viabl/e" line-split, the mis-numbered §9/§10 headings, and leaked variable names.** Un-split the word; renumber §9 (currently "4.1/4.2/4.3") and §10 (currently "5.1/5.2/5.3"); replace leaked identifiers (`relative_ascendency`, `number_of_roles`) with human labels. | Gap #8, #9 | **H** | Presentation (text/label) |
| **R4** | **Add the α reference band + a one-line "so-what" under each headline metric** (α viability 0.2–0.6, robustness optimum ≈0.37 = 1/e, org anchor 0.30–0.45 Fath 2019), per the per-metric table in §4. Bands are read from code, not invented. | Gap #7, #3 | **H** | Presentation (band overlay + caption) |
| **R5** | **Stop printing α and ascendency-unit bounds in the same table.** Separate the 0–1 α ratio from the raw Window bounds (2756.558 / 8269.674) so the central exhibit never shows "0.066 vs 2756." Render α against the α band; unit bounds in their own labelled panel. | Gap #5 | **H** | Presentation (table split / relabel) |
| **R6** | **Promote the Fath 2019 org anchor into the §5 benchmark table; demote wetlands to a footnote.** Put "High-performing organizations: α ≈ 0.30–0.45 (Fath et al., 2019)" at the top of the exhibit (it already drives the Optimal/Warning verdict); move Cone Spring / Crystal River / Florida Bay to a scale-validation note. | Gap #3, #6 | **H** | Presentation (table content re-order) |

### Horizon 2 — Short-term (high-impact, moderate-effort)

| # | Recommendation | Traces to | Impact | Effort |
|---|----------------|-----------|:------:|--------|
| **R7** | **Embed the visualizations into the PDF.** Render the network diagram, Window-of-Viability robustness curve, OASIS radar, Sankey, and gauges into the ReportLab path so §3.3 stops rendering zero images. Each figure carries a finding caption, not a bare title. | Gap #4 | **H** | Structural IA (render pipeline into PDF) |
| **R8** | **Restructure to an exec one-pager with analyst depth gated behind a divider.** Build the 5-element one-pager: (1) one reconciled verdict + consequence; (2) 3–4 KPI cards with target anchors; (3) the captioned "you are here" WoV curve; (4) top-3 risks in Evidence→Implication form; (5) the roadmap. Demote the 12-row metrics table, extended metrics, redundant radars, and appendix A2. | Gap #7, #1 | **H** | Structural IA (re-layout + gating) |
| **R9** | **Promote the "why ecosystem math applies to your org" justification to the cover / first exec page, and add an in-app equivalent.** Lift the §1.1/§1.2 analogy off page 4, led by organizational (Fath 2019) validation, and add the same paragraph as an in-app panel where the ecological vocabulary first appears. | Gap #2 | **H** | Structural IA (content promotion + new in-app panel) |
| **R10** | **Rebuild the TOC to match real headings, with page numbers.** Regenerate the Table of Contents from the actual body headings and add page numbers. | Gap #8 | **H** | Structural IA (generated-TOC wiring) |
| **R11** | **Apply the gradient-not-pass/fail reframe to the viability verdict.** Render the α axis with three zones, plot the org's dot, state the direction of travel; replace FAIL/PASS with position + move; anchor the destination on the Fath 2019 band; carry the calibration caveat as one honesty line. Uses existing `position` / `distance_to_optimum` outputs. | Gap #6, #1 | **H** | Structural IA (gradient rendering + copy) |

### Horizon 3 — Medium-term (high-impact, higher-effort)

| # | Recommendation | Traces to | Impact | Effort |
|---|----------------|-----------|:------:|--------|
| **R12** | **Add Tier-2 reference anchors from the 22 shipped datasets as illustrative "you-are-here" positions — led by human-system networks** (`us_airport_network`, `manufacturing_network`, `pharma_development_network`, `dblp_coauthorship_network`), each labelled "illustrative reference point — not an organizational target." | Gap #3 | **M** | Higher-effort (wire runtime lookups + new exhibit) |
| **R13** | **Replace the one-to-one ESG code lookup with a finding-specific crosswalk.** For each *finding*, attach disclosure text, the relevant data-point / materiality logic, and the matching GRI/ESRS/TCFD reference; retire the stretch mappings. Keep the "indicative, not a compliance attestation" caveat. | Gap #10 | **M** | Higher-effort (finding-driven crosswalk logic) |
| **R14** | **Plan the Tier-3 anonymized peer-cohort benchmark** (data pipeline + minimum-N gating: N ≥ 30 per cell for quartiles, N ≥ 8–10 for a coarse band). Until it ships, the section stays titled "Position relative to the theoretical viability range," never "Benchmarking." | Gap #3, #6 | **H** | Higher-effort (data pipeline, cohort ingestion, percentile logic) |

**Totals: Immediate 6 · Short-term 5 · Medium-term 3 = 14 recommendations.** Every top-10 gap is covered by at least one recommendation.

### Traceability

| Gap | Short name | Addressed by |
|-----|------------|--------------|
| #1 | Self-contradicting HEALTHY vs Non-Viable | **R1**, R8, R11 |
| #2 | Credibility keystone buried / app-absent | **R9** |
| #3 | "Benchmarking" has no peer basis | R4, **R6**, R12, R14 |
| #4 | Zero embedded visualizations | **R7** |
| #5 | Viability table compares two scales | **R5** |
| #6 | Near-universal "fail" / binary framing | R6, **R11**, R14 |
| #7 | Raw telemetry, no band, untranslated jargon | **R4**, R8 |
| #8 | TOC matches no section; numbering leaks | R3, **R10** |
| #9 | Exec Summary inconsistent, un-anchored, mis-colored | **R2**, R3 |
| #10 | ESG crosswalk superficial | **R13** |

---

## 6. Appendix

### Full evidence files

| File | What it contains |
|------|------------------|
| `evidence/scored-matrix.md` | The reconciled scored matrix (41 surfaces × 7 dimensions), the gap heatmap, and the top-10 ranked gap list with per-gap evidence and business consequence. |
| `evidence/audit-uiux.md` | The dashboard (D1–D21) lens audit — authoritative for on-screen surfaces; the top-5 dashboard gaps. |
| `evidence/audit-report.md` | The PDF report (R1–R21) lens audit — authoritative for the report; the top-5 report gaps and the formula-validator hand-off list. |
| `evidence/audit-pm.md` | The operator→exec value-chain audit — decision relevance & so-what, the five strategic answers (credibility keystone, near-universal fail, benchmark basis, overload, handoff-readiness ranking). |
| `evidence/benchmarking-model.md` | The Tier 1/2/3 model, code-verified reference values, per-metric contextualization table, and the gradient reframe. |
| `evidence/roadmap.md` | The 14 recommendations (R1–R14), impact×effort horizons, the formula-guardrail check, and full traceability. |
| `evidence/surface-inventory.md` | The canonical 42-surface list with source-code references for every surface. |

### The three contrasting organizations & their artifacts

- **TechFlow Innovations** (unsustainable, α 0.066) — dashboards at `evidence/dashboards/techflow-*.png`; report at `evidence/reports/techflow-report.pdf`. The marquee contradiction is `evidence/dashboards/techflow-oasis-health.png`.
- **Balanced Test Org** (also unsustainable, α 0.095 — itself a finding) — `evidence/dashboards/balanced-*.png`; `evidence/reports/balanced-report.pdf`.
- **Cone Spring Ecosystem** (viable/green reference, α 0.577) — `evidence/dashboards/viable-cone-spring-*.png`; `evidence/reports/viable-cone-spring-report.pdf`.

### Formula-guardrail result

**No recommendation in this review alters a scientific formula.** Every R1–R14 change is a copy, colour, label, band-overlay, caption, table-split, section-title, re-sequencing, render-pipeline, illustrative-anchor, crosswalk-content, or data-pipeline change. The bands and anchors used (α 0.2–0.6, robustness optimum ≈0.37 = 1/e, Fath 2019 α 0.30–0.45) are read from existing code, not modified.

**Four math-rooted issues were handed to `formula-validator`** (business framing retained above; math not actioned here):

1. **HEALTHY-vs-Non-Viable roll-up weighting** (Gap #1) — whether the weighting should allow three 100/100 pillars to mask a CRITICAL pillar, and whether the HEALTHY banding thresholds (46/49 labeled HEALTHY) are calibrated correctly.
2. **α-vs-bounds scale / units** (Gap #5) — whether the Window bounds are computed in the right units, and whether "Lower FAIL / Upper PASS" is coherent for a below-window system (§6 quotes the lower bound as 0.2; §3.2 as 2756).
3. **Near-universal-fail threshold calibration** (Gap #6) — whether the food-web-calibrated α bounds are valid for organizational flow networks, or need re-calibration.
4. **Network-Efficiency-vs-α identity** (Gap #9) — whether "Network Efficiency" and α are intended to be the same quantity (both print 0.066 on TechFlow).

*Scope: presentation, framing, information architecture, and narrative only. No formula, threshold, coefficient, or weighting is changed by this review. All cited code values were verified against source on branch `feat/detailed-ecosystemic-report`.*
